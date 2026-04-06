"""
Agreement-Aware Correlation-Guided Distillation (AACD) model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.components.agreement import AgreementModule
from src.models.components.campus import TeacherNet, StudentNet
from src.models.components.dino_teacher import DINOv2Teacher

feature_norm = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-10)


class AACDTeacherStudent(nn.Module):
    def __init__(
        self,
        teacher,
        dino,
        student,
        data_attributes,
        shared_dim: int = 128,
        agreement_alpha: float = 2.0,
        use_mobilevit: bool = False,
    ):
        super().__init__()
        self.use_mobilevit = use_mobilevit

        self.register_buffer('_clip_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('_clip_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.register_buffer('_imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.clip_teacher = TeacherNet(teacher)
        clip_dim = self.clip_teacher.last_features_dim
        self.dino_teacher = DINOv2Teacher(dino.model_name)
        dino_dim = self.dino_teacher.output_dim

        if use_mobilevit:
            from src.models.components.feature_distillation import FeatureWiseDistillation
            from src.models.components.mobilevit_student import MobileViTStudent
            from src.models.components.patch_aggregation import SemanticAwareAggregation

            self.student = MobileViTStudent(
                arch=student.arch,
                num_classes=data_attributes.class_num,
            )
            student_dim = self.student.num_features
            self.patch_agg = SemanticAwareAggregation(student_dim)
            self.feat_distill = FeatureWiseDistillation(
                student_dims=self.student.stage_dims[:-1],
                target_dim=shared_dim,
            )
        else:
            self.student = StudentNet(student, data_attributes.class_num, use_teacher=True)
            student_dim = self.student.num_features
            self.patch_agg = None
            self.feat_distill = None

        self.align_nlp = nn.Sequential(
            nn.Linear(clip_dim, student_dim),
            nn.ReLU(),
            nn.Linear(student_dim, student_dim),
        )
        self.condensation_shared = nn.Sequential(
            nn.Linear(student_dim, student_dim),
            nn.ReLU(),
            nn.Linear(student_dim, shared_dim),
        )
        self.agreement = AgreementModule(
            num_classes=data_attributes.class_num,
            shared_dim=shared_dim,
            alpha=agreement_alpha,
            clip_dim=clip_dim,
            dino_dim=dino_dim,
        )
        self.data_attributes = data_attributes
        self.frozen_nlp_features = self._get_frozen_nlp_features(data_attributes)

    def _normalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (x - mean.to(x.device)) / std.to(x.device)

    def preprocess_for_clip(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize(x, self._clip_mean, self._clip_std)

    def preprocess_for_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize(x, self._imagenet_mean, self._imagenet_std)

    def get_branch_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_x = self.preprocess_for_clip(x)
        imagenet_x = self.preprocess_for_imagenet(x)
        return clip_x, imagenet_x, imagenet_x

    def _get_frozen_nlp_features(self, attributes) -> torch.Tensor:
        prompt_tmpl = attributes.prompt_tmpl
        classes_list = list(attributes.classes.values())
        tokens = self.clip_teacher.tokenizer([prompt_tmpl.format(w) for w in classes_list])
        nlp_feats = self.clip_teacher.encode_text(tokens).detach()
        return feature_norm(nlp_feats)

    def _get_clip_logit_scale(self, device: torch.device) -> torch.Tensor:
        logit_scale = getattr(self.clip_teacher.model, 'logit_scale', None)
        if logit_scale is None:
            return torch.tensor(1.0 / 0.07, device=device)
        return logit_scale.detach().exp().to(device)

    def forward(self, x: torch.Tensor) -> dict:
        clip_x, dino_x, student_x = self.get_branch_inputs(x)
        clip_img = self.clip_teacher(clip_x)
        dino_img = self.dino_teacher(dino_x)
        frozen_nlp = self.frozen_nlp_features.to(x.device)

        aligned_nlp = feature_norm(self.align_nlp(frozen_nlp))

        projected_intermediates = None
        gap_features = None
        gap_logits = None
        patch_entropy = torch.tensor(0.0, device=x.device)

        if self.use_mobilevit:
            patch_tokens, gap_features, gap_logits, intermediates = self.student(student_x)
            aggregated, attn_weights = self.patch_agg(patch_tokens)
            hidden_features = feature_norm(aggregated)
            logits = self.student.classify(hidden_features)
            patch_entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(dim=1).mean()
            projected_intermediates = self.feat_distill.project(intermediates)
        else:
            hidden_features, logits = self.student(student_x)

        if self.agreement._initialized:
            agreement_outputs = self.agreement(clip_img.detach(), dino_img.detach())
        else:
            batch_size = x.size(0)
            device = x.device
            agreement_outputs = {
                'agreement_w': torch.ones(batch_size, device=device),
                'shared_target': torch.zeros(batch_size, self.agreement.shared_dim, device=device),
                'delta': torch.zeros(batch_size, device=device),
                'clip_proj': torch.zeros(batch_size, self.agreement.shared_dim, device=device),
                'dino_proj': torch.zeros(batch_size, self.agreement.shared_dim, device=device),
                'clip_top1': torch.zeros(batch_size, dtype=torch.long, device=device),
                'dino_top1': torch.zeros(batch_size, dtype=torch.long, device=device),
                'clip_top1_score': torch.zeros(batch_size, device=device),
                'clip_top2_score': torch.zeros(batch_size, device=device),
                'dino_top1_score': torch.zeros(batch_size, device=device),
                'dino_top2_score': torch.zeros(batch_size, device=device),
                'clip_margin': torch.zeros(batch_size, device=device),
                'dino_margin': torch.zeros(batch_size, device=device),
                'agree_top1': torch.zeros(batch_size, dtype=torch.bool, device=device),
                'kd_shared_weight': torch.ones(batch_size, device=device),
                'clip_margin_lo': torch.tensor(0.0, device=device),
                'clip_margin_hi': torch.tensor(0.0, device=device),
                'dino_margin_lo': torch.tensor(0.0, device=device),
                'dino_margin_hi': torch.tensor(0.0, device=device),
                'delta_hi': torch.tensor(0.0, device=device),
            }

        student_shared = self.condensation_shared(hidden_features)
        return {
            'hidden_features': hidden_features,
            'logits': logits,
            'clip_img_feats': clip_img,
            'dino_img_feats': dino_img,
            'frozen_nlp_feats': frozen_nlp,
            'aligned_nlp': aligned_nlp,
            'student_shared': student_shared,
            'shared_target': agreement_outputs['shared_target'].detach(),
            'agreement_w': agreement_outputs['agreement_w'].detach(),
            'kd_shared_weight': agreement_outputs['kd_shared_weight'].detach(),
            'delta': agreement_outputs['delta'].detach(),
            'clip_top1': agreement_outputs['clip_top1'].detach(),
            'dino_top1': agreement_outputs['dino_top1'].detach(),
            'clip_margin': agreement_outputs['clip_margin'].detach(),
            'dino_margin': agreement_outputs['dino_margin'].detach(),
            'agree_top1': agreement_outputs['agree_top1'].detach(),
            'clip_margin_lo': agreement_outputs['clip_margin_lo'].detach(),
            'clip_margin_hi': agreement_outputs['clip_margin_hi'].detach(),
            'dino_margin_lo': agreement_outputs['dino_margin_lo'].detach(),
            'dino_margin_hi': agreement_outputs['dino_margin_hi'].detach(),
            'delta_hi': agreement_outputs['delta_hi'].detach(),
            'projected_intermediates': projected_intermediates,
            'clip_logit_scale': self._get_clip_logit_scale(x.device),
            'patch_entropy': patch_entropy.detach(),
            'gap_features': gap_features,
            'gap_logits': gap_logits,
        }
