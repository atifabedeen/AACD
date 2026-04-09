"""
Agreement-Aware Correlation-Guided Distillation (AACD) model (upgraded).

Changes from original:
  - AE-SVC encoders applied to frozen teacher features before CCA projection
  - ConceptBasis provides per-concept gating and correlation-weighted fusion
  - Text KD uses concept-space targets (same gates as shared KD)
  - No more discrete {1.0, 0.3, 0.0} sample-level gates
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.components.ae_svc import AE_SVC
from src.models.components.agreement import AgreementModule
from src.models.components.campus import TeacherNet, StudentNet
from src.models.components.concept_basis import ConceptBasis
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
        num_concepts: int = 128,
        agreement_alpha: float = 2.0,
        use_mobilevit: bool = False,
    ):
        super().__init__()
        self.use_mobilevit = use_mobilevit
        self.shared_dim = shared_dim
        self.num_concepts = num_concepts

        self.register_buffer('_clip_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('_clip_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.register_buffer('_imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.clip_teacher = TeacherNet(teacher)
        clip_dim = self.clip_teacher.last_features_dim
        self.dino_teacher = DINOv2Teacher(dino.model_name)
        dino_dim = self.dino_teacher.output_dim
        self.clip_feature_dim = clip_dim
        self.dino_feature_dim = dino_dim


        self.student = StudentNet(student, data_attributes.class_num, use_teacher=True)
        student_dim = self.student.num_features
        self.patch_agg = None
        self.feat_distill = None

        self.align_nlp = nn.Sequential(
            nn.Linear(clip_dim, student_dim),
            nn.ReLU(),
            nn.Linear(student_dim, student_dim),
        )
        # This branch is diagnostic only; it is not part of the AACD loss.
        self._freeze_module(self.align_nlp)
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

        self.concept_basis = ConceptBasis(
            shared_dim=shared_dim,
            num_concepts=num_concepts,
        )

        # Pre-register encoder modules so checkpoint loading can restore them
        # before AACD initialization runs in a fresh process.
        self.clip_ae_encoder = self._build_ae_encoder(clip_dim)
        self.dino_ae_encoder = self._build_ae_encoder(dino_dim)
        self._freeze_module(self.clip_ae_encoder)
        self._freeze_module(self.dino_ae_encoder)
        self.register_buffer('_clip_ae_ready', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('_dino_ae_ready', torch.tensor(False, dtype=torch.bool))

        self.data_attributes = data_attributes
        self.frozen_nlp_features = self._get_frozen_nlp_features(data_attributes)

    @staticmethod
    def _build_ae_encoder(feature_dim: int) -> nn.Module:
        return AE_SVC(feature_dim, feature_dim).encoder

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    @property
    def ae_encoders_ready(self) -> bool:
        return bool(self._clip_ae_ready.item() and self._dino_ae_ready.item())

    @property
    def initialized_for_distillation(self) -> bool:
        return self.agreement._initialized and self.ae_encoders_ready

    def set_ae_encoders(self, clip_encoder: nn.Module, dino_encoder: nn.Module):
        """Load trained AE-SVC weights into the pre-registered frozen encoders."""
        self.clip_ae_encoder.load_state_dict(clip_encoder.state_dict())
        self.dino_ae_encoder.load_state_dict(dino_encoder.state_dict())
        self._freeze_module(self.clip_ae_encoder)
        self._freeze_module(self.dino_ae_encoder)
        self._clip_ae_ready.fill_(True)
        self._dino_ae_ready.fill_(True)

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

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> dict:
        clip_x, dino_x, student_x = self.get_branch_inputs(x)
        clip_img = self.clip_teacher(clip_x)
        dino_img = self.dino_teacher(dino_x)
        frozen_nlp = self.frozen_nlp_features.to(x.device)

        aligned_nlp = feature_norm(self.align_nlp(frozen_nlp))

        projected_intermediates = None
        gap_features = None
        gap_logits = None
        patch_entropy = torch.tensor(0.0, device=x.device)


        hidden_features, logits = self.student(student_x)

        clip_feat = clip_img.detach()
        dino_feat = dino_img.detach()
        if self._clip_ae_ready.item():
            clip_feat = self.clip_ae_encoder(clip_feat)
        if self._dino_ae_ready.item():
            dino_feat = self.dino_ae_encoder(dino_feat)

        if self.agreement._initialized:
            agreement_outputs = self.agreement(clip_feat, dino_feat)
            clip_proj = agreement_outputs['clip_proj']
            dino_proj = agreement_outputs['dino_proj']
            agree_top1 = agreement_outputs['agree_top1']
        else:
            batch_size = x.size(0)
            device = x.device
            clip_proj = torch.zeros(batch_size, self.agreement.shared_dim, device=device)
            dino_proj = torch.zeros(batch_size, self.agreement.shared_dim, device=device)
            agree_top1 = torch.zeros(batch_size, dtype=torch.bool, device=device)

        z_c, z_d, per_concept_gate, shared_target = self.concept_basis(clip_proj, dino_proj)
        delta = (z_c - z_d).abs()
        mean_delta = delta.mean().item()

        student_shared = self.condensation_shared(hidden_features)
        student_concept_acts = self.concept_basis.project_to_concepts(student_shared)

        if labels is not None:
            text_feat_gt = frozen_nlp[labels]
            if self._clip_ae_ready.item():
                text_feat_gt = self.clip_ae_encoder(text_feat_gt)
            text_feat_cca = (text_feat_gt - self.agreement.mu_C) @ self.agreement.cca_A.T
            text_concept_targets = self.concept_basis.project_to_concepts(text_feat_cca)
        else:
            text_concept_targets = torch.zeros_like(student_concept_acts)

        concept_anchoring_loss = self.concept_basis.anchoring_loss()
        concept_orth_loss = self.concept_basis.orthogonality_loss()

        return {
            'hidden_features': hidden_features,
            'logits': logits,
            'clip_img_feats': clip_img,
            'dino_img_feats': dino_img,
            'frozen_nlp_feats': frozen_nlp,
            'aligned_nlp': aligned_nlp,
            'student_shared': student_shared,
            'student_concept_acts': student_concept_acts,
            'shared_target': shared_target.detach(),
            'per_concept_gate': per_concept_gate.detach(),
            'text_concept_targets': text_concept_targets.detach(),
            'agree_top1': agree_top1.detach(),
            'mean_delta': mean_delta,
            'concept_anchoring_loss': concept_anchoring_loss,
            'concept_orth_loss': concept_orth_loss,
            'projected_intermediates': projected_intermediates,
            'clip_logit_scale': self._get_clip_logit_scale(x.device),
            'patch_entropy': patch_entropy.detach(),
            'gap_features': gap_features,
            'gap_logits': gap_logits,
        }
