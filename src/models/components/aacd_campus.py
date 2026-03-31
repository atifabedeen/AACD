"""
Agreement-Aware Correlation-Guided Distillation (AACD) model.

Extends VL2Lite's TeacherStudent with:
  * A second frozen teacher: DINOv2
  * CCA-based shared feature space between the two teachers
  * Per-sample agreement weighting
  * A condensation layer mapping student -> shared CCA space
  * (Optional) MobileViT student with semantic-aware patch aggregation
  * (Optional) Feature-wise multi-scale distillation (NanoSD-inspired)

Forward output is a dict consumed by AACDCriterion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.components.campus import TeacherNet, StudentNet
from src.models.components.dino_teacher import DINOv2Teacher
from src.models.components.agreement import AgreementModule

feature_norm = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-10)


class AACDTeacherStudent(nn.Module):
    """
    Dual-teacher student model.

    Parameters
    ----------
    teacher        : Namespace with .arch and .pretrained (for CLIP, same as VL2Lite).
    dino           : Namespace with .model_name (e.g. 'dinov2_vits14').
    student        : Namespace with .arch (e.g. 'resnet18' or 'mobilevit_s').
    data_attributes: Dataset attributes namespace (class_num, prompt_tmpl, classes).
    shared_dim     : Dimensionality of the CCA shared space (= s).
                     Must match what CCAProjection.s produces; set via config.
    agreement_alpha: Temperature alpha for w = exp(-alpha*delta).
    use_mobilevit  : If True, use MobileViT student with patch aggregation
                     and feature-wise distillation. If False, use ResNet
                     student (original AACD behaviour).
    """

    def __init__(
        self,
        teacher,
        dino,
        student,
        data_attributes,
        shared_dim: int = 256,
        agreement_alpha: float = 2.0,
        use_mobilevit: bool = False,
    ):
        super().__init__()
        self.use_mobilevit = use_mobilevit

        # ---- CLIP teacher (frozen) -----------------------------------
        self.clip_teacher = TeacherNet(teacher)
        clip_dim = self.clip_teacher.last_features_dim   # e.g. 1024

        # ---- DINOv2 teacher (frozen) ---------------------------------
        self.dino_teacher = DINOv2Teacher(dino.model_name)
        dino_dim = self.dino_teacher.output_dim          # e.g. 384/768/...

        # ---- Student --------------------------------------------------
        if use_mobilevit:
            from src.models.components.feature_distillation import FeatureWiseDistillation
            from src.models.components.mobilevit_student import MobileViTStudent
            from src.models.components.patch_aggregation import SemanticAwareAggregation

            self.student = MobileViTStudent(
                arch=student.arch,
                num_classes=data_attributes.class_num,
            )
            student_dim = self.student.num_features

            # Semantic-aware patch aggregation (SS5.9)
            self.patch_agg = SemanticAwareAggregation(student_dim)

            # Feature-wise distillation projectors (SS5.11)
            self.feat_distill = FeatureWiseDistillation(
                student_dims=self.student.stage_dims[:-1],
                target_dim=shared_dim,
            )
        else:
            self.student = StudentNet(
                student, data_attributes.class_num, use_teacher=True,
            )
            student_dim = self.student.num_features
            self.patch_agg = None
            self.feat_distill = None

        # ---- Alignment layers (teacher -> student space) -------------
        self.align_img = nn.Sequential(
            nn.Linear(clip_dim, student_dim),
            nn.ReLU(),
            nn.Linear(student_dim, student_dim),
        )
        self.align_nlp = nn.Sequential(
            nn.Linear(clip_dim, student_dim),
            nn.ReLU(),
            nn.Linear(student_dim, student_dim),
        )
        self.align_dino = nn.Sequential(
            nn.Linear(dino_dim, student_dim),
            nn.ReLU(),
            nn.Linear(student_dim, student_dim),
        )

        # ---- Shared-space condensation layer -------------------------
        self.condensation_shared = nn.Sequential(
            nn.Linear(student_dim, student_dim),
            nn.ReLU(),
            nn.Linear(student_dim, shared_dim),
        )

        # ---- Agreement module (initialised lazily via setup) ---------
        self.agreement = AgreementModule(
            num_classes=data_attributes.class_num,
            shared_dim=shared_dim,
            alpha=agreement_alpha,
            clip_dim=clip_dim,
            dino_dim=dino_dim,
        )

        # ---- Frozen text features ------------------------------------
        self.data_attributes = data_attributes
        self.frozen_nlp_features = self._get_frozen_nlp_features(data_attributes)

    # ------------------------------------------------------------------
    def _get_frozen_nlp_features(self, attributes) -> torch.Tensor:
        """Pre-compute class text embeddings (same as VL2Lite)."""
        prompt_tmpl = attributes.prompt_tmpl
        classes_list = list(attributes.classes.values())
        tokens = self.clip_teacher.tokenizer(
            [prompt_tmpl.format(w) for w in classes_list]
        )
        nlp_feats = self.clip_teacher.encode_text(tokens).detach()
        return feature_norm(nlp_feats)

    def _get_clip_logit_scale(self, device: torch.device) -> torch.Tensor:
        """Read the frozen CLIP logit scale from the instantiated teacher."""
        logit_scale = getattr(self.clip_teacher.model, "logit_scale", None)
        if logit_scale is None:
            return torch.tensor(1.0 / 0.07, device=device)
        return logit_scale.detach().exp().to(device)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : (B, 3, H, W) input images.

        Returns
        -------
        dict with keys:
          hidden_features        (B, student_dim)  - L2-normed student repr
          logits                 (B, num_classes)
          clip_img_feats         (B, clip_dim)     - normed CLIP image feats
          dino_img_feats         (B, dino_dim)     - normed DINOv2 feats
          frozen_nlp_feats       (C, clip_dim)     - text embeddings
          aligned_img            (B, student_dim)  - condensed CLIP img feats
          aligned_dino           (B, student_dim)  - condensed DINOv2 img feats
          aligned_nlp            (C, student_dim)  - condensed text feats
          student_shared         (B, shared_dim)   - student in CCA space
          shared_target          (B, shared_dim)   - CCA avg signal (detached)
          agreement_w            (B,)              - per-sample agreement weights
          delta                  (B,)              - per-sample disagreement
          projected_intermediates list[(B, shared_dim)] or None
        """
        # ---- Teacher forward (no grad) -------------------------------
        clip_img = self.clip_teacher(x)           # normed, (B, clip_dim)
        dino_img = self.dino_teacher(x)           # normed, (B, dino_dim)

        frozen_nlp = self.frozen_nlp_features.to(x.device)   # (C, clip_dim)

        # ---- Alignment layers ----------------------------------------
        aligned_img = feature_norm(self.align_img(clip_img))      # (B, student_dim)
        aligned_nlp = feature_norm(self.align_nlp(frozen_nlp))    # (C, student_dim)
        aligned_dino = feature_norm(self.align_dino(dino_img))    # (B, student_dim)

        # ---- Student forward -----------------------------------------
        projected_intermediates = None
        gap_features = None
        gap_logits = None
        patch_entropy = torch.tensor(0.0, device=x.device)

        if self.use_mobilevit:
            patch_tokens, gap_features, gap_logits, intermediates = self.student(x)

            # Semantic-aware aggregation (SS5.9)
            aggregated, attn_weights = self.patch_agg(patch_tokens)
            hidden_features = feature_norm(aggregated)
            logits = self.student.classify(hidden_features)
            patch_entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(dim=1).mean()

            # Feature-wise distillation projections (SS5.11)
            projected_intermediates = self.feat_distill.project(intermediates)
        else:
            hidden_features, logits = self.student(x)   # (B, d_s), (B, C)

        # ---- Agreement computation -----------------------------------
        if self.agreement._initialized:
            w, z_shared, delta, _, _ = self.agreement(
                clip_img.detach(), dino_img.detach()
            )
        else:
            batch_size = x.size(0)
            w = torch.ones(batch_size, device=x.device)
            z_shared = torch.zeros(batch_size, self.agreement.shared_dim, device=x.device)
            delta = torch.zeros(batch_size, device=x.device)

        # ---- Student condensation to shared space --------------------
        student_shared = self.condensation_shared(hidden_features)   # (B, shared_dim)

        return {
            "hidden_features": hidden_features,
            "logits": logits,
            "clip_img_feats": clip_img,
            "dino_img_feats": dino_img,
            "frozen_nlp_feats": frozen_nlp,
            "aligned_img": aligned_img,
            "aligned_dino": aligned_dino,
            "aligned_nlp": aligned_nlp,
            "student_shared": student_shared,
            "shared_target": z_shared.detach(),
            "agreement_w": w.detach(),
            "delta": delta.detach(),
            "projected_intermediates": projected_intermediates,
            "clip_logit_scale": self._get_clip_logit_scale(x.device),
            "patch_entropy": patch_entropy.detach(),
            "gap_features": gap_features,
            "gap_logits": gap_logits,
        }
