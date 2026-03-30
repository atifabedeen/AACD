"""
Agreement-Aware Correlation-Guided Distillation (AACD) model.

Extends VL2Lite's TeacherStudent with:
  * A second frozen teacher: DINOv2
  * CCA-based shared feature space between the two teachers
  * Per-sample agreement weighting
  * A condensation layer mapping student → shared CCA space

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
    student        : Namespace with .arch (e.g. 'resnet18').
    data_attributes: Dataset attributes namespace (class_num, prompt_tmpl, classes).
    shared_dim     : Dimensionality of the CCA shared space (= s).
                     Must match what CCAProjection.s produces; set via config.
    agreement_alpha: Temperature α for w = exp(-α·Δ).
    """

    def __init__(
        self,
        teacher,
        dino,
        student,
        data_attributes,
        shared_dim: int = 256,
        agreement_alpha: float = 2.0,
    ):
        super().__init__()

        # ---- CLIP teacher (frozen) -----------------------------------
        self.clip_teacher = TeacherNet(teacher)
        clip_dim = self.clip_teacher.last_features_dim   # e.g. 1024

        # ---- DINOv2 teacher (frozen) ---------------------------------
        self.dino_teacher = DINOv2Teacher(dino.model_name)
        dino_dim = self.dino_teacher.output_dim           # e.g. 384/768/…

        # ---- Student -------------------------------------------------
        self.student = StudentNet(student, data_attributes.class_num, use_teacher=True)
        student_dim = self.student.num_features           # e.g. 512 for ResNet-18

        # ---- VL2Lite alignment layers (image + text) ----------------
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

        # ---- Shared-space condensation layer -------------------------
        # Projects student features → CCA shared space for agreement-distillation
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

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x : (B, 3, H, W) input images.

        Returns
        -------
        dict with keys:
          hidden_features  (B, student_dim)  – L2-normed student repr
          logits           (B, num_classes)
          clip_img_feats   (B, clip_dim)     – normed CLIP image feats
          dino_img_feats   (B, dino_dim)     – normed DINOv2 feats
          frozen_nlp_feats (C, clip_dim)     – text embeddings
          aligned_img      (B, student_dim)  – condensed CLIP img feats
          aligned_nlp      (C, student_dim)  – condensed text feats
          student_shared   (B, shared_dim)   – student in CCA space
          shared_target    (B, shared_dim)   – CCA avg signal (detached)
          agreement_w      (B,)              – per-sample agreement weights
          delta            (B,)              – per-sample disagreement
        """
        # ---- Teacher forward (no grad) --------------------------------
        clip_img = self.clip_teacher(x)           # normed, (B, clip_dim)
        dino_img = self.dino_teacher(x)           # normed, (B, dino_dim)

        frozen_nlp = self.frozen_nlp_features.to(x.device)   # (C, clip_dim)

        # ---- VL2Lite alignment layers --------------------------------
        aligned_img = feature_norm(self.align_img(clip_img))   # (B, student_dim)
        aligned_nlp = feature_norm(self.align_nlp(frozen_nlp)) # (C, student_dim)

        # ---- Student forward -----------------------------------------
        hidden_features, logits = self.student(x)   # (B, d_s), (B, C)

        # ---- Agreement computation -----------------------------------
        if self.agreement._initialized:
            w, z_shared, delta, _, _ = self.agreement(
                clip_img.detach(), dino_img.detach()
            )
        else:
            # Fallback before agreement is initialized (e.g. sanity check)
            B = x.size(0)
            w = torch.ones(B, device=x.device)
            z_shared = torch.zeros(B, self.agreement.shared_dim, device=x.device)
            delta = torch.zeros(B, device=x.device)

        # ---- Student condensation to shared space --------------------
        student_shared = self.condensation_shared(hidden_features)   # (B, shared_dim)

        return {
            "hidden_features": hidden_features,
            "logits":          logits,
            "clip_img_feats":  clip_img,
            "dino_img_feats":  dino_img,
            "frozen_nlp_feats": frozen_nlp,
            "aligned_img":     aligned_img,
            "aligned_nlp":     aligned_nlp,
            "student_shared":  student_shared,
            "shared_target":   z_shared.detach(),
            "agreement_w":     w.detach(),
            "delta":           delta.detach(),
        }
