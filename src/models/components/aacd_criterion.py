"""
AACD combined loss function.

Total loss (proposal SS6):

  L = cls_w   * L_cls
    + kd_scale * lambda_shared * L_shared   (agreement-weighted CCA distillation)
    + kd_scale * lambda_vis    * L_vis      (symmetric visual KD from both teachers)
    + kd_scale * lambda_txt    * L_txt      (KL on CLIP text-similarity distribution)
    + lambda_geom              * L_geom     (AE-SVC geometry constraint)
    + kd_scale * lambda_feat   * L_feat     (agreement-weighted feature-wise distillation)

Dynamic weighting follows VL2Lite (SS3.5):
  - L_cls weight grows from near-0 -> 1 over training
  - KD losses scale down by 0.5 over training
  This lets the student first learn from teachers, then refine with labels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-loss: geometry preservation (AE-SVC / EDC inspired)
# ---------------------------------------------------------------------------

class GeometryPreservationLoss(nn.Module):
    """
    Encourages student embeddings to have:
      * Zero mean                (L_mean)
      * Unit variance per dim   (L_var)
      * Decorrelated dimensions (L_cov)

    Adapted from AE-SVC (Omama et al., ICLR 2025).

    Weights follow the AE-SVC paper: lambda_cov=1, lambda_var=15, lambda_mean=1.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z : (B, d) student feature vectors."""
        B, d = z.shape
        z_c = z - z.mean(dim=0, keepdim=True)         # centred
        cov = (z_c.T @ z_c) / B                        # (d, d)

        # Off-diagonal covariance -> 0  (exclude diagonal to avoid double-counting with loss_var)
        off_diag_mask = ~torch.eye(d, dtype=torch.bool, device=z.device)
        loss_cov = (cov[off_diag_mask] ** 2).sum() / d

        # Diagonal (variance) -> 1
        loss_var = ((cov.diagonal() - 1.0) ** 2).mean()

        # Mean -> 0
        loss_mean = (z.mean(dim=0) ** 2).mean()

        return 1.0 * loss_cov + 15.0 * loss_var + 1.0 * loss_mean


# ---------------------------------------------------------------------------
# Sub-loss: agreement-weighted shared-signal distillation
# ---------------------------------------------------------------------------

def agreement_shared_loss(
    student_shared: torch.Tensor,   # (B, shared_dim)
    shared_target: torch.Tensor,    # (B, shared_dim)
    w: torch.Tensor,                # (B,)
) -> torch.Tensor:
    """MSE between student and CCA shared signal, weighted by agreement."""
    per_sample = ((student_shared - shared_target) ** 2).mean(dim=1)   # (B,)
    return (w * per_sample).mean()


# ---------------------------------------------------------------------------
# Sub-loss: agreement-weighted visual KD (symmetric dual-teacher)
# ---------------------------------------------------------------------------

def agreement_visual_kd_loss(
    student_feats: torch.Tensor,    # (B, d_s)
    aligned_img: torch.Tensor,      # (B, d_s)  CLIP condensed to student dim
    aligned_dino: torch.Tensor,     # (B, d_s)  DINOv2 condensed to student dim
    w: torch.Tensor,                # (B,)
) -> torch.Tensor:
    """
    Per-sample L1 between student and each teacher's aligned features,
    averaged across teachers and weighted by agreement.
    """
    clip_l1 = (student_feats - aligned_img).abs().mean(dim=1)    # (B,)
    dino_l1 = (student_feats - aligned_dino).abs().mean(dim=1)   # (B,)
    per_sample = 0.5 * (clip_l1 + dino_l1)
    return (w * per_sample).mean()


# ---------------------------------------------------------------------------
# Sub-loss: agreement-weighted linguistic KD (VL2Lite text similarity)
# ---------------------------------------------------------------------------

def agreement_linguistic_kd_loss(
    student_feats: torch.Tensor,       # (B, d_s)
    aligned_nlp: torch.Tensor,         # (C, d_s)   condensed text feats
    clip_img_feats: torch.Tensor,      # (B, clip_dim)
    frozen_nlp_feats: torch.Tensor,    # (C, clip_dim)
    w: torch.Tensor,                   # (B,)
    temperature: float,
    logit_scale: float,
) -> torch.Tensor:
    """
    KL divergence between student and CLIP cosine-similarity distributions
    over class text embeddings, weighted by agreement.
    Mirrors VL2Lite's kd_loss (criterion.py).
    """
    C = aligned_nlp.size(0)

    # Student: cosine sim over condensed text
    student_logits = logit_scale * student_feats @ aligned_nlp.T / temperature   # (B, C)
    # Teacher: cosine sim over raw CLIP text
    teacher_logits = logit_scale * clip_img_feats @ frozen_nlp_feats.T / temperature  # (B, C)

    p_student = F.log_softmax(student_logits, dim=1)
    p_teacher = F.softmax(teacher_logits,     dim=1)

    # Per-sample KL
    kl_per_sample = F.kl_div(p_student, p_teacher, reduction="none").sum(dim=1)  # (B,)

    loss = (temperature ** 2) * (w * kl_per_sample).mean() * C / 2
    return loss


# ---------------------------------------------------------------------------
# Sub-loss: agreement-weighted feature-wise distillation (NanoSD-inspired)
# ---------------------------------------------------------------------------

def feature_wise_loss(
    projected_intermediates: list[torch.Tensor],   # list of (B, shared_dim)
    shared_target: torch.Tensor,                   # (B, shared_dim)
    w: torch.Tensor,                               # (B,)
) -> torch.Tensor:
    """
    L1 between each projected student intermediate feature and the CCA
    shared signal, weighted by agreement.  Averaged over scales.
    """
    total = torch.tensor(0.0, device=shared_target.device)
    for proj_feat in projected_intermediates:
        per_sample = (proj_feat - shared_target).abs().mean(dim=1)   # (B,)
        total = total + (w * per_sample).mean()
    return total / max(len(projected_intermediates), 1)


# ---------------------------------------------------------------------------
# Combined AACD criterion
# ---------------------------------------------------------------------------

class AACDCriterion:
    """
    Combined loss for AACD training.

    Parameters (loss weights and temperature) are set via the Hydra config.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        lambda_cls: float = 0.01,
        lambda_shared: float = 0.3,
        lambda_vis: float = 0.2,
        lambda_txt: float = 0.2,
        lambda_geom: float = 0.1,
        lambda_feat: float = 0.15,
        class_num: int = 200,
    ):
        self.temperature  = temperature
        self.lambda_cls   = lambda_cls
        self.lambda_shared = lambda_shared
        self.lambda_vis   = lambda_vis
        self.lambda_txt   = lambda_txt
        self.lambda_geom  = lambda_geom
        self.lambda_feat  = lambda_feat
        self.class_num    = class_num

        self.ce = nn.CrossEntropyLoss()
        self.geo_loss = GeometryPreservationLoss()

        logit_scale = torch.nn.Parameter(
            torch.ones([]) * float(np.log(1 / 0.07)), requires_grad=False
        )
        self.logit_scale = float(logit_scale.exp())

    def __call__(
        self,
        outputs: dict,
        labels: torch.Tensor,
        epoch: int = 0,
        max_epochs: int = 100,
    ) -> dict:
        """
        Parameters
        ----------
        outputs    : dict from AACDTeacherStudent.forward()
        labels     : (B,) ground-truth class indices
        epoch      : current training epoch (0-based)
        max_epochs : total training epochs
        """
        # ---- Dynamic weight scheduling (VL2Lite SS3.5) ---------------
        progress  = epoch / max(max_epochs, 1)
        cls_w     = self.lambda_cls + progress * (1.0 - self.lambda_cls)
        kd_scale  = 1.0 - progress * 0.5          # 1.0 -> 0.5 over training

        w = outputs["agreement_w"]   # (B,)

        # ---- 1. Classification loss ----------------------------------
        loss_cls = self.ce(outputs["logits"], labels)

        # ---- 2. Agreement-weighted shared distillation ---------------
        loss_shared = agreement_shared_loss(
            outputs["student_shared"],
            outputs["shared_target"],
            w,
        )

        # ---- 3. Agreement-weighted visual KD (symmetric, both teachers)
        loss_vis = agreement_visual_kd_loss(
            outputs["hidden_features"],
            outputs["aligned_img"],
            outputs["aligned_dino"],
            w,
        )

        # ---- 4. Agreement-weighted linguistic KD (KL, VL2Lite style) -
        loss_txt = agreement_linguistic_kd_loss(
            student_feats   = outputs["hidden_features"],
            aligned_nlp     = outputs["aligned_nlp"],
            clip_img_feats  = outputs["clip_img_feats"],
            frozen_nlp_feats= outputs["frozen_nlp_feats"],
            w               = w,
            temperature     = self.temperature,
            logit_scale     = self.logit_scale,
        )

        # ---- 5. Geometry preservation --------------------------------
        # Applied to student_shared (unconstrained CCA space) rather than
        # hidden_features (L2-normalized), because the unit-variance target
        # conflicts with L2 normalization (per-dim variance ≈ 1/d on the sphere).
        loss_geom = self.geo_loss(outputs["student_shared"])

        # ---- 6. Feature-wise distillation (NanoSD-inspired) ----------
        proj_inter = outputs.get("projected_intermediates")
        if proj_inter is not None:
            loss_feat = feature_wise_loss(
                proj_inter,
                outputs["shared_target"],
                w,
            )
        else:
            loss_feat = torch.tensor(0.0, device=loss_cls.device)

        # ---- Total ---------------------------------------------------
        total = (
            cls_w    * loss_cls
            + kd_scale * self.lambda_shared * loss_shared
            + kd_scale * self.lambda_vis    * loss_vis
            + kd_scale * self.lambda_txt    * loss_txt
            +            self.lambda_geom   * loss_geom
            + kd_scale * self.lambda_feat   * loss_feat
        )

        return {
            "total":          total,
            "cls":            loss_cls.item(),
            "shared":         loss_shared.item(),
            "vis":            loss_vis.item(),
            "txt":            loss_txt.item(),
            "geom":           loss_geom.item(),
            "feat":           loss_feat.item() if torch.is_tensor(loss_feat) else loss_feat,
            "mean_agreement": w.mean().item(),
            "mean_delta":     outputs["delta"].mean().item(),
            "cls_w":          cls_w,
            "kd_scale":       kd_scale,
        }
