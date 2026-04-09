"""
AACD combined loss function (upgraded).

Active training loss:
  L = cls_w * L_cls
    + kd_scale * lambda_shared * L_shared   (per-concept gated)
    + kd_scale * lambda_txt * L_txt         (unified concept-space text KD)
    + L_concept_reg                         (anchoring + orthogonality)

Removed:  geometry loss, separate CLIP-gated text KD, discrete {1,0.3,0} gates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def concept_shared_kd_loss(
    student_concept_acts: torch.Tensor,
    shared_target: torch.Tensor,
    per_concept_gate: torch.Tensor,
) -> torch.Tensor:
    """Per-concept gated shared KD.

    L_shared = (1/BK) * sum_i sum_k w^{kd}_{i,k} * ([z_hat_i]_k - [z_shared_i]_k)^2

    Args:
        student_concept_acts: student's concept activations [B, K]
        shared_target: correlation-weighted teacher fusion [B, K]
        per_concept_gate: w_{i,k} [B, K]
    """
    per_concept_error = (student_concept_acts - shared_target) ** 2  # [B, K]
    weighted = per_concept_gate * per_concept_error  # [B, K]
    return weighted.mean()


def unified_text_kd_loss(
    student_concept_acts: torch.Tensor,
    text_concept_targets: torch.Tensor,
    per_concept_gate: torch.Tensor,
) -> torch.Tensor:
    """Text KD in concept space with per-concept agreement gating.

    Instead of separate CLIP-gated text KD, project CLIP text features for
    the ground-truth class into concept space and distill there.

    Args:
        student_concept_acts: student's projection into concept space [B, K]
        text_concept_targets: CLIP text features for GT class in concept space [B, K]
        per_concept_gate: w_{i,k} from ConceptBasis [B, K]
    """
    per_concept_error = (student_concept_acts - text_concept_targets) ** 2  # [B, K]
    weighted_error = per_concept_gate * per_concept_error  # [B, K]
    return weighted_error.mean()


def feature_wise_loss(
    projected_intermediates: list[torch.Tensor],
    shared_target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    total = torch.tensor(0.0, device=shared_target.device)
    for proj_feat in projected_intermediates:
        per_sample = (proj_feat - shared_target).abs().mean(dim=1)
        total = total + (weight * per_sample).mean()
    return total / max(len(projected_intermediates), 1)


class AACDCriterion:
    def __init__(
        self,
        temperature: float = 2.0,
        lambda_cls: float = 0.01,
        lambda_shared: float = 0.3,
        lambda_txt: float = 0.2,
        lambda_anchor: float = 0.01,
        lambda_orth: float = 0.001,
        lambda_feat: float = 0.0,
        class_num: int = 200,
    ):
        self.temperature = temperature
        self.lambda_cls = lambda_cls
        self.lambda_shared = lambda_shared
        self.lambda_txt = lambda_txt
        self.lambda_anchor = lambda_anchor
        self.lambda_orth = lambda_orth
        self.lambda_feat = lambda_feat
        self.class_num = class_num
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def __call__(
        self,
        outputs: dict,
        labels: torch.Tensor,
        epoch: int = 0,
        max_epochs: int = 100,
    ) -> dict:
        progress = epoch / max(max_epochs, 1)
        cls_w = self.lambda_cls + progress * (1.0 - self.lambda_cls)
        kd_scale = 1.0 - progress * 0.5

        # Classification loss
        loss_cls = self.ce(outputs['logits'], labels)

        # Per-concept gated shared KD
        loss_shared = concept_shared_kd_loss(
            outputs['student_concept_acts'],
            outputs['shared_target'],
            outputs['per_concept_gate'],
        )

        # Unified concept-space text KD
        loss_txt = unified_text_kd_loss(
            outputs['student_concept_acts'],
            outputs['text_concept_targets'],
            outputs['per_concept_gate'],
        )

        # Concept basis regularization (replaces geometry loss)
        loss_concept_reg = (
            self.lambda_anchor * outputs['concept_anchoring_loss']
            + self.lambda_orth * outputs['concept_orth_loss']
        )

        # Feature-wise distillation (optional, for MobileViT)
        proj_inter = outputs.get('projected_intermediates')
        if proj_inter is not None and self.lambda_feat > 0:
            # Use mean of per_concept_gate as a per-sample weight for feature KD
            feat_weight = outputs['per_concept_gate'].mean(dim=1)  # [B]
            loss_feat = feature_wise_loss(
                proj_inter,
                outputs['shared_target'],
                feat_weight,
            )
        else:
            loss_feat = torch.tensor(0.0, device=loss_cls.device)

        total = (
            cls_w * loss_cls
            + kd_scale * self.lambda_shared * loss_shared
            + kd_scale * self.lambda_txt * loss_txt
            + loss_concept_reg
            + kd_scale * self.lambda_feat * loss_feat
        )

        agree = outputs['agree_top1'].float()

        return {
            'total': total,
            'cls': loss_cls.item(),
            'shared': loss_shared.item(),
            'txt': loss_txt.item(),
            'txt_raw': loss_txt.item(),
            'concept_reg': loss_concept_reg.item(),
            'feat': loss_feat.item() if torch.is_tensor(loss_feat) else loss_feat,
            'agreement_rate': agree.mean().item(),
            'mean_agreement': agree.mean().item(),
            'mean_delta': outputs['mean_delta'],
            'mean_clip_margin': 0.0,
            'mean_dino_margin': 0.0,
            'full_shared_frac': 0.0,
            'soft_shared_frac': 0.0,
            'label_only_frac': 0.0,
            'mean_text_kd_weight': 1.0,
            'cls_w': cls_w,
            'kd_scale': kd_scale,
        }
