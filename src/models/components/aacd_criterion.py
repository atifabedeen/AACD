"""
AACD combined loss function.

Active training loss:
  L = cls_w * L_cls
    + kd_scale * lambda_shared * L_shared
    + kd_scale * lambda_txt * L_txt_gated
    + lambda_geom * L_geom
    + kd_scale * lambda_feat * L_feat
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryPreservationLoss(nn.Module):
    """Mean/variance/covariance regularizer on shared student features."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, d = z.shape
        z_c = z - z.mean(dim=0, keepdim=True)
        cov = (z_c.T @ z_c) / B

        off_diag_mask = ~torch.eye(d, dtype=torch.bool, device=z.device)
        loss_cov = (cov[off_diag_mask] ** 2).sum() / d
        loss_var = ((cov.diagonal() - 1.0) ** 2).mean()
        loss_mean = (z.mean(dim=0) ** 2).mean()
        return 1.0 * loss_cov + 15.0 * loss_var + 1.0 * loss_mean


def agreement_shared_loss(
    student_shared: torch.Tensor,
    shared_target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    per_sample = ((student_shared - shared_target) ** 2).mean(dim=1)
    return (weight * per_sample).mean()


def agreement_linguistic_kd_loss(
    student_feats: torch.Tensor,
    aligned_nlp: torch.Tensor,
    clip_img_feats: torch.Tensor,
    frozen_nlp_feats: torch.Tensor,
    sample_weight: torch.Tensor | None,
    temperature: float,
    logit_scale: float,
) -> torch.Tensor:
    student_logits = logit_scale * student_feats @ aligned_nlp.T / temperature
    teacher_logits = logit_scale * clip_img_feats @ frozen_nlp_feats.T / temperature

    p_student = F.log_softmax(student_logits, dim=1)
    p_teacher = F.softmax(teacher_logits, dim=1)
    kl_per_sample = F.kl_div(p_student, p_teacher, reduction='none').sum(dim=1)

    if sample_weight is None:
        reduced = kl_per_sample.mean()
    else:
        reduced = (sample_weight * kl_per_sample).mean()
    return (temperature ** 2) * reduced


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
        lambda_geom: float = 0.1,
        lambda_feat: float = 0.15,
        class_num: int = 200,
    ):
        self.temperature = temperature
        self.lambda_cls = lambda_cls
        self.lambda_shared = lambda_shared
        self.lambda_txt = lambda_txt
        self.lambda_geom = lambda_geom
        self.lambda_feat = lambda_feat
        self.class_num = class_num
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.geo_loss = GeometryPreservationLoss()
        self.logit_scale = float(1.0 / 0.07)

    @staticmethod
    def _text_kd_weight(outputs: dict, labels: torch.Tensor) -> torch.Tensor:
        clip_top1 = outputs['clip_top1']
        clip_margin = outputs['clip_margin']
        clip_margin_lo = outputs['clip_margin_lo']
        clip_margin_hi = outputs['clip_margin_hi']

        sample_weight = torch.zeros_like(clip_margin)
        label_match = clip_top1.eq(labels)
        full = label_match & (clip_margin >= clip_margin_hi)
        soft = label_match & (clip_margin >= clip_margin_lo) & (clip_margin < clip_margin_hi)
        sample_weight[soft] = 0.5
        sample_weight[full] = 1.0
        return sample_weight

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

        shared_weight = outputs['kd_shared_weight']
        text_weight = self._text_kd_weight(outputs, labels)
        logit_scale = outputs.get('clip_logit_scale', self.logit_scale)
        if torch.is_tensor(logit_scale):
            logit_scale = float(logit_scale.detach().item())

        loss_cls = self.ce(outputs['logits'], labels)
        loss_shared = agreement_shared_loss(
            outputs['student_shared'],
            outputs['shared_target'],
            shared_weight,
        )
        loss_txt_raw = agreement_linguistic_kd_loss(
            student_feats=outputs['hidden_features'],
            aligned_nlp=outputs['aligned_nlp'],
            clip_img_feats=outputs['clip_img_feats'],
            frozen_nlp_feats=outputs['frozen_nlp_feats'],
            sample_weight=None,
            temperature=self.temperature,
            logit_scale=logit_scale,
        )
        loss_txt = agreement_linguistic_kd_loss(
            student_feats=outputs['hidden_features'],
            aligned_nlp=outputs['aligned_nlp'],
            clip_img_feats=outputs['clip_img_feats'],
            frozen_nlp_feats=outputs['frozen_nlp_feats'],
            sample_weight=text_weight,
            temperature=self.temperature,
            logit_scale=logit_scale,
        )
        loss_geom = self.geo_loss(outputs['student_shared'])

        proj_inter = outputs.get('projected_intermediates')
        if proj_inter is not None:
            loss_feat = feature_wise_loss(
                proj_inter,
                outputs['shared_target'],
                shared_weight,
            )
        else:
            loss_feat = torch.tensor(0.0, device=loss_cls.device)

        total = (
            cls_w * loss_cls
            + kd_scale * self.lambda_shared * loss_shared
            + kd_scale * self.lambda_txt * loss_txt
            + self.lambda_geom * loss_geom
            + kd_scale * self.lambda_feat * loss_feat
        )

        full_shared = (shared_weight == 1.0).float()
        soft_shared = ((shared_weight > 0.0) & (shared_weight < 1.0)).float()
        label_only = (shared_weight == 0.0).float()
        agree = outputs['agree_top1'].float()

        return {
            'total': total,
            'cls': loss_cls.item(),
            'shared': loss_shared.item(),
            'txt': loss_txt.item(),
            'txt_raw': loss_txt_raw.item(),
            'geom': loss_geom.item(),
            'feat': loss_feat.item() if torch.is_tensor(loss_feat) else loss_feat,
            'agreement_rate': agree.mean().item(),
            'mean_agreement': agree.mean().item(),
            'mean_delta': outputs['delta'].mean().item(),
            'mean_clip_margin': outputs['clip_margin'].mean().item(),
            'mean_dino_margin': outputs['dino_margin'].mean().item(),
            'full_shared_frac': full_shared.mean().item(),
            'soft_shared_frac': soft_shared.mean().item(),
            'label_only_frac': label_only.mean().item(),
            'mean_text_kd_weight': text_weight.mean().item(),
            'cls_w': cls_w,
            'kd_scale': kd_scale,
        }
