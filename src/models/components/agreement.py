"""
Agreement-Aware weighting module.

Projects frozen CLIP and DINOv2 features into the shared CCA space,
computes per-teacher prototype similarities, and derives agreement,
confidence, and hard-gated distillation weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.cca_module import CCAProjection


class AgreementModule(nn.Module):
    """Per-sample agreement analysis between two frozen teachers."""

    def __init__(
        self,
        num_classes: int,
        shared_dim: int,
        alpha: float = 2.0,
        clip_dim: int | None = None,
        dino_dim: int | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.shared_dim = shared_dim
        self.alpha = alpha
        self.clip_dim = clip_dim
        self.dino_dim = dino_dim
        self.soft_shared_weight = 0.3

        a_shape = (shared_dim, clip_dim) if clip_dim is not None else (shared_dim, 1)
        b_shape = (shared_dim, dino_dim) if dino_dim is not None else (shared_dim, 1)
        mean_c_shape = (clip_dim,) if clip_dim is not None else (1,)
        mean_d_shape = (dino_dim,) if dino_dim is not None else (1,)
        self.register_buffer('_A', torch.zeros(a_shape))
        self.register_buffer('_B', torch.zeros(b_shape))
        self.register_buffer('_mean_c', torch.zeros(mean_c_shape))
        self.register_buffer('_mean_d', torch.zeros(mean_d_shape))
        self.register_buffer('prototypes', torch.zeros(num_classes, shared_dim))
        self.register_buffer('clip_margin_lo', torch.tensor(0.0))
        self.register_buffer('clip_margin_hi', torch.tensor(0.0))
        self.register_buffer('dino_margin_lo', torch.tensor(0.0))
        self.register_buffer('dino_margin_hi', torch.tensor(0.0))
        self.register_buffer('delta_hi', torch.tensor(0.0))
        self.register_buffer('_initialized_flag', torch.tensor(0, dtype=torch.bool))

    @property
    def _initialized(self) -> bool:
        return bool(self._initialized_flag.item())

    @staticmethod
    def _teacher_stats(similarity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        topk = torch.topk(similarity, k=min(2, similarity.size(1)), dim=1)
        top1_score = topk.values[:, 0]
        if similarity.size(1) > 1:
            top2_score = topk.values[:, 1]
        else:
            top2_score = torch.zeros_like(top1_score)
        margin = top1_score - top2_score
        top1_class = topk.indices[:, 0]
        return top1_class, top1_score, top2_score, margin

    def initialize(
        self,
        cca: CCAProjection,
        clip_feats: torch.Tensor,
        dino_feats: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        assert cca.fitted, 'CCAProjection must be fitted before initializing AgreementModule.'

        device = self.prototypes.device
        A = torch.tensor(cca.A_s, dtype=torch.float32, device=device)
        B = torch.tensor(cca.B_s, dtype=torch.float32, device=device)
        mean_c = torch.tensor(cca.mean_c, dtype=torch.float32, device=device)
        mean_d = torch.tensor(cca.mean_d, dtype=torch.float32, device=device)
        self._A.data = A
        self._B.data = B
        self._mean_c.data = mean_c
        self._mean_d.data = mean_d

        clip_feats = clip_feats.float().to(device)
        dino_feats = dino_feats.float().to(device)
        labels = labels.long().to(device)

        clip_proj = (clip_feats - mean_c) @ A.T
        dino_proj = (dino_feats - mean_d) @ B.T
        shared = 0.5 * (clip_proj + dino_proj)

        prototypes = torch.zeros(self.num_classes, self.shared_dim, device=device)
        for k in range(self.num_classes):
            mask = labels == k
            if mask.any():
                prototypes[k] = shared[mask].mean(dim=0)
        self.prototypes.data = F.normalize(prototypes, dim=1, eps=1e-10)

        clip_n = F.normalize(clip_proj, dim=1, eps=1e-10)
        dino_n = F.normalize(dino_proj, dim=1, eps=1e-10)
        S_clip = clip_n @ self.prototypes.T
        S_dino = dino_n @ self.prototypes.T
        delta = torch.norm(S_clip - S_dino, p=2, dim=1)
        _, _, _, clip_margin = self._teacher_stats(S_clip)
        _, _, _, dino_margin = self._teacher_stats(S_dino)

        self.clip_margin_lo.data = torch.quantile(clip_margin, 0.25)
        self.clip_margin_hi.data = torch.quantile(clip_margin, 0.75)
        self.dino_margin_lo.data = torch.quantile(dino_margin, 0.25)
        self.dino_margin_hi.data = torch.quantile(dino_margin, 0.75)
        self.delta_hi.data = torch.quantile(delta, 0.90)
        self._initialized_flag.fill_(True)
        print(
            f'[AgreementModule] Initialized classes={self.num_classes} '
            f'shared_dim={self.shared_dim} alpha={self.alpha}'
        )

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError('AgreementModule.initialize() must be called before forward.')

    def forward(
        self,
        clip_feats: torch.Tensor,
        dino_feats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._check_initialized()

        clip_proj = (clip_feats - self._mean_c) @ self._A.T
        dino_proj = (dino_feats - self._mean_d) @ self._B.T
        clip_n = F.normalize(clip_proj, dim=1, eps=1e-10)
        dino_n = F.normalize(dino_proj, dim=1, eps=1e-10)
        proto = self.prototypes.to(clip_feats.device)

        S_clip = clip_n @ proto.T
        S_dino = dino_n @ proto.T
        delta = torch.norm(S_clip - S_dino, p=2, dim=1)
        agreement_w = torch.exp(-self.alpha * delta)

        clip_top1, clip_top1_score, clip_top2_score, clip_margin = self._teacher_stats(S_clip)
        dino_top1, dino_top1_score, dino_top2_score, dino_margin = self._teacher_stats(S_dino)
        agree_top1 = clip_top1 == dino_top1

        strong_disagree = (~agree_top1) | (delta >= self.delta_hi)
        full_shared_kd = (
            agree_top1
            & (clip_margin >= self.clip_margin_hi)
            & (dino_margin >= self.dino_margin_hi)
            & (delta < self.delta_hi)
        )
        soft_shared_kd = agree_top1 & (delta < self.delta_hi) & (~full_shared_kd)

        kd_shared_weight = torch.zeros_like(delta)
        kd_shared_weight[soft_shared_kd] = self.soft_shared_weight
        kd_shared_weight[full_shared_kd] = 1.0
        kd_shared_weight[strong_disagree] = 0.0

        z_shared = 0.5 * (clip_proj + dino_proj)
        return {
            'agreement_w': agreement_w,
            'shared_target': z_shared,
            'delta': delta,
            'clip_proj': clip_proj,
            'dino_proj': dino_proj,
            'clip_top1': clip_top1,
            'dino_top1': dino_top1,
            'clip_top1_score': clip_top1_score,
            'clip_top2_score': clip_top2_score,
            'dino_top1_score': dino_top1_score,
            'dino_top2_score': dino_top2_score,
            'clip_margin': clip_margin,
            'dino_margin': dino_margin,
            'agree_top1': agree_top1,
            'kd_shared_weight': kd_shared_weight,
            'clip_margin_lo': self.clip_margin_lo.to(delta.device),
            'clip_margin_hi': self.clip_margin_hi.to(delta.device),
            'dino_margin_lo': self.dino_margin_lo.to(delta.device),
            'dino_margin_hi': self.dino_margin_hi.to(delta.device),
            'delta_hi': self.delta_hi.to(delta.device),
        }
