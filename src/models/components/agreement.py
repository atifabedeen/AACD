"""
Agreement-Aware weighting module.

Projects frozen CLIP and DINOv2 features into the shared CCA space.
Per-concept gating is delegated to the ConceptBasis module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.cca_module import CCAProjection


class AgreementModule(nn.Module):
    """Per-sample agreement analysis between two frozen teachers.

    After the upgrade, this module:
      - Projects CLIP/DINO features into CCA space (kept)
      - Builds class prototypes for diagnostics (kept)
      - Delegates per-concept gating to ConceptBasis (new)
      - No longer computes discrete {1.0, 0.3, 0.0} gates (removed)
    """

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
        self.clip_dim = clip_dim
        self.dino_dim = dino_dim

        a_shape = (shared_dim, clip_dim) if clip_dim is not None else (shared_dim, 1)
        b_shape = (shared_dim, dino_dim) if dino_dim is not None else (shared_dim, 1)
        mean_c_shape = (clip_dim,) if clip_dim is not None else (1,)
        mean_d_shape = (dino_dim,) if dino_dim is not None else (1,)
        self.register_buffer('_A', torch.zeros(a_shape))
        self.register_buffer('_B', torch.zeros(b_shape))
        self.register_buffer('_mean_c', torch.zeros(mean_c_shape))
        self.register_buffer('_mean_d', torch.zeros(mean_d_shape))
        self.register_buffer('prototypes', torch.zeros(num_classes, shared_dim))
        self.register_buffer('_initialized_flag', torch.tensor(0, dtype=torch.bool))

    # --- Public properties for external access to CCA matrices ---
    @property
    def mu_C(self) -> torch.Tensor:
        return self._mean_c

    @property
    def mu_D(self) -> torch.Tensor:
        return self._mean_d

    @property
    def cca_A(self) -> torch.Tensor:
        """CCA projection matrix for CLIP: [shared_dim, clip_dim]."""
        return self._A

    @property
    def cca_B(self) -> torch.Tensor:
        """CCA projection matrix for DINO: [shared_dim, dino_dim]."""
        return self._B

    @property
    def _initialized(self) -> bool:
        return bool(self._initialized_flag.item())

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

        self._initialized_flag.fill_(True)
        print(
            f'[AgreementModule] Initialized classes={self.num_classes} '
            f'shared_dim={self.shared_dim}'
        )

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError('AgreementModule.initialize() must be called before forward.')

    def forward(
        self,
        clip_feats: torch.Tensor,
        dino_feats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Project teacher features into CCA space.

        Returns CCA projections. Per-concept gating is handled by ConceptBasis
        in the campus forward pass.
        """
        self._check_initialized()

        clip_proj = (clip_feats - self._mean_c) @ self._A.T  # [B, shared_dim]
        dino_proj = (dino_feats - self._mean_d) @ self._B.T  # [B, shared_dim]

        # Diagnostic: top-1 agreement (for logging only)
        clip_n = F.normalize(clip_proj, dim=1, eps=1e-10)
        dino_n = F.normalize(dino_proj, dim=1, eps=1e-10)
        proto = self.prototypes.to(clip_feats.device)
        S_clip = clip_n @ proto.T
        S_dino = dino_n @ proto.T
        clip_top1 = S_clip.argmax(dim=1)
        dino_top1 = S_dino.argmax(dim=1)
        agree_top1 = clip_top1 == dino_top1

        return {
            'clip_proj': clip_proj,
            'dino_proj': dino_proj,
            'agree_top1': agree_top1,
        }
