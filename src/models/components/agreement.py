"""
Agreement-Aware weighting module.

Given per-sample CLIP and DINOv2 features projected into the shared CCA
space, this module measures how much the two teachers agree on each
sample's class assignment and returns a per-sample weight in (0, 1].

Algorithm (proposal §5.5–5.7)
------------------------------
1. Project both teacher features into the shared CCA space.
2. Compute cosine similarity to each class prototype → S_clip, S_dino.
3. Measure disagreement:  Δ_i = ‖S_clip(i,:) − S_dino(i,:)‖₂
4. Compute agreement weight:  w_i = exp(−α · Δ_i)
5. Shared distillation target:  z_i = ½(proj_clip + proj_dino)

Initialization (call once before training)
------------------------------------------
After fitting CCAProjection on training features, call:

    module.initialize(cca, clip_feats_np, dino_feats_np, labels_np)

This stores A, B as plain tensors and builds per-class prototypes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.cca_module import CCAProjection


class AgreementModule(nn.Module):
    """Per-sample agreement weighting between two frozen teachers."""

    def __init__(self, num_classes: int, shared_dim: int, alpha: float = 2.0):
        """
        Parameters
        ----------
        num_classes : Number of output classes.
        shared_dim  : Dimensionality of the CCA shared space (= s).
        alpha       : Temperature for agreement weight w = exp(-α·Δ).
                      Larger α → stricter filtering of disagreed samples.
        """
        super().__init__()
        self.num_classes = num_classes
        self.shared_dim = shared_dim
        self.alpha = alpha

        # Projection matrices (set by initialize(); kept as plain tensors,
        # not parameters, so they don't accumulate gradients).
        self._A: torch.Tensor | None = None   # (s, dim_c)
        self._B: torch.Tensor | None = None   # (s, dim_d)

        # Class prototypes in the shared CCA space (L2-normalised).
        self.register_buffer(
            "prototypes", torch.zeros(num_classes, shared_dim)
        )
        self._initialized: bool = False

    # ------------------------------------------------------------------
    def initialize(
        self,
        cca: CCAProjection,
        clip_feats: torch.Tensor,   # (N, dim_c)
        dino_feats: torch.Tensor,   # (N, dim_d)
        labels: torch.Tensor,       # (N,)  int64, 0-based
    ) -> None:
        """
        Fit agreement module from pre-extracted training features.

        Should be called once (e.g. in LightningModule.setup) before the
        first training step.
        """
        assert cca.fitted, "CCAProjection must be fitted before initializing AgreementModule."

        device = self.prototypes.device

        # Store projection matrices on the module's device
        A = torch.tensor(cca.A_s, dtype=torch.float32).to(device)  # (s, dim_c)
        B = torch.tensor(cca.B_s, dtype=torch.float32).to(device)  # (s, dim_d)
        self._A = A
        self._B = B

        clip_feats = clip_feats.float().to(device)
        dino_feats = dino_feats.float().to(device)
        labels = labels.long().to(device)

        # Project all training features
        clip_proj = clip_feats @ A.T   # (N, s)
        dino_proj = dino_feats @ B.T   # (N, s)
        shared = 0.5 * (clip_proj + dino_proj)  # (N, s)

        # Compute per-class mean → prototype
        prototypes = torch.zeros(self.num_classes, self.shared_dim, device=device)
        for k in range(self.num_classes):
            mask = labels == k
            if mask.sum() > 0:
                prototypes[k] = shared[mask].mean(dim=0)

        # L2-normalise prototypes
        self.prototypes.data = F.normalize(prototypes, dim=1)
        self._initialized = True
        print(
            f"[AgreementModule] Initialized  classes={self.num_classes}  "
            f"shared_dim={self.shared_dim}  alpha={self.alpha}"
        )

    # ------------------------------------------------------------------
    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "AgreementModule.initialize() must be called before forward."
            )

    # ------------------------------------------------------------------
    def forward(
        self,
        clip_feats: torch.Tensor,   # (B, dim_c)
        dino_feats: torch.Tensor,   # (B, dim_d)
    ):
        """
        Compute per-sample agreement weights and shared distillation target.

        Returns
        -------
        w          : (B,) agreement weights in (0, 1].
        z_shared   : (B, s) shared CCA signal for distillation.
        delta      : (B,) per-sample disagreement magnitude.
        clip_proj  : (B, s) CLIP projected features.
        dino_proj  : (B, s) DINOv2 projected features.
        """
        self._check_initialized()

        A = self._A.to(clip_feats.device)
        B = self._B.to(dino_feats.device)

        # Project to shared CCA space
        clip_proj = clip_feats @ A.T   # (B, s)
        dino_proj = dino_feats @ B.T   # (B, s)

        # Normalise for cosine similarity
        clip_n = F.normalize(clip_proj, dim=1)
        dino_n = F.normalize(dino_proj, dim=1)
        proto  = self.prototypes.to(clip_feats.device)   # (C, s)

        # Similarity vectors against class prototypes  (B, C)
        S_clip = clip_n @ proto.T
        S_dino = dino_n @ proto.T

        # Disagreement: L2 distance between similarity vectors
        delta = torch.norm(S_clip - S_dino, p=2, dim=1)   # (B,)

        # Agreement weight
        w = torch.exp(-self.alpha * delta)   # (B,) ∈ (0,1]

        # Shared signal (un-normalised; normalisation happens in loss if needed)
        z_shared = 0.5 * (clip_proj + dino_proj)   # (B, s)

        return w, z_shared, delta, clip_proj, dino_proj
