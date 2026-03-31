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

This stores A, B and the training-set means as registered buffers
and builds per-class prototypes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.cca_module import CCAProjection


class AgreementModule(nn.Module):
    """Per-sample agreement weighting between two frozen teachers."""

    def __init__(
        self,
        num_classes: int,
        shared_dim: int,
        alpha: float = 2.0,
        clip_dim: int | None = None,
        dino_dim: int | None = None,
    ):
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
        self.clip_dim = clip_dim
        self.dino_dim = dino_dim

        # Projection matrices and training-set means are stored as buffers
        # so they persist in checkpoints. Allocate final shapes up front
        # whenever the teacher dimensions are known to keep load_state_dict
        # stable across save/load cycles.
        a_shape = (shared_dim, clip_dim) if clip_dim is not None else (shared_dim, 1)
        b_shape = (shared_dim, dino_dim) if dino_dim is not None else (shared_dim, 1)
        mean_c_shape = (clip_dim,) if clip_dim is not None else (1,)
        mean_d_shape = (dino_dim,) if dino_dim is not None else (1,)
        self.register_buffer("_A", torch.zeros(a_shape))             # (s, dim_c)
        self.register_buffer("_B", torch.zeros(b_shape))             # (s, dim_d)
        self.register_buffer("_mean_c", torch.zeros(mean_c_shape))   # (dim_c,)
        self.register_buffer("_mean_d", torch.zeros(mean_d_shape))   # (dim_d,)

        # Class prototypes in the shared CCA space (L2-normalised).
        self.register_buffer(
            "prototypes", torch.zeros(num_classes, shared_dim)
        )

        # Track initialization state as a 1-element buffer so it survives checkpoint save/load.
        self.register_buffer(
            "_initialized_flag", torch.tensor(0, dtype=torch.bool)
        )

    @property
    def _initialized(self) -> bool:
        return bool(self._initialized_flag.item())

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

        # Store projection matrices and training-set means as buffers
        A = torch.tensor(cca.A_s, dtype=torch.float32).to(device)  # (s, dim_c)
        B = torch.tensor(cca.B_s, dtype=torch.float32).to(device)  # (s, dim_d)
        mean_c = torch.tensor(cca.mean_c, dtype=torch.float32).to(device)  # (dim_c,)
        mean_d = torch.tensor(cca.mean_d, dtype=torch.float32).to(device)  # (dim_d,)
        self._A.data = A
        self._B.data = B
        self._mean_c.data = mean_c
        self._mean_d.data = mean_d

        clip_feats = clip_feats.float().to(device)
        dino_feats = dino_feats.float().to(device)
        labels = labels.long().to(device)

        # Centre then project (CCA was fitted on centred features)
        clip_proj = (clip_feats - mean_c) @ A.T   # (N, s)
        dino_proj = (dino_feats - mean_d) @ B.T   # (N, s)
        shared = 0.5 * (clip_proj + dino_proj)  # (N, s)

        # Compute per-class mean → prototype
        prototypes = torch.zeros(self.num_classes, self.shared_dim, device=device)
        for k in range(self.num_classes):
            mask = labels == k
            if mask.sum() > 0:
                prototypes[k] = shared[mask].mean(dim=0)

        # L2-normalise prototypes
        self.prototypes.data = F.normalize(prototypes, dim=1)
        self._initialized_flag.fill_(True)
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

        A = self._A   # buffer, already on correct device
        B = self._B

        # Centre then project (CCA was fitted on centred features)
        clip_proj = (clip_feats - self._mean_c) @ A.T   # (B, s)
        dino_proj = (dino_feats - self._mean_d) @ B.T   # (B, s)

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
