"""
Feature-wise distillation module (proposal SS5.11, NanoSD-inspired).

Projects student intermediate features to the shared CCA dimension
so they can be matched against the shared distillation target at
every depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureWiseDistillation(nn.Module):
    """
    Multi-scale projectors that map student intermediate features
    to the CCA shared dimension.

    Parameters
    ----------
    student_dims : Channel dimensions at each intermediate student stage.
    target_dim   : CCA shared-space dimension (= shared_dim).
    """

    def __init__(self, student_dims: list[int], target_dim: int):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, target_dim),
                nn.ReLU(),
                nn.Linear(target_dim, target_dim),
            )
            for d in student_dims
        ])

    def project(
        self, intermediates: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Project each intermediate feature to the shared CCA space.

        Parameters
        ----------
        intermediates : list of (B, D_l) student features (one per stage).

        Returns
        -------
        list of (B, target_dim) projected features.
        """
        return [
            proj(feat)
            for proj, feat in zip(self.projectors, intermediates)
        ]
