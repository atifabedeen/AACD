"""
Semantic-aware patch token aggregation (proposal SS5.9).

Learns per-patch attention weights and aggregates patch tokens into a
single global representation:

    a_j  = sigmoid(W @ s_j)        (B, N, 1)
    s    = sum_j  a_j * s_j        (B, D)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SemanticAwareAggregation(nn.Module):
    """Learnable attention-weighted pooling over spatial patch tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(
        self, patch_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        patch_tokens : (B, N, D) patch-level features.

        Returns
        -------
        aggregated : (B, D)  attention-weighted global feature.
        weights    : (B, N)  per-patch attention weights.
        """
        weights = torch.sigmoid(self.gate(patch_tokens))   # (B, N, 1)
        aggregated = (weights * patch_tokens).sum(dim=1)   # (B, D)
        return aggregated, weights.squeeze(-1)
