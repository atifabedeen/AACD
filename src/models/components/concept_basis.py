"""
Concept Basis Module for per-concept disagreement gating.

The projection matrix A is from EZPC (Ozdemir et al., 2026):
  - Initialized from CLIP text embeddings of concept phrases (Eq. 3 in EZPC)
  - Anchored via matching loss ||A - Phi||_F^2 (Eq. 3)
  - Orthogonality regularized via ||A^T A - I||_F^2

The per-concept gating is new to AACD:
  - Each teacher's CCA-projected features are projected: z^c_i = c_tilde_i @ A
  - Per-concept disagreement: delta_ik = |z^c_{i,k} - z^d_{i,k}|
  - Per-concept gate: w_{i,k} = exp(-alpha_k * delta_ik)
  - alpha_k is estimated per-concept from training data (not a hyperparameter)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptBasis(nn.Module):
    """
    Learnable concept projection on top of CCA-aligned space.

    Args:
        shared_dim: dimension of CCA shared space (input)
        num_concepts: number of concept directions (output)
    """

    def __init__(self, shared_dim: int, num_concepts: int):
        super().__init__()
        self.shared_dim = shared_dim
        self.num_concepts = num_concepts

        # Learnable projection matrix A in R^{shared_dim x num_concepts}
        # Will be initialized from CLIP text embeddings in init_from_projected_text()
        self.A = nn.Parameter(torch.randn(shared_dim, num_concepts) * 0.01)

        # Frozen anchor matrix Phi (initialized same as A, then frozen)
        self.register_buffer("Phi", torch.zeros(shared_dim, num_concepts))

        # Per-concept alpha (estimated from data, not learned)
        # alpha_k = 1 / (2 * sigma_k^2) where sigma_k^2 = Var(delta_k) / 2
        self.register_buffer(
            "alpha", torch.ones(num_concepts) * 2.0  # default, overwritten by calibrate()
        )

        # Per-concept canonical correlation weights for fusion
        self.register_buffer("correlation_weights", torch.ones(num_concepts))

    @torch.no_grad()
    def init_from_projected_text(self, projected_text_embeddings: torch.Tensor):
        """
        Initialize from text embeddings already projected into CCA space.
        projected_text_embeddings: [shared_dim, num_concepts]
        """
        assert projected_text_embeddings.shape == (self.shared_dim, self.num_concepts), (
            f"Expected shape ({self.shared_dim}, {self.num_concepts}), "
            f"got {projected_text_embeddings.shape}"
        )
        self.Phi.copy_(projected_text_embeddings)
        self.A.data.copy_(self.Phi)

    def forward(self, c_tilde: torch.Tensor, d_tilde: torch.Tensor):
        """
        Project CCA-aligned teacher features into concept space and compute gates.

        Args:
            c_tilde: CLIP features in CCA space [B, shared_dim]
            d_tilde: DINO features in CCA space [B, shared_dim]

        Returns:
            z_c: CLIP concept activations [B, num_concepts]
            z_d: DINO concept activations [B, num_concepts]
            per_concept_gate: w_{i,k} = exp(-alpha_k * |z_c_{i,k} - z_d_{i,k}|)  [B, K]
            shared_target: correlation-weighted fusion [B, num_concepts]
        """
        # Normalize A columns (EZPC Appendix A.3, Eq. 11)
        A_normalized = F.normalize(self.A, dim=0)

        # Project into concept space
        z_c = c_tilde @ A_normalized  # [B, K]
        z_d = d_tilde @ A_normalized  # [B, K]

        # Per-concept disagreement
        delta = (z_c - z_d).abs()  # [B, K]

        # Per-concept gate with per-concept alpha
        per_concept_gate = torch.exp(-self.alpha.unsqueeze(0) * delta)  # [B, K]

        # Per-concept correlation-weighted fusion
        # correlation_weights[k] in [0.01, 1] from CCA — higher means CLIP/DINO
        # agree more on concept k, so trust CLIP more for that concept.
        w = self.correlation_weights.unsqueeze(0)  # [1, K]
        shared_target = w * z_c + (1.0 - w) * z_d  # per-concept blend

        return z_c, z_d, per_concept_gate, shared_target

    @torch.no_grad()
    def calibrate(
        self,
        c_tilde_train: torch.Tensor,
        d_tilde_train: torch.Tensor,
        cca_correlations: torch.Tensor = None,
    ):
        """
        Estimate per-concept alpha from training data disagreement statistics.

        Probabilistic derivation:
          Assume z^c_{i,k} = mu_k + eps^c, z^d_{i,k} = mu_k + eps^d
          where eps ~ N(0, sigma_k^2).
          Then delta_{i,k} = |eps^c - eps^d| follows a folded normal.
          The optimal weight is related to 1/(2*sigma_k^2).
          We estimate sigma_k^2 = Var(delta_k) / 2.

        Args:
            c_tilde_train: all CLIP training features in CCA space [N, shared_dim]
            d_tilde_train: all DINO training features in CCA space [N, shared_dim]
            cca_correlations: canonical correlation values [shared_dim] or [num_concepts]
        """
        A_normalized = F.normalize(self.A, dim=0)
        z_c = c_tilde_train @ A_normalized
        z_d = d_tilde_train @ A_normalized
        delta = (z_c - z_d).abs()  # [N, K]

        # Per-concept variance of disagreement
        sigma_sq = delta.var(dim=0) / 2.0  # [K]
        sigma_sq = sigma_sq.clamp(min=1e-6)

        # alpha_k = 1 / (2 * sigma_k^2)
        self.alpha.copy_(1.0 / (2.0 * sigma_sq))

        # Set correlation weights from CCA if provided
        if cca_correlations is not None:
            K = self.num_concepts
            if cca_correlations.numel() >= K:
                self.correlation_weights.copy_(cca_correlations[:K].clamp(min=0.01))
            else:
                self.correlation_weights.fill_(1.0)

    def anchoring_loss(self):
        """
        EZPC matching loss (Eq. 3): L_match = (1/dm) * ||A - Phi||_F^2
        """
        return ((self.A - self.Phi) ** 2).mean()

    def orthogonality_loss(self):
        """
        Orthogonality regularizer: ||A^T A - I||_F^2
        """
        A_norm = F.normalize(self.A, dim=0)
        K = self.num_concepts
        gram = A_norm.T @ A_norm
        return ((gram - torch.eye(K, device=gram.device)) ** 2).sum()

    def project_to_concepts(self, features: torch.Tensor) -> torch.Tensor:
        """Project features from shared CCA space to concept space.

        Args:
            features: [B, shared_dim] features in CCA space

        Returns:
            [B, num_concepts] concept activations
        """
        A_normalized = F.normalize(self.A, dim=0)
        return features @ A_normalized
