"""
Kernel operations for KUEA-style pairwise kernel geometry losses.

Provides batchwise kernel matrix construction for cosine and polynomial
kernels, used to replace the CCA + ConceptBasis concept-space distillation.
"""

from __future__ import annotations

import torch


def cosine_kernel(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Cosine similarity kernel: L2-normalize rows, return x @ x.T.

    Args:
        x: Feature matrix [B, D].
        eps: Small constant for numerical stability.

    Returns:
        Kernel matrix [B, B].
    """
    x_norm = x / (x.norm(dim=-1, keepdim=True) + eps)
    return x_norm @ x_norm.T


def normalized_polynomial_kernel(
    x: torch.Tensor,
    gamma: float | torch.Tensor,
    coef0: float | torch.Tensor,
    degree: int = 3,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Normalized polynomial kernel: (gamma * x@x.T + coef0)^degree, diagonal-normalized.

    Args:
        x: Feature matrix [B, D].
        gamma: Scaling factor for the dot product.
        coef0: Additive constant inside the polynomial.
        degree: Polynomial degree.
        eps: Small constant for numerical stability.

    Returns:
        Diagonal-normalized kernel matrix [B, B].
    """
    raw = (gamma * (x @ x.T) + coef0) ** degree
    diag = raw.diag().clamp(min=eps)
    norm = (diag.unsqueeze(1) * diag.unsqueeze(0)).sqrt()
    return raw / norm


def kuea_dino_polynomial_kernel(
    x: torch.Tensor,
    degree: int = 3,
    eps: float = 1e-10,
) -> torch.Tensor:
    """KUEA fixed-style polynomial kernel: (x@x.T / n_feat + 1)^degree, diagonal-normalized.

    Args:
        x: Feature matrix [B, D].
        degree: Polynomial degree.
        eps: Small constant for numerical stability.

    Returns:
        Diagonal-normalized kernel matrix [B, B].
    """
    n_feat = x.shape[1]
    raw = (x @ x.T / n_feat + 1.0) ** degree
    diag = raw.diag().clamp(min=eps)
    norm = (diag.unsqueeze(1) * diag.unsqueeze(0)).sqrt()
    return raw / norm


def build_kernel(
    x: torch.Tensor,
    kernel_name: str,
    gamma: float | torch.Tensor | None = None,
    coef0: float | torch.Tensor | None = None,
    degree: int = 3,
    role: str = "generic",
) -> torch.Tensor:
    """Dispatch to the appropriate kernel function.

    Args:
        x: Feature matrix [B, D].
        kernel_name: One of ``"cosine"`` or ``"poly"``.
        gamma: Required for ``"poly"`` with role != ``"dino"``.
        coef0: Required for ``"poly"`` with role != ``"dino"``.
        degree: Polynomial degree (used by poly kernels).
        role: ``"dino"`` selects the KUEA fixed-style polynomial kernel;
              any other value selects the parameterized normalized polynomial.

    Returns:
        Kernel matrix [B, B].
    """
    if kernel_name == "cosine":
        return cosine_kernel(x)
    elif kernel_name == "poly":
        if role == "dino":
            return kuea_dino_polynomial_kernel(x, degree=degree)
        else:
            if gamma is None or coef0 is None:
                raise ValueError(
                    "gamma and coef0 are required for normalized polynomial kernel "
                    f"(kernel_name='poly', role='{role}')."
                )
            return normalized_polynomial_kernel(x, gamma, coef0, degree=degree)
    else:
        raise ValueError(f"Unknown kernel_name: {kernel_name!r}. Expected 'cosine' or 'poly'.")
