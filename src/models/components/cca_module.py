"""
Canonical Correlation Analysis (CCA) projection module.

Solves CCA between CLIP and DINOv2 feature spaces to find the maximally
correlated shared subspace.  Follows the derivation in:
    CSA: Correlation-Aware Supervision Alignment (Li et al., ICLR 2025)

Usage
-----
1. Extract all training-set features from both teachers (see scripts/extract_teacher_features.py).
2. Call CCAProjection(dim_c, dim_d).fit(clip_feats, dino_feats).
3. Pass the fitted object to AgreementModule for use during training.

The projected features satisfy:
    corr( A @ z_clip ,  B @ z_dino )  is maximized dimension-wise,
where A ∈ ℝ^{s×dim_c}, B ∈ ℝ^{s×dim_d}.
"""

from __future__ import annotations

import numpy as np


class CCAProjection:
    """Offline CCA solver that stores projection matrices A and B."""

    @staticmethod
    def _inv_sqrt_psd(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Stable inverse square root for symmetric PSD matrices."""
        matrix = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.clip(eigvals, a_min=eps, a_max=None)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return 0.5 * (inv_sqrt + inv_sqrt.T)

    def __init__(
        self,
        dim_c: int,
        dim_d: int,
        s: int | None = None,
        tau: float = 0.1,
        reg: float = 1e-4,
    ):
        """
        Parameters
        ----------
        dim_c : CLIP feature dimension (e.g. 1024 for convnext_xxlarge).
        dim_d : DINOv2 feature dimension (e.g. 384/768/1024/1536).
        s     : Number of correlated dimensions to keep.
                Auto-detected from ``tau`` if None.
        tau   : Correlation threshold for auto-detecting s.
                Dimensions whose canonical correlation > tau are kept;
                minimum is 10.
        reg   : Tikhonov regularisation added to diagonal of covariance
                matrices for numerical stability.
        """
        self.dim_c = dim_c
        self.dim_d = dim_d
        self.s = s
        self.tau = tau
        self.reg = reg

        # Filled by fit()
        self.A: np.ndarray | None = None      # (s, dim_c)
        self.B: np.ndarray | None = None      # (s, dim_d)
        self.mean_c: np.ndarray | None = None # (dim_c,) training-set CLIP mean
        self.mean_d: np.ndarray | None = None # (dim_d,) training-set DINOv2 mean
        self.rho: np.ndarray | None = None    # (r,) descending correlation coefficients
        self.fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, Z_clip: np.ndarray, Z_dino: np.ndarray) -> "CCAProjection":
        """
        Fit CCA on training-set features.

        Parameters
        ----------
        Z_clip : (N, dim_c) float32/float64 array.
        Z_dino : (N, dim_d) float32/float64 array.
        """
        Z_clip = Z_clip.astype(np.float64)
        Z_dino = Z_dino.astype(np.float64)
        N = Z_clip.shape[0]

        # ---- 1. Centre -----------------------------------------------
        self.mean_c = Z_clip.mean(axis=0)   # (dim_c,)
        self.mean_d = Z_dino.mean(axis=0)   # (dim_d,)
        Z_c = Z_clip - self.mean_c[None, :]
        Z_d = Z_dino - self.mean_d[None, :]

        # ---- 2. Covariance matrices (with regularisation) ------------
        Sigma_cc = Z_c.T @ Z_c / N + self.reg * np.eye(self.dim_c)
        Sigma_dd = Z_d.T @ Z_d / N + self.reg * np.eye(self.dim_d)
        Sigma_cd = Z_c.T @ Z_d / N   # (dim_c, dim_d)

        # ---- 3. Whitening --------------------------------------------
        Scc_inv_sqrt = self._inv_sqrt_psd(Sigma_cc)
        Sdd_inv_sqrt = self._inv_sqrt_psd(Sigma_dd)

        # ---- 4. SVD of whitened cross-covariance ---------------------
        #   M = Scc^{-1/2} Σ_{cd} Sdd^{-1/2}  →  U S Vt
        #   Canonical directions for clip : columns of U
        #   Canonical directions for dino : rows of Vt (= columns of V)
        M = Scc_inv_sqrt @ Sigma_cd @ Sdd_inv_sqrt
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        # U : (dim_c, r),  S : (r,),  Vt : (r, dim_d)
        r = len(S)
        self.rho = S  # canonical correlations, descending

        # ---- 5. Projection matrices ----------------------------------
        #   A[k,:] = U[:,k]^T @ Scc^{-1/2}   →  project clip features
        #   B[k,:] = Vt[k,:]  @ Sdd^{-1/2}   →  project dino features
        self.A = (U.T @ Scc_inv_sqrt)   # (r, dim_c)
        self.B = (Vt  @ Sdd_inv_sqrt)   # (r, dim_d)

        # ---- 6. Choose s ---------------------------------------------
        if self.s is None:
            n_valid = int((self.rho > self.tau).sum())
            self.s = max(n_valid, 10)
        self.s = min(self.s, r)

        self.fitted = True
        print(
            f"[CCA] Fitted  r={r}  →  keeping s={self.s} dims.  "
            f"Top-5 ρ: {self.rho[:5].round(4).tolist()}  "
            f"ρ[s-1]={self.rho[self.s - 1]:.4f}"
        )
        return self

    # ------------------------------------------------------------------
    # Convenience slices (first s rows only)

    @property
    def A_s(self) -> np.ndarray:
        """Projection matrix for CLIP, shape (s, dim_c)."""
        return self.A[: self.s]

    @property
    def B_s(self) -> np.ndarray:
        """Projection matrix for DINOv2, shape (s, dim_d)."""
        return self.B[: self.s]

    @property
    def rho_s(self) -> np.ndarray:
        """Top-s canonical correlations."""
        return self.rho[: self.s]

    def project_clip(self, Z_clip: np.ndarray) -> np.ndarray:
        """Project CLIP features using the training-set centering from fit()."""
        assert self.fitted, "CCAProjection must be fitted before projection."
        Z_clip = Z_clip.astype(np.float64)
        centered = Z_clip - self.mean_c[None, :]
        return centered @ self.A_s.T

    def project_dino(self, Z_dino: np.ndarray) -> np.ndarray:
        """Project DINOv2 features using the training-set centering from fit()."""
        assert self.fitted, "CCAProjection must be fitted before projection."
        Z_dino = Z_dino.astype(np.float64)
        centered = Z_dino - self.mean_d[None, :]
        return centered @ self.B_s.T
