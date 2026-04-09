"""
AE-SVC: Autoencoders with Strong Variance Constraints
Exact reproduction from Omama et al., ICLR 2025.
"""
import torch
import torch.nn as nn


class AE_SVC(nn.Module):
    """
    Autoencoder with Strong Variance Constraints.

    Architecture: 3-layer encoder, 3-layer decoder (AE-SVC_3 from Table 1).
    Constraints applied at the latent (bottleneck) layer, NOT at decoder output
    (Table 2 in paper shows this is optimal).
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + latent_dim) // 2

        # 3-layer encoder (AE-SVC_3 from paper appendix A.7)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # 3-layer decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec

    @staticmethod
    def compute_losses(z: torch.Tensor, x: torch.Tensor, x_rec: torch.Tensor):
        """
        Compute AE-SVC losses exactly as in paper Equations 1-5.
        Uses hyperparameters from Appendix A.6:
            lambda_rec=25, lambda_cov=1, lambda_var=15, lambda_mean=1

        Args:
            z: latent representations [n, d]
            x: original input [n, d_in]
            x_rec: reconstruction [n, d_in]

        Returns:
            total_loss, dict of individual losses
        """
        n, d = z.shape

        # Eq. 1: Reconstruction loss
        L_rec = torch.mean((x - x_rec) ** 2)

        # Eq. 2: Covariance loss (penalize off-diagonal of covariance matrix)
        mu = z.mean(dim=0, keepdim=True)
        z_centered = z - mu
        cov = (z_centered.T @ z_centered) / n
        # ||cov - I||_F^2
        L_cov = ((cov - torch.eye(d, device=z.device)) ** 2).sum()

        # Eq. 3: Variance loss (push each dimension's variance toward 1)
        var_per_dim = z.var(dim=0)
        L_var = ((var_per_dim - 1.0) ** 2).mean()

        # Eq. 4: Mean centering loss
        mean_per_dim = z.mean(dim=0)
        L_mean = (mean_per_dim ** 2).mean()

        # Eq. 5: Total with paper hyperparameters (Appendix A.6)
        total = 25.0 * L_rec + 1.0 * L_cov + 15.0 * L_var + 1.0 * L_mean

        return total, {
            "ae_svc_rec": L_rec,
            "ae_svc_cov": L_cov,
            "ae_svc_var": L_var,
            "ae_svc_mean": L_mean,
        }
