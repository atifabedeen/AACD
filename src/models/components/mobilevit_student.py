"""
MobileViT student backbone with patch token access.

Wraps timm's MobileViT to expose:
  * Patch tokens from the final stage (for semantic-aware aggregation)
  * Intermediate stage features (for feature-wise distillation)
  * GAP features for debugging/ablation
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class MobileViTStudent(nn.Module):
    """
    Lightweight ViT student that produces patch tokens at the final stage
    and global features at each intermediate stage.

    Parameters
    ----------
    arch        : timm model name (e.g. ``mobilevit_s``, ``mobilevit_xs``).
    num_classes : Number of output classes.
    pretrained  : Load ImageNet-pretrained weights.
    """

    def __init__(
        self,
        arch: str = "mobilevit_s",
        num_classes: int = 200,
        pretrained: bool = True,
    ):
        super().__init__()

        # Multi-scale feature extractor (returns one tensor per stage)
        self.backbone = timm.create_model(
            arch, pretrained=pretrained, features_only=True,
        )
        self.feature_info = self.backbone.feature_info

        # Channel dimensions at each stage (e.g. [32, 64, 96, 128, 640])
        self.stage_dims: list[int] = [
            info["num_chs"] for info in self.feature_info
        ]

        # Final-stage channels = student feature dimension
        self.num_features: int = self.stage_dims[-1]

        # Classification head. AACD applies it to the aggregated features.
        self.classifier = nn.Linear(self.num_features, num_classes)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the student classifier to a global feature representation."""
        return self.classifier(features)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Parameters
        ----------
        x : (B, 3, H, W) input images.

        Returns
        -------
        patch_tokens   : (B, N, D) spatial tokens from the final stage.
        global_features: (B, D) GAP features from the final stage.
        gap_logits     : (B, C) classification logits from GAP features.
        intermediates  : list[(B, D_l)] GAP features from earlier stages.
        """
        stage_features = self.backbone(x)  # list of (B, C_l, H_l, W_l)

        # ---- Final stage -> patch tokens + GAP ------------------------
        final = stage_features[-1]                       # (B, C, H, W)
        B, C, H, W = final.shape
        patch_tokens = final.permute(0, 2, 3, 1).reshape(B, H * W, C)
        global_features = final.mean(dim=[2, 3])        # (B, C)

        # Retained for debugging/ablation; AACD uses classify(hidden_features).
        gap_logits = self.classify(global_features)     # (B, num_classes)

        # ---- Intermediate stages -> GAP each --------------------------
        intermediates = [
            feat.mean(dim=[2, 3]) for feat in stage_features[:-1]
        ]

        return patch_tokens, global_features, gap_logits, intermediates
