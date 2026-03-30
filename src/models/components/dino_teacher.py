import timm
import torch
import torch.nn as nn


class DINOv2Teacher(nn.Module):
    """
    Frozen DINOv2 teacher encoder.

    Uses timm's DINOv2 backbones instead of facebookresearch/dinov2 via
    torch.hub so the model can be instantiated under Python 3.9.

    Available model_name options (increasing capacity):
      - dinov2_vits14:  384-dim  (fast iteration / debugging)
      - dinov2_vitb14:  768-dim  (development)
      - dinov2_vitl14: 1024-dim  (strong baseline)
      - dinov2_vitg14: 1536-dim  (final results, matches VL2Lite paper setup)
    """

    MODEL_MAP = {
        "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
        "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
        "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
        "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
    }

    def __init__(self, model_name: str = "dinov2_vits14"):
        super().__init__()

        timm_name = self.MODEL_MAP.get(model_name, model_name)
        self.model = timm.create_model(timm_name, pretrained=True, num_classes=0, img_size=224)
        self.model.requires_grad_(False)
        self.model.eval()
        self.output_dim = getattr(self.model, "num_features", getattr(self.model, "embed_dim"))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalized global features (B, output_dim)."""
        feats = self.model(x)
        return feats / (feats.norm(dim=-1, keepdim=True) + 1e-10)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor):
        """Returns (cls_token, patch_tokens) when available."""
        out = self.model.forward_features(x)

        if isinstance(out, dict):
            return out["x_norm_clstoken"], out["x_norm_patchtokens"]

        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return out[0], out[1]

        if torch.is_tensor(out) and out.ndim == 3:
            return out[:, 0], out[:, 1:]

        feats = self.model(x)
        return feats, None
