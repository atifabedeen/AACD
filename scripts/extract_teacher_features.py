"""
Extract and cache CLIP + DINOv2 features for the training set.

Run this ONCE before starting AACD training to speed up the CCA fitting
step that happens in AACDModule.setup().  The output .pth file is also
used by AACDModule when resuming training.

For quick debugging use --dino_model dinov2_vits14 (384-dim, fastest).
For final results use   --dino_model dinov2_vitg14 (1536-dim).
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

import open_clip

# Allow running from the repo root
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.components.dino_teacher import DINOv2Teacher
from src.data.components.kd_dataloader import KDDataset


# ---------------------------------------------------------------------------

def build_attribute_stub(data_name: str) -> SimpleNamespace:
    """Minimal attributes namespace needed by KDDataset."""
    # We only need class_num and prompt_tmpl for extraction;
    # pull from the YAML configs via a lightweight reader.
    import yaml
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "data", "attributes",
        f"{data_name}.yaml",
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


def extract_features(
    data_name: str,
    data_root: str,
    clip_arch: str,
    clip_ckpt: str,
    dino_model: str,
    out_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    device_str: str = "cuda",
) -> str:
    """Extract and save features; returns path to saved .pth file."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[extract] Device: {device}")

    # ---- Build dataset (test transform = no augmentation) ------------
    attrs = build_attribute_stub(data_name)
    dataset = KDDataset(
        data_name=data_name,
        data_root=data_root,
        attributes=attrs,
        split="train",
    )
    # Override with no-augmentation transform
    dataset.dataloader.transform = dataset.get_transform("test")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---- Load CLIP ---------------------------------------------------
    print(f"[extract] Loading CLIP {clip_arch} ({clip_ckpt}) …")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        clip_arch, pretrained=clip_ckpt
    )
    clip_model = clip_model.to(device).eval()
    clip_model.requires_grad_(False)

    # ---- Load DINOv2 ------------------------------------------------
    print(f"[extract] Loading DINOv2 {dino_model} …")
    dino = DINOv2Teacher(dino_model).to(device)

    # ---- Extract -----------------------------------------------------
    all_clip, all_dino, all_labels = [], [], []
    n_batches = len(loader)

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader, 1):
            images = images.to(device)

            clip_f = clip_model.encode_image(images)
            clip_f = clip_f / (clip_f.norm(dim=-1, keepdim=True) + 1e-10)

            dino_f = dino(images)   # already normalized in DINOv2Teacher.forward

            all_clip.append(clip_f.cpu())
            all_dino.append(dino_f.cpu())
            all_labels.append(labels)

            if i % 10 == 0 or i == n_batches:
                print(f"  batch {i}/{n_batches}", flush=True)

    clip_feats = torch.cat(all_clip,   dim=0)
    dino_feats = torch.cat(all_dino,   dim=0)
    labels_all = torch.cat(all_labels, dim=0)

    print(
        f"[extract] CLIP:  {clip_feats.shape}  "
        f"DINOv2: {dino_feats.shape}  "
        f"Labels: {labels_all.shape}"
    )

    # ---- Save -------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"aacd_features_{data_name}.pth")
    torch.save(
        {"clip": clip_feats, "dino": dino_feats, "labels": labels_all},
        out_path,
    )
    print(f"[extract] Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract AACD teacher features")
    parser.add_argument("--data_name",   required=True,
                        help="Dataset name (e.g. 0_CUB_200_2011)")
    parser.add_argument("--data_root",   required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--clip_arch",   default="convnext_xxlarge",
                        help="CLIP architecture (open_clip model name)")
    parser.add_argument("--clip_ckpt",   default="laion2b_s34b_b82k_augreg_soup",
                        help="CLIP pretrained weights tag")
    parser.add_argument("--dino_model",  default="dinov2_vits14",
                        help="DINOv2 hub model name")
    parser.add_argument("--out_dir",     default=".cache",
                        help="Directory to save the .pth feature cache")
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    extract_features(
        data_name=args.data_name,
        data_root=args.data_root,
        clip_arch=args.clip_arch,
        clip_ckpt=args.clip_ckpt,
        dino_model=args.dino_model,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
