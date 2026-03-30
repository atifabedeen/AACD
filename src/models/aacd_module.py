"""
AACD Lightning module.

Extends VL2Lite's KDModule with:
  1. DINOv2 as a second frozen teacher.
  2. CCA fitting + agreement-module initialization in setup().
  3. Agreement-weighted, geometry-preserving AACD loss.

CCA setup flow
--------------
  setup("fit") is called once per process before the first training step.
  It:
    a) Creates a clean (no-augmentation) DataLoader over the training set.
    b) Extracts CLIP + DINOv2 features for every training sample.
    c) Fits CCAProjection.
    d) Calls AgreementModule.initialize() to build class prototypes.
    e) Caches features to <cache_dir>/aacd_features.pth so re-runs are fast.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from lightning import LightningModule
from torch.utils.data import DataLoader
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.cca_module import CCAProjection
from src.models.components.aacd_criterion import AACDCriterion


class AACDModule(LightningModule):
    """
    Lightning module for Agreement-Aware Correlation-Guided Distillation.

    Parameters
    ----------
    net          : AACDTeacherStudent instance (from Hydra config).
    optimizer    : Partial optimizer constructor.
    scheduler    : Partial LR-scheduler constructor (or None).
    kd_criterion : AACDCriterion instance.
    cca_s        : Number of CCA shared dimensions (must match net.agreement.shared_dim).
    cca_tau      : Correlation threshold for auto-detecting s (used if cca_s=0).
    cache_dir    : Directory to cache extracted features.  Defaults to CWD.
    compile      : Whether to torch.compile the model.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        kd_criterion: AACDCriterion,
        use_teacher: bool = True,
        cca_s: int = 256,
        cca_tau: float = 0.1,
        cache_dir: str = ".",
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "optimizer", "scheduler", "kd_criterion"], logger=False)

        self.net = net
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.kd_criterion = kd_criterion
        self.use_teacher = use_teacher
        self.compile_model = compile
        self.criterion = torch.nn.CrossEntropyLoss()
        self.cca_s = cca_s
        self.cca_tau = cca_tau
        self.cache_dir = cache_dir

        num_classes = self.net.data_attributes.class_num

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()
        self.test_loss  = MeanMetric()

        # Extra AACD-specific metrics
        self.kd_loss     = MeanMetric()
        self.cls_loss    = MeanMetric()
        self.shared_loss = MeanMetric()
        self.geom_loss   = MeanMetric()
        self.mean_agree  = MeanMetric()

        self.val_acc_best = MaxMetric()

    # ------------------------------------------------------------------
    # Setup: CCA fitting
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        if self.compile_model and stage == "fit":
            self.net = torch.compile(self.net)

        if stage == "fit":
            self._initialize_agreement()

    def _initialize_agreement(self) -> None:
        """Extract features, fit CCA, initialize AgreementModule."""
        dm = self.trainer.datamodule
        data_name = dm.hparams.attributes.name   # e.g. "0_CUB_200_2011"
        device = self.device

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(
            self.cache_dir,
            f"aacd_features_{data_name}.pth",
        )

        # ---- Feature extraction (or load from cache) -----------------
        if os.path.exists(cache_path):
            print(f"[AACD] Loading cached teacher features from {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            clip_feats = cached["clip"]
            dino_feats = cached["dino"]
            labels_all = cached["labels"]
        else:
            print("[AACD] Extracting teacher features for CCA fitting …")
            clip_feats, dino_feats, labels_all = self._extract_features(
                dm, device
            )
            torch.save(
                {"clip": clip_feats, "dino": dino_feats, "labels": labels_all},
                cache_path,
            )
            print(f"[AACD] Cached features saved to {cache_path}")

        # ---- CCA fitting ---------------------------------------------
        clip_dim = clip_feats.shape[1]
        dino_dim = dino_feats.shape[1]
        s = self.cca_s if self.cca_s > 0 else None

        cca = CCAProjection(dim_c=clip_dim, dim_d=dino_dim, s=s, tau=self.cca_tau)
        cca.fit(clip_feats.numpy(), dino_feats.numpy())

        # Safety check: shared_dim in agreement module must match cca.s
        assert self.net.agreement.shared_dim == cca.s, (
            f"AgreementModule.shared_dim={self.net.agreement.shared_dim} "
            f"!= cca.s={cca.s}. Set cca_s={cca.s} in the config."
        )

        # ---- Agreement module initialization -------------------------
        self.net.agreement.initialize(cca, clip_feats, dino_feats, labels_all)
        print("[AACD] Agreement module initialized.")

    @torch.no_grad()
    def _extract_features(
        self, dm, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a no-augmentation loader and extract CLIP + DINOv2 features.

        ``dm.data_train`` is the raw underlying dataset (e.g. CUB200Dataset).
        We temporarily swap its transform to a clean test-style transform so
        there is no augmentation during feature extraction.
        """
        from torchvision import transforms as T

        normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        clean_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            normalize,
        ])

        raw_dataset = dm.data_train  # underlying CUBDataset / etc.
        orig_transform = raw_dataset.transform
        raw_dataset.transform = clean_transform

        loader = DataLoader(
            raw_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        clip_teacher = self.net.clip_teacher.to(device)
        dino_teacher = self.net.dino_teacher.to(device)

        all_clip, all_dino, all_labels = [], [], []
        n = len(loader)
        for i, (images, labels_batch) in enumerate(loader, 1):
            images = images.to(device)
            clip_f = clip_teacher(images)
            dino_f = dino_teacher(images)
            all_clip.append(clip_f.cpu())
            all_dino.append(dino_f.cpu())
            all_labels.append(labels_batch)
            if i % 20 == 0 or i == n:
                print(f"  [extract] batch {i}/{n}", flush=True)

        # Restore original transform
        raw_dataset.transform = orig_transform

        return (
            torch.cat(all_clip,   dim=0),
            torch.cat(all_dino,   dim=0),
            torch.cat(all_labels, dim=0),
        )

    # ------------------------------------------------------------------
    # Core training logic
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        outputs = self.forward(x)
        loss_dict = self.kd_criterion(
            outputs,
            y,
            epoch=self.current_epoch,
            max_epochs=self.trainer.max_epochs,
        )
        preds = torch.argmax(outputs["logits"], dim=1)
        return loss_dict, preds, y

    # ------------------------------------------------------------------
    # Training / validation / test steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss_dict, preds, targets = self.model_step(batch)
        loss = loss_dict["total"]

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.cls_loss(loss_dict["cls"])
        self.shared_loss(loss_dict["shared"])
        self.kd_loss(loss_dict["vis"] + loss_dict["txt"])
        self.geom_loss(loss_dict["geom"])
        self.mean_agree(loss_dict["mean_agreement"])

        self.log("train/loss",   self.train_loss,  on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc",    self.train_acc,   on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/cls",    self.cls_loss,    on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/shared", self.shared_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/kd",     self.kd_loss,     on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/geom",   self.geom_loss,   on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/agree",  self.mean_agree,  on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch, batch_idx: int) -> None:
        loss_dict, preds, targets = self.model_step(batch)
        self.val_loss(loss_dict["total"])
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc",  self.val_acc,  on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        loss_dict, preds, targets = self.model_step(batch)
        self.test_loss(loss_dict["total"])
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc",  self.test_acc,  on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        # Only train student + alignment layers; teachers are frozen
        trainable_params = [
            p for name, p in self.net.named_parameters()
            if p.requires_grad
        ]
        optimizer = self.optimizer_factory(params=trainable_params)
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
