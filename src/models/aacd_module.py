"""
AACD Lightning module — KUEA kernel refactor.

Replaces CCA + ConceptBasis initialization and concept-space loss logging
with kernel-geometry distillation.  AE-SVC teacher preprocessing is kept
(controlled by ``use_ae_svc``).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.ae_svc import AE_SVC
from src.models.components.aacd_criterion import AACDKernelCriterion


class AACDModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        kd_criterion: AACDKernelCriterion,
        use_teacher: bool = True,
        use_ae_svc: bool = True,
        cache_dir: str = ".",
        compile: bool = False,
        ae_svc_epochs: int = 50,
        ae_svc_lr: float = 1e-3,
        ae_svc_batch_size: int = 256,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["net", "optimizer", "scheduler", "kd_criterion"], logger=False
        )

        self.net = net
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.kd_criterion = kd_criterion
        self.use_teacher = use_teacher
        self.use_ae_svc = use_ae_svc
        self.compile_model = compile
        self.criterion = torch.nn.CrossEntropyLoss()
        self.cache_dir = cache_dir
        self.ae_svc_epochs = ae_svc_epochs
        self.ae_svc_lr = ae_svc_lr
        self.ae_svc_batch_size = ae_svc_batch_size

        num_classes = self._aacd_net.data_attributes.class_num
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.cls_loss = MeanMetric()
        self.clip_kernel_loss = MeanMetric()
        self.dino_kernel_loss = MeanMetric()
        self.txt_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    @property
    def _aacd_net(self):
        return getattr(self.net, "_orig_mod", self.net)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        if self.compile_model and stage == "fit":
            self.net = torch.compile(self.net)

        if stage == "fit":
            self._setup_aacd_state()
        elif stage == "test":
            if not self.use_ae_svc:
                return  # nothing to check
            if not self._aacd_net.ae_encoders_ready:
                raise RuntimeError(
                    "AACD AE-SVC state was not restored after loading checkpoint. "
                    "Ensure the checkpoint was saved from a trained AACD model."
                )

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cache_stem(self, dm) -> str:
        net = self._aacd_net
        key_parts = [
            f"data={dm.hparams.attributes.name}",
            f"split_seed={getattr(dm.hparams, 'split_seed', 'na')}",
            f"clip_dim={getattr(net, 'clip_feature_dim', 'na')}",
            f"dino_dim={getattr(net, 'dino_feature_dim', 'na')}",
            f"shared_dim={net.shared_dim}",
            f"kernel={net.kernel_name}",
        ]
        digest = hashlib.sha1("|".join(key_parts).encode("utf-8")).hexdigest()[:12]
        data_name = dm.hparams.attributes.name.replace("/", "_")
        return f"{data_name}_{digest}"

    def _feature_cache_path(self, dm) -> str:
        return os.path.join(
            self.cache_dir, f"aacd_features_v3_{self._cache_stem(dm)}.pth"
        )

    def _init_state_path(self, dm) -> str:
        return os.path.join(
            self.cache_dir, f"aacd_init_v1_{self._cache_stem(dm)}.pth"
        )

    # ------------------------------------------------------------------
    # AACD state management
    # ------------------------------------------------------------------

    def _move_aacd_modules_to_device(self) -> None:
        device = self.device
        net = self._aacd_net
        net.clip_ae_encoder.to(device)
        net.dino_ae_encoder.to(device)

    def _save_aacd_state(self, init_state_path: str) -> None:
        net = self._aacd_net
        payload = {
            "clip_ae_encoder": net.clip_ae_encoder.state_dict(),
            "dino_ae_encoder": net.dino_ae_encoder.state_dict(),
            "clip_ae_ready": bool(net._clip_ae_ready.item()),
            "dino_ae_ready": bool(net._dino_ae_ready.item()),
        }
        torch.save(payload, init_state_path)

    def _load_aacd_state(self, init_state_path: str) -> None:
        net = self._aacd_net
        payload = torch.load(init_state_path, map_location="cpu")
        net.clip_ae_encoder.load_state_dict(payload["clip_ae_encoder"])
        net.dino_ae_encoder.load_state_dict(payload["dino_ae_encoder"])
        net._freeze_module(net.clip_ae_encoder)
        net._freeze_module(net.dino_ae_encoder)
        net._clip_ae_ready.fill_(bool(payload.get("clip_ae_ready", True)))
        net._dino_ae_ready.fill_(bool(payload.get("dino_ae_ready", True)))
        self._move_aacd_modules_to_device()

    def _restore_or_initialize_aacd_state(self, dm) -> None:
        self._move_aacd_modules_to_device()

        if not self.use_ae_svc:
            return

        if self._aacd_net.ae_encoders_ready:
            return

        init_state_path = self._init_state_path(dm)
        if os.path.exists(init_state_path):
            print(
                f"[AACD] Loading cached AACD initialization from {init_state_path}"
            )
            self._load_aacd_state(init_state_path)
            return

        self._initialize_teacher_preproc()
        self._save_aacd_state(init_state_path)
        print(f"[AACD] Cached AACD initialization saved to {init_state_path}")

    def _setup_aacd_state(self) -> None:
        dm = self.trainer.datamodule
        world_size = getattr(self.trainer, "world_size", 1)

        if world_size <= 1:
            self._restore_or_initialize_aacd_state(dm)
            return

        if self.trainer.is_global_zero:
            self._restore_or_initialize_aacd_state(dm)
        self.trainer.strategy.barrier("aacd_init")
        if not self.trainer.is_global_zero:
            self._load_aacd_state(self._init_state_path(dm))
        self.trainer.strategy.barrier("aacd_init_loaded")

    # ------------------------------------------------------------------
    # Teacher preprocessing (AE-SVC only, no CCA/ConceptBasis)
    # ------------------------------------------------------------------

    def _initialize_teacher_preproc(self) -> None:
        dm = self.trainer.datamodule
        device = self.device
        net = self._aacd_net

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._feature_cache_path(dm)

        if os.path.exists(cache_path):
            print(f"[AACD] Loading cached teacher features from {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            clip_feats = cached["clip"]
            dino_feats = cached["dino"]
        else:
            print("[AACD] Extracting teacher features for AE-SVC training ...")
            clip_feats, dino_feats, _labels_all = self._extract_features(dm, device)
            torch.save(
                {"clip": clip_feats, "dino": dino_feats, "labels": _labels_all},
                cache_path,
            )
            print(f"[AACD] Cached features saved to {cache_path}")

        clip_feats = clip_feats.float().to(device)
        dino_feats = dino_feats.float().to(device)

        clip_dim = clip_feats.shape[1]
        dino_dim = dino_feats.shape[1]

        print("[AACD] Training AE-SVC for CLIP features ...")
        _clip_feats_svc, clip_ae_encoder = self._train_ae_svc(
            clip_feats, clip_dim, clip_dim
        )
        print("[AACD] Training AE-SVC for DINO features ...")
        _dino_feats_svc, dino_ae_encoder = self._train_ae_svc(
            dino_feats, dino_dim, dino_dim
        )

        net.set_ae_encoders(clip_ae_encoder, dino_ae_encoder)
        print("[AACD] AE-SVC preprocessing complete.")

    # ------------------------------------------------------------------
    # Feature extraction & AE-SVC training (unchanged)
    # ------------------------------------------------------------------

    def _train_ae_svc(
        self, features: torch.Tensor, input_dim: int, latent_dim: int
    ) -> Tuple[torch.Tensor, torch.nn.Module]:
        """Train AE-SVC offline on cached features and return transformed features + encoder."""
        device = features.device
        model = AE_SVC(input_dim, latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.ae_svc_lr)
        dataset = TensorDataset(features)
        loader = DataLoader(
            dataset, batch_size=self.ae_svc_batch_size, shuffle=True
        )

        model.train()
        for epoch in range(self.ae_svc_epochs):
            for (batch,) in loader:
                z, x_rec = model(batch)
                loss, _ = AE_SVC.compute_losses(z, batch, x_rec)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(
                    f"  [AE-SVC] epoch {epoch+1}/{self.ae_svc_epochs}, loss={loss.item():.4f}"
                )

        model.eval()
        with torch.no_grad():
            z_all, _ = model(features)
        return z_all, model.encoder

    @torch.no_grad()
    def _extract_features(
        self, dm, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torchvision import transforms as T

        clean_transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
            ]
        )

        raw_dataset = dm.data_train
        orig_transform = raw_dataset.transform
        raw_dataset.transform = clean_transform

        loader = DataLoader(
            raw_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        net = self._aacd_net
        clip_teacher = net.clip_teacher.to(device)
        dino_teacher = net.dino_teacher.to(device)

        all_clip, all_dino, all_labels = [], [], []
        n = len(loader)
        for i, (images, labels_batch) in enumerate(loader, 1):
            images = images.to(device)
            clip_inputs = net.preprocess_for_clip(images)
            dino_inputs = net.preprocess_for_imagenet(images)
            clip_f = clip_teacher(clip_inputs)
            dino_f = dino_teacher(dino_inputs)
            all_clip.append(clip_f.cpu())
            all_dino.append(dino_f.cpu())
            all_labels.append(labels_batch)
            if i % 20 == 0 or i == n:
                print(f"  [extract] batch {i}/{n}", flush=True)

        raw_dataset.transform = orig_transform
        return (
            torch.cat(all_clip, dim=0),
            torch.cat(all_dino, dim=0),
            torch.cat(all_labels, dim=0),
        )

    # ------------------------------------------------------------------
    # Forward / model step
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        return self.net(x, labels=labels)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        outputs = self.forward(x, labels=y)
        loss_dict = self.kd_criterion(
            outputs,
            y,
            epoch=self.current_epoch,
            max_epochs=self.trainer.max_epochs,
        )
        preds = torch.argmax(outputs["logits"], dim=1)
        return loss_dict, preds, y, outputs

    @staticmethod
    def _module_grad_norm(module: torch.nn.Module) -> torch.Tensor | None:
        grads = [
            p.grad.detach().norm(2)
            for p in module.parameters()
            if p.grad is not None
        ]
        if not grads:
            return None
        return torch.stack(grads).norm(2)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss_dict, preds, targets, outputs = self.model_step(batch)
        loss = loss_dict["total"]

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.cls_loss(loss_dict["cls"])
        self.clip_kernel_loss(loss_dict["clip_kernel"])
        self.dino_kernel_loss(loss_dict["dino_kernel"])
        self.txt_loss(loss_dict["txt_kernel"])

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/cls", self.cls_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/clip_kernel", self.clip_kernel_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dino_kernel", self.dino_kernel_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/txt_kernel", self.txt_loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def on_after_backward(self) -> None:
        if not getattr(self._aacd_net, "use_mobilevit", False):
            return

        classifier_grad = self._module_grad_norm(self._aacd_net.student.classifier)
        if classifier_grad is not None:
            self.log(
                "train/grad_classifier",
                classifier_grad,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        patch_agg = getattr(self._aacd_net, "patch_agg", None)
        if patch_agg is not None:
            patch_grad = self._module_grad_norm(patch_agg)
            if patch_grad is not None:
                self.log(
                    "train/grad_patchagg",
                    patch_grad,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )

    def on_train_epoch_end(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx: int) -> None:
        loss_dict, preds, targets, outputs = self.model_step(batch)
        self.val_loss(loss_dict["total"])
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/clip_kernel", loss_dict["clip_kernel"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/dino_kernel", loss_dict["dino_kernel"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/txt_kernel", loss_dict["txt_kernel"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx: int) -> None:
        loss_dict, preds, targets, outputs = self.model_step(batch)
        self.test_loss(loss_dict["total"])
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/clip_kernel", loss_dict["clip_kernel"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/dino_kernel", loss_dict["dino_kernel"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/txt_kernel", loss_dict["txt_kernel"], on_step=False, on_epoch=True, prog_bar=False)

    def on_test_epoch_end(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_params = [
            p for _name, p in self.net.named_parameters() if p.requires_grad
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
