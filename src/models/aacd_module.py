"""
AACD Lightning module.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torch.utils.data import DataLoader
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.aacd_criterion import AACDCriterion
from src.models.components.cca_module import CCAProjection


class AACDModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        kd_criterion: AACDCriterion,
        use_teacher: bool = True,
        cca_s: int = 128,
        cca_tau: float = 0.1,
        cache_dir: str = '.',
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['net', 'optimizer', 'scheduler', 'kd_criterion'], logger=False)

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

        num_classes = self._aacd_net.data_attributes.class_num
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.cls_loss = MeanMetric()
        self.shared_loss = MeanMetric()
        self.kd_loss = MeanMetric()
        self.geom_loss = MeanMetric()
        self.feat_loss = MeanMetric()
        self.txt_loss = MeanMetric()
        self.patch_entropy = MeanMetric()
        self.val_acc_best = MaxMetric()

    @property
    def _aacd_net(self):
        return getattr(self.net, '_orig_mod', self.net)

    def setup(self, stage: str) -> None:
        if self.compile_model and stage == 'fit':
            self.net = torch.compile(self.net)

        if stage == 'fit':
            self._initialize_agreement()
        elif stage == 'test' and not self._aacd_net.agreement._initialized:
            raise RuntimeError(
                'AgreementModule was not initialized after loading checkpoint. '
                'Ensure the checkpoint was saved from a trained AACD model.'
            )

    def _initialize_agreement(self) -> None:
        dm = self.trainer.datamodule
        data_name = dm.hparams.attributes.name
        device = self.device
        net = self._aacd_net

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f'aacd_features_v2_{data_name}.pth')

        if os.path.exists(cache_path):
            print(f'[AACD] Loading cached teacher features from {cache_path}')
            cached = torch.load(cache_path, map_location='cpu')
            clip_feats = cached['clip']
            dino_feats = cached['dino']
            labels_all = cached['labels']
        else:
            print('[AACD] Extracting teacher features for CCA fitting ...')
            clip_feats, dino_feats, labels_all = self._extract_features(dm, device)
            torch.save({'clip': clip_feats, 'dino': dino_feats, 'labels': labels_all}, cache_path)
            print(f'[AACD] Cached features saved to {cache_path}')

        clip_dim = clip_feats.shape[1]
        dino_dim = dino_feats.shape[1]
        s = self.cca_s if self.cca_s > 0 else None
        cca = CCAProjection(dim_c=clip_dim, dim_d=dino_dim, s=s, tau=self.cca_tau)
        cca.fit(clip_feats.numpy(), dino_feats.numpy())

        assert net.agreement.shared_dim == cca.s, (
            f'AgreementModule.shared_dim={net.agreement.shared_dim} '
            f'!= cca.s={cca.s}. Set cca_s={cca.s} in the config.'
        )
        net.agreement.initialize(cca, clip_feats, dino_feats, labels_all)
        print('[AACD] Agreement module initialized.')

    @torch.no_grad()
    def _extract_features(
        self, dm, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torchvision import transforms as T

        clean_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

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
                print(f'  [extract] batch {i}/{n}', flush=True)

        raw_dataset.transform = orig_transform
        return (
            torch.cat(all_clip, dim=0),
            torch.cat(all_dino, dim=0),
            torch.cat(all_labels, dim=0),
        )

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
        preds = torch.argmax(outputs['logits'], dim=1)
        return loss_dict, preds, y, outputs

    @staticmethod
    def _module_grad_norm(module: torch.nn.Module) -> torch.Tensor | None:
        grads = [p.grad.detach().norm(2) for p in module.parameters() if p.grad is not None]
        if not grads:
            return None
        return torch.stack(grads).norm(2)

    def _log_aux(self, prefix: str, loss_dict: dict, outputs: dict) -> None:
        self.log(f'{prefix}/agree', loss_dict['agreement_rate'], on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{prefix}/delta', loss_dict['mean_delta'], on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{prefix}/clip_margin', loss_dict['mean_clip_margin'], on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{prefix}/dino_margin', loss_dict['mean_dino_margin'], on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{prefix}/shared_full', loss_dict['full_shared_frac'], on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{prefix}/shared_soft', loss_dict['soft_shared_frac'], on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{prefix}/label_only', loss_dict['label_only_frac'], on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{prefix}/text_weight', loss_dict['mean_text_kd_weight'], on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{prefix}/patch_entropy', outputs['patch_entropy'], on_step=False, on_epoch=True, prog_bar=False)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss_dict, preds, targets, outputs = self.model_step(batch)
        loss = loss_dict['total']

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.cls_loss(loss_dict['cls'])
        self.shared_loss(loss_dict['shared'])
        self.kd_loss(loss_dict['shared'] + loss_dict['txt'])
        self.geom_loss(loss_dict['geom'])
        self.feat_loss(loss_dict['feat'])
        self.txt_loss(loss_dict['txt'])
        self.patch_entropy(outputs['patch_entropy'])

        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/cls', self.cls_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/shared', self.shared_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/kd', self.kd_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/txt', self.txt_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/txt_raw', loss_dict['txt_raw'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/geom', self.geom_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/feat', self.feat_loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_aux('train', loss_dict, outputs)
        return loss

    def on_after_backward(self) -> None:
        if not getattr(self._aacd_net, 'use_mobilevit', False):
            return

        classifier_grad = self._module_grad_norm(self._aacd_net.student.classifier)
        if classifier_grad is not None:
            self.log('train/grad_classifier', classifier_grad, on_step=True, on_epoch=False, prog_bar=False)

        patch_agg = getattr(self._aacd_net, 'patch_agg', None)
        if patch_agg is not None:
            patch_grad = self._module_grad_norm(patch_agg)
            if patch_grad is not None:
                self.log('train/grad_patchagg', patch_grad, on_step=True, on_epoch=False, prog_bar=False)

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch, batch_idx: int) -> None:
        loss_dict, preds, targets, outputs = self.model_step(batch)
        self.val_loss(loss_dict['total'])
        self.val_acc(preds, targets)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/shared', loss_dict['shared'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/txt', loss_dict['txt'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/txt_raw', loss_dict['txt_raw'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/geom', loss_dict['geom'], on_step=False, on_epoch=True, prog_bar=False)
        self._log_aux('val', loss_dict, outputs)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log('val/acc_best', self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        loss_dict, preds, targets, outputs = self.model_step(batch)
        self.test_loss(loss_dict['total'])
        self.test_acc(preds, targets)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/shared', loss_dict['shared'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/txt', loss_dict['txt'], on_step=False, on_epoch=True, prog_bar=False)
        self._log_aux('test', loss_dict, outputs)

    def on_test_epoch_end(self) -> None:
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_params = [
            p for _name, p in self.net.named_parameters()
            if p.requires_grad
        ]
        optimizer = self.optimizer_factory(params=trainable_params)
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }
        return {'optimizer': optimizer}
