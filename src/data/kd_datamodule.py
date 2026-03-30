import copy
from typing import Any, Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class IndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices, attributes) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.attributes = attributes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

    @property
    def transform(self):
        datasets = getattr(self.base_dataset, "datasets", None)
        if datasets:
            return getattr(datasets[0], "transform", None)
        return getattr(self.base_dataset, "transform", None)

    @transform.setter
    def transform(self, value) -> None:
        datasets = getattr(self.base_dataset, "datasets", None)
        if datasets:
            for dataset in datasets:
                if hasattr(dataset, "transform"):
                    dataset.transform = value
        elif hasattr(self.base_dataset, "transform"):
            self.base_dataset.transform = value


class KDDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        attributes,
        data_name,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        split_seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataset", "test_dataset"], logger=False)

        self.train_source = train_dataset
        self.test_source = test_dataset
        self.attributes = attributes

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return self.attributes.class_num

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train is None and self.data_val is None and self.data_test is None:
            self._build_splits()

    def _build_splits(self) -> None:
        train_transform = self.train_source.transform
        eval_transform = self.test_source.transform

        train_augmented = copy.deepcopy(self.train_source.dataloader)
        test_augmented = copy.deepcopy(self.test_source.dataloader)
        train_eval = copy.deepcopy(self.train_source.dataloader)
        test_eval = copy.deepcopy(self.test_source.dataloader)

        for dataset in (train_augmented, test_augmented):
            if hasattr(dataset, "transform"):
                dataset.transform = train_transform

        for dataset in (train_eval, test_eval):
            if hasattr(dataset, "transform"):
                dataset.transform = eval_transform

        augmented_full = ConcatDataset([train_augmented, test_augmented])
        eval_full = ConcatDataset([train_eval, test_eval])

        total_size = len(eval_full)
        train_size = int(total_size * 0.70)
        val_size = int(total_size * 0.15)
        test_size = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(self.hparams.split_seed)
        indices = torch.randperm(total_size, generator=generator).tolist()

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:train_size + val_size + test_size]

        self.data_train = IndexedDataset(augmented_full, train_indices, self.attributes)
        self.data_val = IndexedDataset(eval_full, val_indices, self.attributes)
        self.data_test = IndexedDataset(eval_full, test_indices, self.attributes)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    _ = KDDataModule()
