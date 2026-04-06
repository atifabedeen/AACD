import copy
from typing import Any, Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


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
        val_ratio: float = 0.10,
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
        train_eval = copy.deepcopy(self.train_source.dataloader)
        test_eval = copy.deepcopy(self.test_source.dataloader)

        if hasattr(train_augmented, "transform"):
            train_augmented.transform = train_transform
        for dataset in (train_eval, test_eval):
            if hasattr(dataset, "transform"):
                dataset.transform = eval_transform

        train_labels = self._extract_labels(train_eval)
        train_indices, val_indices = self._stratified_train_val_split(
            train_labels,
            val_ratio=self.hparams.val_ratio,
            seed=self.hparams.split_seed,
        )
        test_indices = list(range(len(test_eval)))

        self.data_train = IndexedDataset(train_augmented, train_indices, self.attributes)
        self.data_val = IndexedDataset(train_eval, val_indices, self.attributes)
        self.data_test = IndexedDataset(test_eval, test_indices, self.attributes)

    @staticmethod
    def _extract_labels(dataset: Dataset) -> list[int]:
        labels = getattr(dataset, "labels", None)
        if labels is not None:
            return [int(label) for label in labels]

        data_frame = getattr(dataset, "data_frame", None)
        if data_frame is not None:
            for column in ("ClassId", "label", "labels"):
                if column in data_frame.columns:
                    return data_frame[column].astype(int).tolist()
            if data_frame.shape[1] > 6:
                return data_frame.iloc[:, 6].astype(int).tolist()

        return [int(dataset[idx][1]) for idx in range(len(dataset))]

    @staticmethod
    def _stratified_train_val_split(
        labels: list[int],
        val_ratio: float,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

        per_class: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            per_class.setdefault(int(label), []).append(idx)

        generator = torch.Generator().manual_seed(seed)
        train_indices: list[int] = []
        val_indices: list[int] = []

        for label in sorted(per_class):
            label_indices = per_class[label]
            perm = torch.randperm(len(label_indices), generator=generator).tolist()
            shuffled = [label_indices[i] for i in perm]

            if len(shuffled) <= 1 or val_ratio == 0.0:
                n_val = 0
            else:
                n_val = int(round(len(shuffled) * val_ratio))
                n_val = max(n_val, 1)
                n_val = min(n_val, len(shuffled) - 1)

            val_indices.extend(shuffled[:n_val])
            train_indices.extend(shuffled[n_val:])

        train_perm = torch.randperm(len(train_indices), generator=generator).tolist()
        val_perm = torch.randperm(len(val_indices), generator=generator).tolist()
        train_indices = [train_indices[i] for i in train_perm]
        val_indices = [val_indices[i] for i in val_perm]
        return train_indices, val_indices

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
