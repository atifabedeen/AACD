from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from src.data.kd_datamodule import KDDataModule


class DummyBaseDataset(Dataset):
    def __init__(self, labels, transform=None):
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor([float(idx)], dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


class DummySource:
    def __init__(self, labels, transform):
        self.transform = transform
        self.dataloader = DummyBaseDataset(labels, transform=transform)


def test_kd_datamodule_uses_official_train_for_val_and_keeps_test_untouched() -> None:
    train_labels = [0] * 10 + [1] * 10 + [2] * 10
    test_labels = [0] * 4 + [1] * 4 + [2] * 4
    train_transform = lambda x: x + 100.0
    eval_transform = lambda x: x + 200.0
    attributes = SimpleNamespace(class_num=3)

    dm = KDDataModule(
        train_dataset=DummySource(train_labels, train_transform),
        test_dataset=DummySource(test_labels, eval_transform),
        attributes=attributes,
        data_name='dummy',
        batch_size=4,
        val_ratio=0.10,
        split_seed=7,
    )
    dm.setup()

    assert len(dm.data_train) + len(dm.data_val) == len(train_labels)
    assert len(dm.data_test) == len(test_labels)
    assert set(dm.data_train.indices).isdisjoint(set(dm.data_val.indices))
    assert dm.data_test.indices == list(range(len(test_labels)))

    val_labels = [train_labels[idx] for idx in dm.data_val.indices]
    assert set(val_labels) == {0, 1, 2}

    train_item, _ = dm.data_train[0]
    val_item, _ = dm.data_val[0]
    test_item, _ = dm.data_test[0]
    assert train_item.item() >= 100.0
    assert val_item.item() >= 200.0
    assert test_item.item() >= 200.0

    dm_same = KDDataModule(
        train_dataset=DummySource(train_labels, train_transform),
        test_dataset=DummySource(test_labels, eval_transform),
        attributes=attributes,
        data_name='dummy',
        batch_size=4,
        val_ratio=0.10,
        split_seed=7,
    )
    dm_same.setup()
    assert dm.data_train.indices == dm_same.data_train.indices
    assert dm.data_val.indices == dm_same.data_val.indices
