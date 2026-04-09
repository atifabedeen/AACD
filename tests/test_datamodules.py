from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.kd_datamodule import KDDataModule
from src.data.components.kd_dataloader import Caltech101Dataset


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

def test_caltech101_dataset_honors_saved_train_test_split(tmp_path: Path) -> None:
    root = tmp_path / "7_CALTECH101"
    images_root = root / "101_ObjectCategories"
    for class_name in ("accordion", "airplanes"):
        class_dir = images_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(3):
            Image.new("RGB", (8, 8), color=(idx * 30, 0, 0)).save(class_dir / f"img_{idx}.jpg")

    split_file = root / "train_test_split.txt"
    split_file.write_text(
        "\n".join([
            f"train,{images_root / 'accordion' / 'img_0.jpg'},accordion",
            f"train,{images_root / 'accordion' / 'img_1.jpg'},accordion",
            f"test,{images_root / 'accordion' / 'img_2.jpg'},accordion",
            f"train,{images_root / 'airplanes' / 'img_0.jpg'},airplanes",
            f"train,{images_root / 'airplanes' / 'img_1.jpg'},airplanes",
            f"test,{images_root / 'airplanes' / 'img_2.jpg'},airplanes",
        ])
        + "\n"
    )

    train_ds = Caltech101Dataset(str(root), split="train")
    test_ds = Caltech101Dataset(str(root), split="test")

    assert len(train_ds) == 4
    assert len(test_ds) == 2
    assert set(train_ds.image_paths).isdisjoint(set(test_ds.image_paths))
    assert {Path(path).name for path in test_ds.image_paths} == {"img_2.jpg"}

