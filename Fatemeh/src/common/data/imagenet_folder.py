# src/common/data/imagenet_folder.py
# Data loading for ImageFolder-style datasets.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    dataset_root: Path
    train_subdir: str
    val_subdir: str
    image_size: int
    batch_size: int
    num_workers: int


def build_transforms(image_size: int) -> Dict[str, transforms.Compose]:
    """
    Create train/val transforms suitable for ImageNet-pretrained backbones.
    """
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return {"train": train_tfms, "val": val_tfms}


def create_dataloaders(cfg: DataConfig) -> Tuple[Dict[str, DataLoader], Dict[str, datasets.ImageFolder]]:
    """
    Build PyTorch DataLoaders for train and validation splits.

    Returns:
        dataloaders: {"train": DataLoader, "val": DataLoader}
        datasets_map: {"train": ImageFolder, "val": ImageFolder}
    """
    train_dir = cfg.dataset_root / cfg.train_subdir
    val_dir = cfg.dataset_root / cfg.val_subdir

    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Val folder not found: {val_dir}")

    tfms = build_transforms(cfg.image_size)

    train_ds = datasets.ImageFolder(train_dir, transform=tfms["train"])
    val_ds = datasets.ImageFolder(val_dir, transform=tfms["val"])

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "val": val_loader}, {"train": train_ds, "val": val_ds}
