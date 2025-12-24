# train.py (Model A)
# Model A: linear probing (freeze backbone, train only head).
# Example run (from repo root):
#   python -m src.model_a.train --config configs/model_a.yaml
# src/model_a/train.py
# Model A: linear probing (freeze backbone, train only head).

from __future__ import annotations 

import argparse
import csv
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

from src.common.utils.seed import set_seed
from src.common.utils.metrics import accuracy_top1
from src.common.utils.checkpoint import save_checkpoint
from src.common.utils.paths import find_dataset_root
from src.common.data.imagenet_folder import DataConfig, create_dataloaders
from src.common.models.classifier import ImageClassifier


def pick_device(device_cfg: str) -> torch.device:
    device_cfg = device_cfg.lower()
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but CUDA is not available.")
        return torch.device("cuda")
    raise ValueError(f"Unknown device option: {device_cfg}")


def build_optimizer(name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        acc = accuracy_top1(logits, targets)

        loss_sum += loss.item()
        acc_sum += acc
        n += 1

    return {"loss": loss_sum / max(n, 1), "acc": acc_sum / max(n, 1)}


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)
        acc = accuracy_top1(logits, targets)

        loss_sum += loss.item()
        acc_sum += acc
        n += 1

    return {"loss": loss_sum / max(n, 1), "acc": acc_sum / max(n, 1)}


def append_metrics_csv(path: Path, epoch: int, train_stats: Dict[str, float], val_stats: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerow([epoch, train_stats["loss"], train_stats["acc"], val_stats["loss"], val_stats["acc"]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_a.yaml", help="Path to YAML config.")
    parser.add_argument("--data", type=str, default="", help="Optional dataset root override.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    set_seed(int(cfg["seed"]), deterministic=True)
    device = pick_device(str(cfg["device"]))
    print(f"Device: {device}")

    preferred = None
    if str(args.data).strip():
        preferred = Path(args.data)
    elif str(cfg.get("dataset_root", "")).strip():
        preferred = Path(cfg["dataset_root"])

    dataset_root = find_dataset_root(
        repo_root=repo_root,
        preferred=preferred,
        train_subdir=str(cfg["train_subdir"]),
        val_subdir=str(cfg["val_subdir"]),
        dataset_folder_name=str(cfg.get("dataset_folder_name", "ImageNetSubset")),
    )
    print(f"Dataset root: {dataset_root}")

    data_cfg = DataConfig(
        dataset_root=dataset_root,
        train_subdir=str(cfg["train_subdir"]),
        val_subdir=str(cfg["val_subdir"]),
        image_size=int(cfg["image_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
    )
    dataloaders, datasets_map = create_dataloaders(data_cfg)

    class_names = datasets_map["train"].classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Model A: backbone frozen (linear probing)
    model = ImageClassifier(
        backbone_name=str(cfg["backbone"]),
        num_classes=num_classes,
        pretrained=bool(cfg["pretrained"]),
        freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # IMPORTANT: train only the head parameters
    optimizer = build_optimizer(
        name=str(cfg["optimizer"]),
        params=model.head.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    output_dir = repo_root / str(cfg["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics" / "metrics.csv"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    best_path = ckpt_dir / "best.pt"

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_stats = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_stats = evaluate(model, dataloaders["val"], criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_stats['loss']:.4f} acc={train_stats['acc']:.4f} | "
            f"val loss={val_stats['loss']:.4f} acc={val_stats['acc']:.4f}"
        )

        append_metrics_csv(metrics_path, epoch, train_stats, val_stats)

        if val_stats["acc"] > best_acc:
            best_acc = val_stats["acc"]
            save_checkpoint(
                best_path,
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": best_acc,
                    "class_names": class_names,
                    "config": cfg,
                },
            )
            print(f"Saved best checkpoint: {best_path} (val_acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
