# src/common/utils/paths.py
# Utilities for locating the dataset root on disk.

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def _looks_like_imagenet_folder(root: Path, train_subdir: str, val_subdir: str) -> bool:
    """
    Check if 'root' looks like an ImageFolder dataset root:
      root/train_subdir/<class_name>/*
      root/val_subdir/<class_name>/*
    """
    if not root.exists() or not root.is_dir():
        return False
    if not (root / train_subdir).exists():
        return False
    if not (root / val_subdir).exists():
        return False
    return True


def find_dataset_root(
    repo_root: Path,
    preferred: Optional[Path],
    train_subdir: str = "train",
    val_subdir: str = "val",
    dataset_folder_name: str = "ImageNetSubset",
) -> Path:
    """
    Locate dataset root using a robust strategy.

    Strategy:
    1) Use 'preferred' if valid.
    2) Try common locations relative to repo_root.
    3) Try sibling repo patterns (common with git worktrees).

    Args:
        repo_root: The folder you run training from (repo root).
        preferred: Optional explicit dataset root.
        train_subdir: Folder name for training split.
        val_subdir: Folder name for validation split.
        dataset_folder_name: The dataset folder name (default: ImageNetSubset).

    Returns:
        Path to the dataset root.
    """
    if preferred is not None:
        preferred = Path(preferred)
        if _looks_like_imagenet_folder(preferred, train_subdir, val_subdir):
            return preferred

    candidates: List[Path] = [
        repo_root / dataset_folder_name,
        repo_root / "data" / dataset_folder_name,
        repo_root.parent / dataset_folder_name,
        # Typical worktree layout:
        # .../Repo-ML-PRO/Fatemeh (worktree)
        # .../Repo-ML-PRO/Repo-xAI-Proj-B/ImageNetSubset (dataset)
        repo_root.parent / "Repo-xAI-Proj-B" / dataset_folder_name,
    ]

    for c in candidates:
        if _looks_like_imagenet_folder(c, train_subdir, val_subdir):
            return c

    msg = (
        "Could not auto-detect dataset root.\n"
        "Tried candidates:\n- " + "\n- ".join(str(p) for p in candidates) + "\n\n"
        "Fix:\n"
        "1) Put the dataset in one of these locations, OR\n"
        "2) Set dataset_root in the config to the correct path.\n\n"
        "Expected structure:\n"
        f"{dataset_folder_name}/\n"
        f"  {train_subdir}/class_name/*.jpg\n"
        f"  {val_subdir}/class_name/*.jpg\n"
    )
    raise FileNotFoundError(msg)
