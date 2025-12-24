# src/common/utils/checkpoint.py
# Saving and loading checkpoints.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    """
    Save a checkpoint to disk.

    Args:
        path: Output .pt file path
        state: Dictionary containing model/optimizer states and metadata
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Load a checkpoint from disk.

    Args:
        path: Input .pt file path
        map_location: Usually "cpu" for safe loading

    Returns:
        The checkpoint dictionary.
    """
    return torch.load(path, map_location=map_location)
