# src/common/utils/seed.py
# Utilities for reproducibility.

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Any fixed integer (e.g., 42).
        deterministic: If True, enforces more deterministic CUDA behavior
                       (can be slower, but more reproducible).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
