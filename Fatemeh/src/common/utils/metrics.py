# src/common/utils/metrics.py
# Basic classification metrics.

from __future__ import annotations

import torch


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy.

    Args:
        logits: Model outputs with shape [B, C]
        targets: Ground truth labels with shape [B]

    Returns:
        Accuracy as a float in [0, 1].
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / max(total, 1)
