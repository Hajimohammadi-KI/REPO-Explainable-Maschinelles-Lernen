# src/common/models/classifier.py
# Shared classifier wrapper: backbone + linear head.

from __future__ import annotations

import torch
import torch.nn as nn

from src.common.models.backbone import build_backbone


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Enable/disable gradients for all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = requires_grad


class ImageClassifier(nn.Module):
    """
    A simple image classifier:
      features = backbone(x)
      logits   = head(features)

    This is suitable for:
    - Model A: freeze backbone, train head only
    - Model B: unfreeze backbone, train all parameters
    """

    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool):
        super().__init__()

        self.backbone, feat_dim = build_backbone(backbone_name, pretrained=pretrained)

        if freeze_backbone:
            set_requires_grad(self.backbone, False)

        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits
