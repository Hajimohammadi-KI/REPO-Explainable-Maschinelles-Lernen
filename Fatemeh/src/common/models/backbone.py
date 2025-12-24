# src/common/models/backbone.py
# Build pretrained backbones (e.g., ResNet18/ResNet50) that output feature vectors.

from __future__ import annotations

from typing import Tuple

import torch.nn as nn
from torchvision import models


def build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """
    Build a backbone model and return (backbone_module, feature_dim).

    The backbone output is a feature vector, so we replace the final FC layer with Identity.

    Args:
        name: "resnet18" or "resnet50"
        pretrained: whether to load pretrained weights

    Returns:
        backbone: nn.Module that outputs [B, feat_dim]
        feat_dim: int feature dimension
    """
    name = name.lower()

    # Torchvision API differs across versions. We support both:
    # - New API: weights=... (preferred)
    # - Old API: pretrained=True/False (deprecated, but may exist)
    if name == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            m = models.resnet18(weights=weights)
        except Exception:
            m = models.resnet18(pretrained=pretrained)

        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim

    if name == "resnet50":
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            m = models.resnet50(weights=weights)
        except Exception:
            m = models.resnet50(pretrained=pretrained)

        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim

    raise ValueError(f"Unsupported backbone: {name}")
