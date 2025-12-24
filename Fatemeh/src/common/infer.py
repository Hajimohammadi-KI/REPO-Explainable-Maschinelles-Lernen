# src/common/infer.py
# Inference (prediction) on a single image using a trained checkpoint.
# Works for BOTH Model A and Model B because it loads everything from the checkpoint.

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.common.utils.checkpoint import load_checkpoint
from src.common.models.classifier import ImageClassifier


def build_infer_transform(image_size: int) -> transforms.Compose:
    """
    Build the same validation-style preprocessing used during training:
    - Resize -> CenterCrop -> ToTensor -> Normalize (ImageNet stats)
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--image", type=str, required=True, help="Path to an input image (jpg/png).")
    parser.add_argument("--topk", type=int, default=3, help="How many top predictions to print.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")

    # These are the class labels (folder names) from your dataset.
    class_names = ckpt["class_names"]

    # We saved the training config inside the checkpoint.
    cfg = ckpt.get("config", {})
    backbone = str(cfg.get("backbone", "resnet18"))
    pretrained = bool(cfg.get("pretrained", True))

    # freeze_backbone does not matter for inference, but we keep it consistent.
    freeze_backbone = bool(cfg.get("freeze_backbone", False))

    model = ImageClassifier(
        backbone_name=backbone,
        num_classes=len(class_names),
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    image_size = int(cfg.get("image_size", 224))
    tfm = build_infer_transform(image_size)

    img_path = Path(args.image)
    img = Image.open(img_path).convert("RGB")

    x = tfm(img).unsqueeze(0).to(device)  # [1, C, H, W]

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)  # [C]

    k = min(int(args.topk), len(class_names))
    values, indices = torch.topk(probs, k=k)

    print("\nTop predictions:")
    for p, idx in zip(values.tolist(), indices.tolist()):
        print(f"- {class_names[idx]} : {p:.4f}")


if __name__ == "__main__":
    main()
