# src/common/infer_folder.py
# Predict ALL images in a folder and save results as a CSV "table".
# The table includes: image name + top-k predicted classes + probabilities.
# Additionally, it saves a separate CSV for files that cannot be opened (bad_files.csv).

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms

from src.common.utils.checkpoint import load_checkpoint
from src.common.models.classifier import ImageClassifier


def build_infer_transform(image_size: int) -> transforms.Compose:
    """
    Build validation-style preprocessing:
    Resize -> CenterCrop -> ToTensor -> Normalize (ImageNet stats).
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def iter_images(folder: Path):
    """Yield all image files under 'folder' (recursively)."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def default_output_csv_from_ckpt(ckpt_path: Path) -> Path:
    """
    If ckpt is like: outputs/model_b/checkpoints/best.pt
    then default CSV will be: outputs/model_b/metrics/predictions.csv
    """
    # ckpt_path.parent = checkpoints
    if ckpt_path.parent.name == "checkpoints" and len(ckpt_path.parents) >= 2:
        model_dir = ckpt_path.parents[1]  # outputs/model_b
    else:
        model_dir = ckpt_path.parent

    return model_dir / "metrics" / "predictions.csv"


def format_rows_as_text_table(header: List[str], rows: List[List[str]], max_rows: int = 20) -> str:
    """
    Create a simple fixed-width text table for console printing (no extra packages).
    This is only a preview. Full results are always saved to CSV.
    """
    show_rows = rows[:max_rows]
    widths = [len(h) for h in header]
    for r in show_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(r: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))

    line = "-+-".join("-" * w for w in widths)

    out = []
    out.append(fmt_row(header))
    out.append(line)
    for r in show_rows:
        out.append(fmt_row(r))

    if len(rows) > max_rows:
        out.append(f"... ({len(rows) - max_rows} more rows not shown)")

    return "\n".join(out)


def safe_open_as_rgb(img_path: Path) -> Image.Image:
    """
    Safely open an image and convert it to RGB.

    - Handles palette images (mode 'P') with transparency by converting to RGBA first,
      then to RGB (drops alpha).
    - Converts any non-RGB image to RGB.
    """
    img = Image.open(img_path)

    # Some images are palette-based ('P') and may have transparency.
    # Converting P -> RGBA first avoids common Pillow warnings.
    if img.mode == "P":
        img = img.convert("RGBA").convert("RGB")
    elif img.mode == "RGBA":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (best.pt).")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing images.")
    parser.add_argument("--topk", type=int, default=3, help="How many top predictions per image.")
    parser.add_argument("--out", type=str, default="", help="Optional output CSV path.")
    parser.add_argument("--preview_rows", type=int, default=20, help="How many rows to preview in console.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.ckpt)
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")

    cfg = ckpt.get("config", {})
    class_names = ckpt["class_names"]

    backbone = str(cfg.get("backbone", "resnet18"))
    pretrained = bool(cfg.get("pretrained", True))
    freeze_backbone = bool(cfg.get("freeze_backbone", False))  # not important for inference, ok

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

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    topk = max(1, int(args.topk))
    out_csv = Path(args.out) if str(args.out).strip() else default_output_csv_from_ckpt(ckpt_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Bad files CSV path
    bad_csv = out_csv.parent / "bad_files.csv"

    # CSV header: image, top1_label, top1_prob, top2_label, top2_prob, ...
    header = ["image"]
    for i in range(1, topk + 1):
        header += [f"top{i}_label", f"top{i}_prob"]

    csv_rows: List[List[str]] = []
    printable_rows: List[List[str]] = []
    bad_rows: List[List[str]] = []  # [image_name, reason]

    count_ok = 0
    count_skip = 0

    for img_path in iter_images(folder):
        try:
            img = safe_open_as_rgb(img_path)
        except Exception as e:
            reason = str(e).replace("\n", " ").strip()
            print(f"[SKIP] {img_path.name} (cannot open: {reason})")
            bad_rows.append([img_path.name, reason])
            count_skip += 1
            continue

        x = tfm(img).unsqueeze(0).to(device)  # [1, C, H, W]
        probs = torch.softmax(model(x), dim=1).squeeze(0)  # [C]

        k = min(topk, len(class_names))
        values, indices = torch.topk(probs, k=k)

        row = [img_path.name]
        pretty = [img_path.name]

        for p, idx in zip(values.tolist(), indices.tolist()):
            label = class_names[idx]
            prob = f"{p:.4f}"
            row += [label, prob]
            pretty += [label, prob]

        # If k < topk, fill empty columns to keep CSV rectangular
        missing = topk - k
        for _ in range(missing):
            row += ["", ""]
            pretty += ["", ""]

        csv_rows.append(row)
        printable_rows.append(pretty)
        count_ok += 1

    # Write predictions CSV table
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)

    print("\nSaved table (CSV):", out_csv)
    print(f"Predicted {count_ok} images. Skipped {count_skip} files.")

    # Write bad files CSV (if any)
    if bad_rows:
        with bad_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "reason"])
            writer.writerows(bad_rows)
        print("Saved bad files list (CSV):", bad_csv)

    # Print a small table preview in console
    preview_n = max(1, int(args.preview_rows))
    print(f"\nPreview (first {preview_n} rows):")
    print(format_rows_as_text_table(header, printable_rows, max_rows=preview_n))


if __name__ == "__main__":
    main()
