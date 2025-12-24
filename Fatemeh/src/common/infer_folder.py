# src/common/infer_folder.py
# Predict ALL images in a folder and save results as a CSV table.
# Auto-detects repo root, default image folder, and default checkpoint path
# so it can run on different machines without changing hard-coded paths.

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from torchvision import transforms

from src.common.utils.checkpoint import load_checkpoint
from src.common.models.classifier import ImageClassifier


def find_repo_root(start: Path) -> Path:
    """
    Find repository root by walking up until we see both 'src/' and 'configs/'.
    This makes the script portable across different machines.

    Args:
        start: starting folder to search from

    Returns:
        repo root path
    """
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "src").exists() and (p / "configs").exists():
            return p
    # fallback (should not happen if project structure is correct)
    return start


def resolve_path(repo_root: Path, user_path: Optional[str]) -> Optional[Path]:
    """
    Convert a user-provided path into an absolute path.
    If user_path is relative, interpret it relative to repo_root.
    """
    if user_path is None:
        return None
    s = str(user_path).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def default_ckpt_path(repo_root: Path, model: str) -> Path:
    """
    Default checkpoint location:
      outputs/model_a/checkpoints/best.pt  or
      outputs/model_b/checkpoints/best.pt
    """
    model = model.lower().strip()
    if model not in {"a", "b"}:
        raise ValueError("model must be 'a' or 'b'")
    return repo_root / "outputs" / f"model_{model}" / "checkpoints" / "best.pt"


def find_latest_checkpoint(repo_root: Path) -> Optional[Path]:
    """
    Fallback: find the most recently modified .pt file under outputs/**/checkpoints/.
    """
    candidates = list(repo_root.glob("outputs/**/checkpoints/*.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def build_infer_transform(image_size: int) -> transforms.Compose:
    """
    Validation-style preprocessing:
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
    then default CSV: outputs/model_b/metrics/predictions.csv
    """
    if ckpt_path.parent.name == "checkpoints" and len(ckpt_path.parents) >= 2:
        model_dir = ckpt_path.parents[1]  # outputs/model_b
    else:
        model_dir = ckpt_path.parent
    return model_dir / "metrics" / "predictions.csv"


def format_rows_as_text_table(header: List[str], rows: List[List[str]], max_rows: int = 20) -> str:
    """Console preview table (only a preview; full results are saved to CSV)."""
    show_rows = rows[:max_rows]
    widths = [len(h) for h in header]
    for r in show_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(r: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))

    line = "-+-".join("-" * w for w in widths)

    out = [fmt_row(header), line]
    out.extend(fmt_row(r) for r in show_rows)

    if len(rows) > max_rows:
        out.append(f"... ({len(rows) - max_rows} more rows not shown)")

    return "\n".join(out)


def safe_open_as_rgb(img_path: Path) -> Image.Image:
    """
    Safely open an image and convert it to RGB.

    Handles palette images ('P') with transparency by converting P -> RGBA -> RGB.
    """
    img = Image.open(img_path)
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
    parser.add_argument("--model", type=str, default="b", help="Which model to use: a or b (default: b).")
    parser.add_argument("--ckpt", type=str, default="", help="Optional checkpoint path. If empty, auto-detect.")
    parser.add_argument("--folder", type=str, default="", help="Optional image folder. If empty, use repo_root/data.")
    parser.add_argument("--topk", type=int, default=3, help="How many top predictions per image.")
    parser.add_argument("--out", type=str, default="", help="Optional output CSV path.")
    parser.add_argument("--preview_rows", type=int, default=20, help="How many rows to preview in console.")
    args = parser.parse_args()

    # Determine repo root based on this file location (portable)
    this_file = Path(__file__).resolve()
    repo_root = find_repo_root(this_file.parent)
    # repo_root should be .../<repo>
    # because infer_folder.py is .../<repo>/src/common/infer_folder.py

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Repo root: {repo_root}")

    # Resolve checkpoint path
    ckpt_path = resolve_path(repo_root, args.ckpt)
    if ckpt_path is None:
        ckpt_path = default_ckpt_path(repo_root, args.model)

    if not ckpt_path.exists():
        latest = find_latest_checkpoint(repo_root)
        if latest is not None:
            print(f"[WARN] Default checkpoint not found: {ckpt_path}")
            print(f"[INFO] Using latest checkpoint instead: {latest}")
            ckpt_path = latest
        else:
            raise FileNotFoundError(
                f"Checkpoint not found.\nTried: {ckpt_path}\n"
                f"Fix: train a model first, or pass --ckpt path/to/best.pt"
            )

    # Resolve folder path
    folder = resolve_path(repo_root, args.folder)
    if folder is None:
        folder = repo_root / "data"

    if not folder.exists():
        raise FileNotFoundError(
            f"Image folder not found: {folder}\n"
            "Fix: put images in repo_root/data OR pass --folder <path>"
        )

    print(f"Checkpoint: {ckpt_path}")
    print(f"Image folder: {folder}")

    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    class_names = ckpt["class_names"]

    backbone = str(cfg.get("backbone", "resnet18"))
    pretrained = bool(cfg.get("pretrained", True))
    freeze_backbone = bool(cfg.get("freeze_backbone", False))  # doesn't matter for inference

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

    topk = max(1, int(args.topk))

    out_csv = resolve_path(repo_root, args.out)
    if out_csv is None:
        out_csv = default_output_csv_from_ckpt(ckpt_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    bad_csv = out_csv.parent / "bad_files.csv"

    # CSV header
    header = ["image"]
    for i in range(1, topk + 1):
        header += [f"top{i}_label", f"top{i}_prob"]

    csv_rows: List[List[str]] = []
    printable_rows: List[List[str]] = []
    bad_rows: List[List[str]] = []

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

        # Keep CSV rectangular
        missing = topk - k
        for _ in range(missing):
            row += ["", ""]
            pretty += ["", ""]

        csv_rows.append(row)
        printable_rows.append(pretty)
        count_ok += 1

    # Write predictions CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)

    print("\nSaved table (CSV):", out_csv)
    print(f"Predicted {count_ok} images. Skipped {count_skip} files.")

    # Write bad files CSV
    if bad_rows:
        with bad_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "reason"])
            writer.writerows(bad_rows)
        print("Saved bad files list (CSV):", bad_csv)

    # Console preview
    preview_n = max(1, int(args.preview_rows))
    print(f"\nPreview (first {preview_n} rows):")
    print(format_rows_as_text_table(header, printable_rows, max_rows=preview_n))


if __name__ == "__main__":
    main()
