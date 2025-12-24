# scaffold_model_a_files.py
# Purpose:
# Create a clean scaffold for BOTH Model A and Model B, with shared code in src/common/.
# Run this script from your repo root (the folder where you want configs/, src/, outputs/ created).

from __future__ import annotations

from pathlib import Path


def write_text_file(path: Path, content: str, overwrite: bool = False) -> None:
    """
    Write a text file (UTF-8).

    - If overwrite=False and file exists, it will be skipped.
    - If overwrite=True, file content will be replaced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        print(f"[SKIP] {path} already exists")
        return

    path.write_text(content, encoding="utf-8")
    print(f"[WRITE] {path}")


def mkdir(path: Path) -> None:
    """Create a directory (including parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    print(f"[DIR]  {path}")


def main() -> None:
    repo_root = Path.cwd()  # This is where the scaffold will be created.
    overwrite = False       # Set True only if you want to overwrite existing files.

    # -----------------------
    # Folder structure
    # -----------------------
    folders = [
        repo_root / "configs",
        repo_root / "src",
        repo_root / "src" / "common",
        repo_root / "src" / "common" / "data",
        repo_root / "src" / "common" / "models",
        repo_root / "src" / "common" / "utils",
        repo_root / "src" / "model_a",
        repo_root / "src" / "model_b",
        repo_root / "outputs" / "model_a" / "checkpoints",
        repo_root / "outputs" / "model_a" / "logs",
        repo_root / "outputs" / "model_a" / "metrics",
        repo_root / "outputs" / "model_b" / "checkpoints",
        repo_root / "outputs" / "model_b" / "logs",
        repo_root / "outputs" / "model_b" / "metrics",
    ]

    for f in folders:
        mkdir(f)

    # -----------------------
    # Files (shared + per-model)
    # -----------------------
    files: dict[Path, str] = {}

    # Package init files (so imports work cleanly)
    files[repo_root / "src" / "__init__.py"] = "# src package\n"
    files[repo_root / "src" / "common" / "__init__.py"] = "# shared code for all models\n"
    files[repo_root / "src" / "common" / "data" / "__init__.py"] = "# shared data pipeline\n"
    files[repo_root / "src" / "common" / "models" / "__init__.py"] = "# shared model components\n"
    files[repo_root / "src" / "common" / "utils" / "__init__.py"] = "# shared utilities\n"
    files[repo_root / "src" / "model_a" / "__init__.py"] = "# Model A package\n"
    files[repo_root / "src" / "model_b" / "__init__.py"] = "# Model B package\n"

    # Shared modules (placeholders you will fill later)
    files[repo_root / "src" / "common" / "utils" / "seed.py"] = (
        "# seed.py\n"
        "# Set random seeds for reproducibility.\n"
    )
    files[repo_root / "src" / "common" / "utils" / "metrics.py"] = (
        "# metrics.py\n"
        "# Common metrics like top-1 accuracy.\n"
    )
    files[repo_root / "src" / "common" / "utils" / "checkpoint.py"] = (
        "# checkpoint.py\n"
        "# Save and load checkpoints.\n"
    )
    files[repo_root / "src" / "common" / "utils" / "paths.py"] = (
        "# paths.py\n"
        "# Find dataset root (e.g., ImageNetSubset) robustly.\n"
    )
    files[repo_root / "src" / "common" / "data" / "imagenet_folder.py"] = (
        "# imagenet_folder.py\n"
        "# Shared ImageFolder dataloader + transforms.\n"
    )
    files[repo_root / "src" / "common" / "models" / "backbone.py"] = (
        "# backbone.py\n"
        "# Build pretrained backbones (e.g., ResNet18/ResNet50).\n"
    )
    files[repo_root / "src" / "common" / "models" / "classifier.py"] = (
        "# classifier.py\n"
        "# Shared classifier head / wrapper model.\n"
    )

    # Model-specific entry points (A vs B)
    files[repo_root / "src" / "model_a" / "train.py"] = (
        "# train.py (Model A)\n"
        "# Model A: linear probing (freeze backbone, train only head).\n"
        "# Example run (from repo root):\n"
        "#   python -m src.model_a.train --config configs/model_a.yaml\n"
    )
    files[repo_root / "src" / "model_b" / "train.py"] = (
        "# train.py (Model B)\n"
        "# Model B: fine-tuning (unfreeze some/all backbone layers).\n"
        "# Example run (from repo root):\n"
        "#   python -m src.model_b.train --config configs/model_b.yaml\n"
    )

    # Configs
    files[repo_root / "configs" / "model_a.yaml"] = (
        "# Model A config (linear probing)\n"
        "dataset_root: \"\"            # leave empty to auto-detect\n"
        "train_subdir: \"train\"\n"
        "val_subdir: \"val\"\n"
        "\n"
        "backbone: \"resnet18\"\n"
        "pretrained: true\n"
        "freeze_backbone: true\n"
        "\n"
        "image_size: 224\n"
        "batch_size: 64\n"
        "epochs: 10\n"
        "learning_rate: 0.001\n"
        "weight_decay: 0.0001\n"
        "optimizer: \"adam\"\n"
        "\n"
        "seed: 42\n"
        "num_workers: 4\n"
        "device: \"auto\"\n"
        "\n"
        "output_dir: \"outputs/model_a\"\n"
    )
    files[repo_root / "configs" / "model_b.yaml"] = (
        "# Model B config (fine-tuning)\n"
        "dataset_root: \"\"            # leave empty to auto-detect\n"
        "train_subdir: \"train\"\n"
        "val_subdir: \"val\"\n"
        "\n"
        "backbone: \"resnet18\"\n"
        "pretrained: true\n"
        "freeze_backbone: false        # important difference vs Model A\n"
        "\n"
        "image_size: 224\n"
        "batch_size: 32                # often smaller for fine-tuning\n"
        "epochs: 10\n"
        "learning_rate: 0.0001         # often smaller for fine-tuning\n"
        "weight_decay: 0.0001\n"
        "optimizer: \"adam\"\n"
        "\n"
        "seed: 42\n"
        "num_workers: 4\n"
        "device: \"auto\"\n"
        "\n"
        "output_dir: \"outputs/model_b\"\n"
    )

    # Requirements + README (minimal placeholders)
    files[repo_root / "requirements.txt"] = (
        "torch\n"
        "torchvision\n"
        "pyyaml\n"
        "tqdm\n"
        "pillow\n"
        "numpy\n"
    )
    files[repo_root / "README.md"] = (
        "# Models A & B Scaffold\n\n"
        "This scaffold separates shared code in `src/common/` and model-specific code in\n"
        "`src/model_a/` and `src/model_b/`.\n"
    )

    # Write files
    for path, content in files.items():
        write_text_file(path, content, overwrite=overwrite)

    print("\nDone.")
    print("Next steps:")
    print("1) Fill shared modules in src/common/")
    print("2) Implement train scripts in src/model_a/train.py and src/model_b/train.py")


if __name__ == "__main__":
    main()
