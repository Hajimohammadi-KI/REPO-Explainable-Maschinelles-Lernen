# Models A & B (Scaffold)

This project contains:
- Shared code in `src/common/`
- Model-specific entry points:
  - `src/model_a/train.py` (linear probing)
  - `src/model_b/train.py` (fine-tuning)

## Install
pip install -r requirements.txt

## Train Model A
python -m src.model_a.train --config configs/model_a.yaml

## Train Model B
python -m src.model_b.train --config configs/model_b.yaml

## Optional: explicitly set dataset root
python -m src.model_a.train --config configs/model_a.yaml --data "C:/path/to/ImageNetSubset"
