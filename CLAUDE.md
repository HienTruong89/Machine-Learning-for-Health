# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

A single-script PyTorch project for multi-class brain tumor MRI classification (4 classes: glioma, meningioma, notumor, pituitary) using transfer learning on ResNet50. The companion `ML_Medical_Imaging_Tutorial.md` is a long-form tutorial document; `train_brain_tumor.py` is the actual runnable pipeline.

## Common commands

Run full pipeline (download dataset if missing, train, evaluate):

```bash
python train_brain_tumor.py
```

Skip auto-download (data already on disk) and override hyperparameters:

```bash
python train_brain_tumor.py --data data --epochs 20 --batch 32 --lr 3e-4 --img_size 224
```

Pass Kaggle credentials inline (otherwise resolved from `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars or `~/.kaggle/kaggle.json`):

```bash
python train_brain_tumor.py --kaggle_user <name> --kaggle_key <key>
```

There is no test suite, linter, or build step — the project is one script plus artifacts.

## Architecture

`train_brain_tumor.py` is a linear end-to-end pipeline with these stages in `main()`:

1. **Dataset acquisition** (`download_dataset`) — shells out to the Kaggle **CLI executable** (never the Python `kaggle` module, to avoid the `KaggleApiExtended`/`__main__.py` breakage in kaggle v1.6+). Credential resolution order: CLI flags → env vars → `~/.kaggle/kaggle.json`. Writes `kaggle.json` *before* any kaggle import because newer versions authenticate at import time. `_kaggle_exe()` searches PATH, then the conda `Scripts/` dir next to `sys.executable`, then user site-packages.
2. **Indexing** (`index_folder`) — walks `data/Training/<class>/` and `data/Testing/<class>/`, verifies each image with `PIL.Image.verify()`, and builds a DataFrame with `ok`/`err` flags so corrupt files are skipped rather than crashing training.
3. **Splits** — `Training/` is stratified-split 85/15 into train/val; `Testing/` is held out as the test set.
4. **Transforms** — grayscale→3-channel (so ImageNet-pretrained weights apply), ImageNet normalization, augmentation only on train (flip, rotate, jitter, affine, RandomErasing).
5. **Model** (`build_model`) — ResNet50 with ImageNet V2 weights; the final `fc` is replaced with a 2-layer MLP head (Dropout 0.3 → Linear 512 → ReLU → Dropout → Linear num_classes).
6. **Training** (`run_epoch`) — single function for both train and eval, branching on `optimizer is not None`. Uses `CrossEntropyLoss` with **class-frequency-inverse weights** and label smoothing 0.05, AdamW, CosineAnnealingLR, and `torch.cuda.amp` mixed-precision when CUDA is present. Best checkpoint (by val acc) is saved to `artifacts/best_model.pt` containing `state_dict`, `classes`, `img_size`, `mean`, `std` — everything needed for standalone inference.
7. **Evaluation** — reloads the best checkpoint and emits `artifacts/history.csv` and `artifacts/test_report.json` (classification report, confusion matrix, macro OvR AUROC).

## Expected data layout

```
data/Training/<class>/*.jpg
data/Testing/<class>/*.jpg
```

Any other layout will produce an empty index DataFrame and a `FileNotFoundError`.

## Platform notes

Primary dev environment is Windows (shell is bash/Git-Bash). The Kaggle credential file `chmod(0o600)` call is wrapped in try/except because Windows does not support POSIX perms. Use forward slashes in paths when invoking commands.
