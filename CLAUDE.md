# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

PyTorch transfer-learning classifiers (ResNet50, ImageNet V2 weights) for two medical imaging tasks, plus the MLOps around them:

| Task key | Classes | Data dir | Artifacts dir |
|---|---|---|---|
| `brain_tumor` | glioma, meningioma, notumor, pituitary | `data/Training|Testing/<class>/` | `artifacts_brain/` |
| `breast_cancer` | benign, malignant, normal | `data_breast/<class>/` (mask files skipped) | `artifacts_breast/` |

## Files

- `mlops_pipeline.py` — the only training entry point. Ten stages: config → Kaggle download (kagglehub) → data validation → splits/transforms → MLflow run → model → training (early stopping) → test evaluation → quality gate → TorchScript export + MLflow registry. Per-task settings (dataset slug, layout, gate thresholds) live in `TASK_META`.
- `model.py` — `build_model`, `eval_transform`, `load_checkpoint`. Shared by every other script; the Dockerfile copies only `model.py` + `serve.py`, so keep it free of training-only deps.
- `explain.py --task <task>` — Grad-CAM, Grad-CAM++, Integrated Gradients, Occlusion; samples from `<artifacts>/test_images.csv`.
- `serve.py` — FastAPI (`/health`, `/predict/{task}`); `batch_predict.py` is its CLI client.
- `patient_app.py`, `dashboard.py` — Streamlit apps (deployed on Streamlit Cloud; fetch artifacts from HF Hub `Slakje89/medical-imaging-models` when missing; deps in `requirements-app.txt`).
- `documentation/` — stage-by-stage reference docs that quote function names from `mlops_pipeline.py`. Update them if you rename those functions.

## Common commands

```bash
python mlops_pipeline.py --task brain_tumor                    # full run, auto-downloads data
python mlops_pipeline.py --task breast_cancer --epochs 25 --batch 32
python explain.py --task brain_tumor --n 8
uvicorn serve:app --port 8000
streamlit run patient_app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Kaggle credentials: `--kaggle_user/--kaggle_key` → `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars → `~/.kaggle/kaggle.json`.

There is no test suite. Quick end-to-end check: create a small synthetic dataset (≥10 images per class) and run `mlops_pipeline.py --data <dir> --out <dir> --epochs 1 --img_size 64 --mlflow_uri sqlite:///<tmp>/mlflow.db`.

## Checkpoint contract

`best_model.pt` holds `state_dict`, `classes`, `img_size`, `mean`, `std`, `epoch`, `val_acc`, and is loaded with `weights_only=True`. Serving, the apps and `explain.py` all depend on it via `model.load_checkpoint`, so change it in one place only.

## CI

`.github/workflows/mlops.yml` trains `brain_tumor` for one epoch on CPU as a smoke test (30% accuracy floor), then builds the Docker image. It proves the pipeline runs, not that the model is good.

## Platform notes

Primary dev environment is Windows (Git-Bash / PowerShell). `chmod(0o600)` on `kaggle.json` is wrapped in try/except because Windows has no POSIX perms.
