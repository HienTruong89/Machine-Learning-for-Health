# End-to-End Patient-Facing ML App — Reference Documentation

A reusable blueprint for building a patient-facing medical-imaging classifier in PyTorch, distilled from the brain-tumor / breast-cancer system in this repository. Use it as a template for the next project.

The architecture is a single linear pipeline with explicit stages, each one boundary-checked and independently runnable. Every stage below maps to a real function in `mlops_pipeline.py`, `serve.py`, or `patient_app.py`.

## Start here

- **[00_mlops_pipeline.md](00_mlops_pipeline.md)** — single-page runbook for the whole pipeline. Read this first.
- **[14_new_dataset_checklist.md](14_new_dataset_checklist.md)** — concrete top-to-bottom recipe for adapting the template to a new dataset.
- **[13_cli_tools.md](13_cli_tools.md)** — install + auth + commands for Git, Docker, Azure, Kaggle, Hugging Face CLIs.

## The 12 stages

| # | Stage | Doc | Source |
|---|-------|-----|--------|
| 1 | Configuration & reproducibility | [01_configuration.md](01_configuration.md) | `mlops_pipeline.py` :: `Config`, `set_seed` |
| 2 | Data acquisition | [02_data_acquisition.md](02_data_acquisition.md) | `mlops_pipeline.py` :: `download_dataset` |
| 3 | Data validation | [03_data_validation.md](03_data_validation.md) | `mlops_pipeline.py` :: `index_images`, `validate_data` |
| 4 | Dataset, transforms, splits | [04_dataset_and_transforms.md](04_dataset_and_transforms.md) | `mlops_pipeline.py` :: `ImageDataset`, `build_transforms`, `build_splits` |
| 5 | Model architecture | [05_model.md](05_model.md) | `model.py` :: `build_model` |
| 6 | Training loop | [06_training.md](06_training.md) | `mlops_pipeline.py` :: `run_epoch`, `EarlyStopping` |
| 7 | Experiment tracking (MLflow) | [07_experiment_tracking.md](07_experiment_tracking.md) | `mlops_pipeline.py` :: `mlflow.start_run` block |
| 8 | Evaluation | [08_evaluation.md](08_evaluation.md) | `mlops_pipeline.py` :: `evaluate_test_set` |
| 9 | Quality gate & registry | [09_validation_gate.md](09_validation_gate.md) | `mlops_pipeline.py` :: `run_validation_gate`, `export_torchscript` |
| 10 | Inference API (FastAPI + Docker) | [10_serving.md](10_serving.md) | `serve.py`, `Dockerfile` |
| 11 | Patient-facing UI (Streamlit) | [11_patient_app.md](11_patient_app.md) | `patient_app.py` |
| 12 | CI/CD (GitHub Actions) | [12_cicd.md](12_cicd.md) | `.github/workflows/mlops.yml` |
| ~ | CLI tools (Git, Docker, Azure, Kaggle, HF) | [13_cli_tools.md](13_cli_tools.md) | external |
| ~ | New-dataset checklist | [14_new_dataset_checklist.md](14_new_dataset_checklist.md) | this template |
| ~ | Adapting to other domains (tabular / text / segmentation / regression) | [15_adapting_to_other_domains.md](15_adapting_to_other_domains.md) | this template |

## How to use this for the next project

1. Copy `model.py`, `mlops_pipeline.py`, `serve.py`, `patient_app.py`, `Dockerfile` and the workflow into the new repo.
2. In `mlops_pipeline.py`, add a new entry to `TASK_META` describing the dataset (Kaggle slug, folder layout, image extensions, gates).
3. If the dataset isn't on Kaggle, replace `download_dataset` with your acquisition step but keep the contract: produce a folder of `<class>/*.png|jpg`.
4. Re-run `python mlops_pipeline.py --task <new_task>`. The rest of the pipeline (validation → training → gate → registry) is task-agnostic.
5. Add a new task entry to `TASKS` in `patient_app.py` with friendly names and risk flags.

## Design choices baked into this template

- **One pipeline script, ten stages.** Easier to reason about than a DAG. Each stage is a function with a clear input/output.
- **Transfer learning on ResNet50 (ImageNet V2 weights).** Strong default for medical imaging with small datasets; the new head is a 2-layer MLP with dropout.
- **Class-frequency-inverse weighting + label smoothing.** Cheap, effective for class-imbalanced medical datasets.
- **AMP + AdamW + CosineAnnealingLR + early stopping.** Safe defaults that rarely need tuning.
- **Quality gate before registry.** Models below the accuracy/AUROC thresholds in `TASK_META` are saved as artifacts but never registered, so deployment can't accidentally pick up a bad model.
- **TorchScript export alongside the `.pt`.** Decouples deployment from the training environment.
- **Hugging Face Hub for app distribution.** The Streamlit app downloads weights at first run, so the container/repo stays small and credentials stay out of the image.
- **Strict separation of training and serving deps.** The `Dockerfile` installs `torch + torchvision + fastapi` only — no `mlflow`, `sklearn`, `kagglehub`.

## Layout produced by the pipeline

```
artifacts_<task>/
├── config.json            # exact Config used (Stage 1)
├── data_stats.json        # per-class counts + dataset fingerprint (Stage 3)
├── history.csv            # per-epoch metrics (Stage 6)
├── best_model.pt          # state_dict + classes/img_size/mean/std (Stage 6)
├── model.torchscript      # portable inference graph (Stage 9)
└── test_report.json       # classification report + confusion matrix + AUROC (Stage 8)
```
