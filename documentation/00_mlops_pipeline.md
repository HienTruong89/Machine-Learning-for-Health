# MLOps pipeline — runbook

A consolidated walkthrough of `mlops_pipeline.py` from start to finish, tying together the per-stage docs (01–09). Read this first if you want the *whole loop* on one page; jump to a numbered doc if you want the depth.

## Mental model

The pipeline is a single Python script with ten ordered stages, each one a function. There are no DAG frameworks, no orchestrators, no plug-ins. It runs the same way on a laptop, on a CI runner, and inside a container.

```
┌─ 1  Configuration       ── Config dataclass + seed
├─ 2  Data acquisition    ── kagglehub download (idempotent)
├─ 3  Data validation     ── PIL.verify, fingerprint, hard-fail thresholds
├─ 4  Feature engineering ── transforms + stratified splits + DataLoaders
├─ 5  Experiment tracking ── mlflow.start_run wraps stages 6-10
├─ 6  Model building      ── ResNet50 + MLP head + class-weighted loss
├─ 7  Training            ── AMP + AdamW + cosine LR + early stopping
├─ 8  Evaluation          ── classification report + confusion + AUROC
├─ 9  Quality gate        ── two thresholds, binary pass/fail
└─10  Export & registry   ── TorchScript + conditional MLflow Registry
```

The deliverable is a directory of artifacts:

```
artifacts_<task>/
├── config.json          # what was attempted
├── data_stats.json      # what data was used (fingerprint inside)
├── history.csv          # how training went, per epoch
├── best_model.pt        # the model + metadata (classes / img_size / mean / std)
├── model.torchscript    # framework-free graph for portable inference
└── test_report.json     # how it performed, plus the MLflow run_id
```

…and an MLflow run holding the same things, queryable in the UI.

## How to run it

The pipeline is task-driven via `--task`:

```bash
# Default: brain tumor MRI, 20 epochs, gates from TASK_META
python mlops_pipeline.py --task brain_tumor

# Different task, custom hyperparameters
python mlops_pipeline.py --task breast_cancer --epochs 25 --batch 32

# Provide Kaggle creds inline (overrides ~/.kaggle/kaggle.json)
python mlops_pipeline.py --task brain_tumor \
  --kaggle_user NAME --kaggle_key KEY

# Smoke run with no quality gate (used by CI)
python mlops_pipeline.py --task brain_tumor \
  --epochs 1 --batch 16 --min_val_acc 0.0 --min_auroc 0.0
```

After it finishes:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://127.0.0.1:5000
```

## Worked walkthrough — what each stage does

### 1. Configuration (see [01_configuration.md](01_configuration.md))

`parse_args()` builds a `Config` dataclass merging CLI flags and `TASK_META[task]` defaults. `set_seed(cfg.seed)` makes the run reproducible. `cfg.save(out / "config.json")` writes the resolved config *before* any work, so a crash still leaves you the audit trail.

### 2. Data acquisition (see [02_data_acquisition.md](02_data_acquisition.md))

`download_dataset(cfg)` uses `kagglehub` to pull the dataset slug from `TASK_META`. Idempotent — re-runs return immediately if the data is already on disk. Credentials come from CLI flags → env vars → `~/.kaggle/kaggle.json`.

### 3. Data validation (see [03_data_validation.md](03_data_validation.md))

`index_images(cfg)` walks the data root and runs `PIL.Image.verify()` on every file, returning a DataFrame with an `ok` flag. `validate_data(df)` raises if corruption > 5% or smallest class < 10, then computes a `dataset_fingerprint` (md5 of sorted paths) and writes `data_stats.json`.

### 4. Feature engineering (see [04_dataset_and_transforms.md](04_dataset_and_transforms.md))

`build_splits(df, cfg)` carves train/val/test with `stratify=label`. `build_transforms(img_size)` returns separate train/eval pipelines (augmentations only on train). `DataLoader` with `num_workers=4`, `pin_memory=True`.

### 5. Experiment tracking (see [07_experiment_tracking.md](07_experiment_tracking.md))

A single `with mlflow.start_run(run_name=...)` block wraps stages 6–10. Tags capture the *what* (`task`, `dataset_fingerprint`, `device`); params capture the knobs; metrics capture the numbers. Artifacts are pushed at the end — `config.json`, `data_stats.json`, `history.csv`, `test_report.json`, the model itself.

### 6. Model building (see [05_model.md](05_model.md))

`build_model(num_classes)` returns a ResNet50 with the head replaced by a 2-layer MLP. Class-frequency-inverse weights go into `CrossEntropyLoss(weight=..., label_smoothing=0.05)`. AdamW + CosineAnnealingLR + GradScaler.

### 7. Training (see [06_training.md](06_training.md))

`run_epoch(model, loader, criterion, device, optimizer=...)` is one function for both train and eval. The loop saves the best checkpoint by val accuracy and tracks `EarlyStopping(patience=5)`. Per-epoch metrics flow to MLflow with `step=epoch+1` so you get curves.

### 8. Evaluation (see [08_evaluation.md](08_evaluation.md))

Reload the *best* checkpoint, not the last. `evaluate_test_set` returns probabilities, predictions, classification_report, confusion_matrix, AUROC. Everything goes into `test_report.json` with the MLflow `run_id` embedded for traceability.

### 9. Quality gate (see [09_validation_gate.md](09_validation_gate.md))

`run_validation_gate(best_val, auc, cfg)` returns a single bool. `gate_passed` is logged as both an MLflow metric and a tag. A failed gate doesn't fail the run — it gates the *registry*, not the pipeline.

### 10. Export & registry (see [09_validation_gate.md](09_validation_gate.md))

`export_torchscript()` produces `model.torchscript` regardless of gate. `mlflow.pytorch.log_model(..., registered_model_name=cfg.experiment if gate_passed else None)` is the gating step: artifacts are always logged, but only gated runs get a Model Registry entry.

## Where each artifact ends up consumed

| Artifact | Consumed by |
|----------|-------------|
| `best_model.pt` | `serve.py`, `patient_app.py`, batch scripts |
| `model.torchscript` | non-Python deployments, edge devices |
| `test_report.json` | CI smoke test (`acc < 0.30 → fail`), human review |
| `history.csv` | dashboards, regression checks between runs |
| `data_stats.json` | comparing two runs to see if the dataset moved |
| MLflow run | the UI; `mlflow.pytorch.load_model("models:/<exp>/Production")` for inference services |

## Failure modes and what they mean

- **`No valid images found ...`** — the data wasn't downloaded or the layout doesn't match `TASK_META["layout"]`. Check `cfg.data` and re-run.
- **`Too many corrupt images`** — usually a partial Kaggle download. Delete the data dir and re-run.
- **Smallest class has only N valid images** — the dataset is too small or you have an unexpected class folder (e.g. a `.git` directory under `data/`).
- **`gate_passed=False`** — the run completed but the model is below threshold. Don't promote; inspect the confusion matrix and Grad-CAMs.
- **MLflow UI shows no runs** — pointing at the wrong DB. `--mlflow_uri sqlite:///mlflow.db` and `mlflow ui --backend-store-uri sqlite:///mlflow.db` must match.

## Adapting to a new dataset

This pipeline is task-agnostic. For a new dataset you only edit `TASK_META` (and matching dicts in `serve.py` / `patient_app.py`). The full step-by-step is in [14_new_dataset_checklist.md](14_new_dataset_checklist.md).
