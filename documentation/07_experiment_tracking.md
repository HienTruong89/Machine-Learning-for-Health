# Stage 7 — Experiment tracking with MLflow

**Goal:** record every run's parameters, metrics, and artifacts so you can compare experiments and reproduce results without grep-ing through stdout.

## SQLite-backed local tracking

Don't run a tracking server in development. The `sqlite:///mlflow.db` URI gives you the full MLflow UI with zero infrastructure.

```python
import mlflow
import mlflow.pytorch

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(cfg.experiment)        # e.g. "brain-tumor-classification"
```

To browse runs locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://127.0.0.1:5000
```

For a team, swap the URI for a remote tracking server (e.g. `http://mlflow.internal:5000`) or Databricks. Nothing else changes.

## A run is a `with` block

Wrap the entire training + evaluation block in `mlflow.start_run`. Anything raised inside is automatically marked FAILED.

```python
import time

run_name = f"{cfg.task}_{time.strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.set_tags({
        "task":                cfg.task,
        "device":              device.type,
        "dataset_fingerprint": data_stats["dataset_fingerprint"],
    })
    mlflow.log_params({
        "epochs":      cfg.epochs,
        "batch":       cfg.batch,
        "lr":          cfg.lr,
        "img_size":    cfg.img_size,
        "patience":    cfg.patience,
        "seed":        cfg.seed,
        "num_classes": len(classes),
        "train_size":  len(train_df),
        "val_size":    len(val_df),
        "test_size":   len(test_df),
        "architecture": "ResNet50-IMAGENET_V2",
    })
    mlflow.log_artifact(str(out / "config.json"))
    mlflow.log_artifact(str(out / "data_stats.json"))
    # ... training loop here ...
```

## Tags vs. params vs. metrics

This is the distinction that pays off when comparing 50 runs in the UI:

| Use | For |
|-----|-----|
| **Tags** | Categorical / discrete identifiers. Things you'd *filter* by: `task`, `device`, `dataset_fingerprint`, `gate_passed`. |
| **Params** | Knobs you set before the run. Hyperparameters, dataset sizes, architecture name. |
| **Metrics** | Numbers that change over time or come out of evaluation. `train_loss`, `val_acc`, `test_macro_auroc`. |

Tags and params are write-once. Metrics support a `step=` argument so you get per-epoch curves in the UI:

```python
mlflow.log_metrics({
    "train_loss": tr_loss, "val_loss": vl_loss,
    "train_acc":  tr_acc,  "val_acc":  vl_acc,
    "lr":         scheduler.get_last_lr()[0],
}, step=epoch + 1)
```

## Final metrics, gate flag, and the model

After training:

```python
mlflow.log_metric("best_val_acc",     best_val)
mlflow.log_metric("epochs_trained",   len(history))
mlflow.log_metric("test_macro_auroc", auc)
mlflow.log_metric("test_accuracy",    report["accuracy"])
mlflow.log_metric("gate_passed",      int(gate_passed))
mlflow.set_tag("gate_passed", str(gate_passed))

mlflow.log_artifact(str(out / "history.csv"))
mlflow.log_artifact(str(out / "test_report.json"))
```

Then log the model itself, with conditional registry — see Stage 9.

## What's worth logging, what isn't

- **Worth it:** `data_stats.json`, `config.json`, `history.csv`, `test_report.json`, the TorchScript export, the best `.pt` checkpoint via `mlflow.pytorch.log_model`.
- **Skip:** the raw dataset (use the fingerprint), per-batch metrics (too noisy), debug images (clutter — explainer outputs go in their own artifacts dir).

## Reading runs back programmatically

When you need to compare runs in a notebook:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
runs   = client.search_runs(
    experiment_ids=[client.get_experiment_by_name(cfg.experiment).experiment_id],
    order_by=["metrics.val_acc DESC"],
    max_results=10,
)
for r in runs:
    print(r.info.run_name, r.data.metrics["val_acc"], r.data.tags["dataset_fingerprint"])
```

## Why not Weights & Biases?

W&B has a slicker UI, but it's a hosted service that wants an API key per machine. MLflow is local-first, OSS, and the API is identical whether you point at SQLite, Postgres, or a remote tracking server. Pick W&B when collaboration / dashboards become the bottleneck — until then MLflow is enough.
