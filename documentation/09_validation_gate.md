# Stage 9 — Quality gate, export, registry

**Goal:** refuse to register or deploy any model that's worse than your stated thresholds, and produce a portable artifact that doesn't depend on the training environment.

## A binary gate, not a recommendation

The gate has two thresholds: a minimum validation accuracy (sanity check) and a minimum test macro AUROC (held-out generalisation). Both are per-task and live in `TASK_META`.

```python
def run_validation_gate(best_val_acc: float,
                        test_auroc:   float,
                        cfg) -> bool:
    acc_ok   = best_val_acc >= cfg.min_val_acc
    auroc_ok = test_auroc   >= cfg.min_auroc
    return acc_ok and auroc_ok
```

Why two thresholds? Val accuracy alone can be cheated by class imbalance (predict majority class → "high" accuracy, awful AUROC). Test AUROC alone may pass even when the model overfit val — you want both.

Log the result so it's queryable in MLflow:

```python
gate_passed = run_validation_gate(best_val, auc, cfg)
mlflow.log_metric("gate_passed", int(gate_passed))
mlflow.set_tag(   "gate_passed", str(gate_passed))
```

## TorchScript: portable inference

`torch.jit.trace` produces a self-contained graph that runs without your Python source. Use it whenever the deploy target isn't your training environment (different Python version, no `model.py` file, etc.).

```python
def export_torchscript(model, img_size: int, out: Path) -> Path:
    model.eval().cpu()
    dummy    = torch.randn(1, 3, img_size, img_size)
    scripted = torch.jit.trace(model, dummy)
    path     = out / "model.torchscript"
    torch.jit.save(scripted, str(path))
    return path
```

`trace` only captures the path your dummy input takes through the network. For ResNet50 (no data-dependent control flow) it's safe. If you ever add `if x.sum() > 0:` style branches in the model, switch to `torch.jit.script`.

## Register only if the gate passes

The MLflow Model Registry is the contract between training and serving. Promotion is gated:

```python
import mlflow.pytorch

mlflow.pytorch.log_model(
    model,
    name="pytorch_model",
    registered_model_name=cfg.experiment if gate_passed else None,
    serialization_format="pickle",
)
```

`registered_model_name=None` means the model is logged as a run artifact only — visible in the UI, but not promoted into the registry where downstream services pick up "the latest production model."

This is the single most important pattern in the whole pipeline: **a failed gate is not a failed run.** You still want the artifacts, the metrics, and the explainability dumps for diagnosis. You just don't want anyone deploying it.

## Three deployable artifacts

After a passing run you have:

```
artifacts_<task>/
├── best_model.pt            # PyTorch state_dict + metadata, used by Streamlit and API
├── model.torchscript        # framework-free, used by edge deploys / non-Python serving
└── test_report.json         # the proof
```

Plus, in MLflow:
- The MLflow Models artifact (a directory containing the model + a `conda.yaml` manifest)
- A registered model version, ready for stage transitions (Staging → Production)

For deployment to Streamlit Cloud / Hugging Face Spaces, you don't need the registry — you just need `best_model.pt`. Upload it to a Hugging Face model repo (see Stage 11) and the app downloads it on first run.

## Setting realistic thresholds

For the next project, base thresholds on:

1. **A baseline model.** Train a simpler model (e.g. ResNet18 frozen backbone) first; that's the floor.
2. **The published state of the art on the dataset, minus 2-3%.** Beats vanity, catches regressions.
3. **The clinical decision threshold.** If a downstream system triages cases below 90% confidence to a human, your AUROC needs to be high enough that "below 90%" isn't every case.

Don't set thresholds to "whatever my best run got" — that locks you into one training run forever.
