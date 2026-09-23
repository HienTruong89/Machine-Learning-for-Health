# Stage 8 — Evaluation

**Goal:** load the *best* checkpoint (not the last one), evaluate on the held-out test set, and produce a structured report you can diff between runs.

## Reload the best, never the last

Validation accuracy improves and degrades over epochs. The last epoch is rarely the best — always reload the saved checkpoint:

```python
import torch

ckpt = torch.load(out / "best_model.pt",
                  map_location=device, weights_only=True)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

`weights_only=True` blocks arbitrary pickle execution — important even for your own files, because someday you'll consume one from elsewhere.

## Collect predictions, probabilities, and labels

Three parallel arrays are all you need to compute every metric afterward.

```python
import numpy as np
import torch
from tqdm import tqdm

def evaluate_test_set(model, loader, device, classes):
    model.eval()
    all_y, all_p, all_prob = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc="test"):
            x    = x.to(device)
            prob = torch.softmax(model(x), dim=1).cpu().numpy()
            all_prob.append(prob)
            all_p.append(prob.argmax(1))
            all_y.append(y.numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    y_prob = np.concatenate(all_prob)
    return y_true, y_pred, y_prob
```

Keep `y_prob` (the full probability matrix) — without it you can't compute AUROC or recalibrate thresholds later.

## Three metrics you actually need

For multi-class medical classification, report all three:

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

report = classification_report(
    y_true, y_pred, target_names=classes,
    digits=4, output_dict=True,
)
cm  = confusion_matrix(y_true, y_pred).tolist()
auc = roc_auc_score(y_true, y_prob, multi_class="ovr")  # macro one-vs-rest
```

- **classification_report** — per-class precision/recall/F1. Catches "model has 95% accuracy but 0% recall on the rare class" — common with imbalance.
- **confusion_matrix** — tells you *which* classes get confused. Glioma misclassified as meningioma is a different clinical risk than glioma misclassified as notumor.
- **AUROC (macro OvR)** — threshold-independent. Useful when you might rebalance precision/recall for clinical deployment.

## Save a structured report

JSON, not Markdown. You'll want to diff and chart these between runs.

```python
import json

summary = {
    "task":                  cfg.task,
    "best_val_acc":          best_val,
    "test_macro_auroc":      auc,
    "test_accuracy":         report["accuracy"],
    "classification_report": report,
    "confusion_matrix":      {"labels": classes, "matrix": cm},
    "classes":               classes,
    "run_id":                run.info.run_id,        # MLflow run for traceability
}
(out / "test_report.json").write_text(json.dumps(summary, indent=2))
```

Embedding the MLflow `run_id` in the JSON closes the loop: given any deployed model file, you can find its training run.

## What's deliberately *not* here

- **Bootstrap CIs on accuracy.** Worth adding for a paper, overkill for routine training. If you do, log them as separate metrics: `test_accuracy_ci_low`, `test_accuracy_ci_high`.
- **Calibration plot (reliability diagram).** If your downstream system uses confidence as an action threshold, add it. Otherwise the macro AUROC is enough.
- **Per-image errors dump.** Useful for debugging — write a separate script that consumes `best_model.pt` and writes a CSV of misclassifications. Don't bloat the main pipeline.

## Also useful: explainability sanity check

The repo has `explain.py` doing Grad-CAM, Grad-CAM++, Integrated Gradients, and Occlusion Sensitivity on the best checkpoint. Run it after every successful training:

```bash
python explain.py --task brain_tumor --n 8
```

This is your last line of defence against a model that gets the test metrics right but is looking at the wrong thing (acquisition artifacts, scanner watermarks). For the next project, integrate a few-image saliency check into the pipeline as Stage 8.5 — even just dumping 4 random Grad-CAMs to the artifacts dir builds intuition fast.
