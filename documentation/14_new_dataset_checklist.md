# New-dataset checklist

The pipeline is designed so that adding a new dataset / task is a *config change*, not a rewrite. Walk this list top-to-bottom and you'll be deployed.

The running example is a hypothetical `chest_xray` task (3-class: normal / pneumonia / covid).

## 1. Pick a dataset

- Find a Kaggle dataset with a stable slug. Note: `<owner>/<dataset-name>` — e.g. `paultimothymooney/chest-xray-pneumonia`.
- Confirm the layout: does it have `train/` and `test/` predefined, or one flat folder of class dirs?
- Note the file extensions (`.jpg`, `.png`, `.dcm` — DICOM needs a different reader).
- Check there are at least 100 images in the smallest class. Fewer is research territory.

## 2. Add a `TASK_META` entry

In `mlops_pipeline.py`:

```python
TASK_META = {
    # ... existing tasks ...
    "chest_xray": {
        "kaggle_dataset":  "paultimothymooney/chest-xray-pneumonia",
        "default_data":    "data_xray",
        "default_out":     "artifacts_xray",
        "layout":          "split",            # has train/ and test/ subdirs
        "image_exts":      {".jpg", ".jpeg", ".png"},
        "mask_filter":     False,              # no mask files to skip
        "experiment":      "chest-xray-classification",
        "check_file_glob": "*.jpeg",
        "check_subdir":    "train",
        "min_val_acc":     0.85,               # set realistic for your dataset
        "min_auroc":       0.85,
    },
}
```

Adjust `default_data` / `default_out` so you don't collide with existing `data/` and `artifacts_*` dirs.

## 3. Run a smoke train

Use `--epochs 1` and zeroed gates to verify the data flow works end-to-end before committing to a real run:

```bash
python mlops_pipeline.py \
  --task chest_xray \
  --epochs 1 --batch 16 \
  --min_val_acc 0.0 --min_auroc 0.0
```

What to check in the output:
- `Stage 3 · Data validation` reports plausible per-class counts.
- `Split sizes — train / val / test` are non-zero for every class.
- One epoch finishes in seconds (CPU) or a few minutes (small GPU).
- `artifacts_xray/test_report.json` exists and isn't empty.

## 4. Run the real train

```bash
python mlops_pipeline.py --task chest_xray
```

Defaults: 20 epochs, batch 32, lr 3e-4, patience 5, gates from `TASK_META`. On a single GPU expect ~30-60 minutes for ~5k images.

Watch the early-stopping behaviour — if it fires at epoch 6 every time, your patience is too low *or* the LR is too high.

## 5. Inspect the results

Sanity-check the trained model before declaring victory:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → open http://127.0.0.1:5000, click the latest run
```

Look for:
- **Confusion matrix** in `test_report.json` — are the off-diagonal cells small?
- **Per-class recall** — every class should be > 70%. A low number for the rare class means class weighting wasn't enough; try oversampling.
- **train_acc vs. val_acc curve** — if train >> val by epoch 5, you're overfitting (raise dropout, more augmentation, more data).

## 6. Run explainability

```bash
# After training, check that the model is looking at the right region
python explain.py --task xray --n 8
```

(`explain.py` takes its defaults from `TASK_META`, so the new task entry is all it needs.)

If Grad-CAM is highlighting the corners of the image rather than the lung field, you have a data-leakage problem (often a scanner watermark). Don't deploy.

## 7. Update the patient-facing app

In `patient_app.py`:

```python
TASKS = {
    # ... existing ...
    "Chest X-Ray": {
        "artifacts":       Path("artifacts_xray"),
        "concern_classes": {"pneumonia", "covid"},
        "safe_class":      "normal",
        "description":     "Classifies chest X-rays: Normal, Pneumonia, COVID",
        "upload_label":    "Upload chest X-ray (JPG or PNG)",
        "color":           "#2BA02B",
    },
}

FRIENDLY_NAMES.update({
    "normal":    "Normal",
    "pneumonia": "Pneumonia Detected",
    "covid":     "COVID Pattern Detected",
})

RISK_FLAGS.update({
    "normal":    ("NO FINDING",       "#28a745", "result-negative"),
    "pneumonia": ("HIGH CONCERN",     "#dc3545", "result-positive"),
    "covid":     ("HIGH CONCERN",     "#dc3545", "result-positive"),
})
```

Test locally:

```bash
streamlit run patient_app.py
# pick "Chest X-Ray", upload a sample, verify result + breakdown
```

## 8. Update the FastAPI server

In `serve.py`:

```python
TASK_CONFIG = {
    # ...
    "chest_xray": {
        "checkpoint":  "artifacts_xray/best_model.pt",
        "description": "3-class chest X-ray: normal / pneumonia / covid",
    },
}
```

Test locally:

```bash
uvicorn serve:app --reload --port 8000
curl http://localhost:8000/health
curl -F "file=@xray.jpg" http://localhost:8000/predict/chest_xray
```

## 9. Update Docker + CI

`Dockerfile` — add a `COPY` line:

```dockerfile
COPY artifacts_xray/best_model.pt artifacts_xray/best_model.pt
```

`.github/workflows/mlops.yml` — duplicate the smoke-train step for the new task and add an upload-artifact step pointing at `artifacts_xray/`.

## 10. Publish the weights

```bash
huggingface-cli upload <user>/medical-imaging-models \
    artifacts_xray/best_model.pt artifacts_xray/best_model.pt
```

## 11. Ship it

```bash
git add mlops_pipeline.py patient_app.py serve.py Dockerfile .github/workflows/mlops.yml
git commit -m "Add chest_xray task"
git push
```

If you're running on Azure (per [13_cli_tools.md](13_cli_tools.md)):

```bash
docker build -t medical-imaging-api:v$(date +%Y%m%d) .
docker tag  medical-imaging-api:v$(date +%Y%m%d) $ACR.azurecr.io/medical-imaging-api:v$(date +%Y%m%d)
docker push                                       $ACR.azurecr.io/medical-imaging-api:v$(date +%Y%m%d)
az webapp config container set --resource-group $RG --name $APP \
    --container-image-name $ACR.azurecr.io/medical-imaging-api:v$(date +%Y%m%d)
```

## What you didn't have to touch

- `model.py` — same architecture works for 2, 3, 4, or 10 classes.
- The training loop, AMP, optimizer, scheduler — all generic.
- MLflow tracking, validation gate, TorchScript export — all generic.
- The Streamlit caching, the FastAPI request handling — all generic.

That's the whole point of this template: dataset onboarding is a `TASK_META` entry plus three small dict updates.
