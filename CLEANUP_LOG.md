# Cleanup log (2026-09-22)

What changed in the repo tidy and code simplification, why, and how each change was checked. Nothing is committed yet. Review with `git status` / `git diff`, then commit or revert.

Input: "Weaknesspoint in the project.docx". The README rewrite, `.gitignore` additions and stackdump removal it describes were already in the working tree before this session. This session handled the items it left open (`mlops_pipeline.py` main(), the `brain_tumor_mlops.py` stub, untracked `documentation/`, `README.md.bak`), plus a simplification pass over all Python files.

## Result

| | Before | After |
|---|---|---|
| Python files | 12 | 7 |
| Python lines | 3,521 | 1,905 |
| Scripts that crashed on start | 4 | 0 |
| Copies of "load checkpoint + eval transform" | 6 | 1 (`model.load_checkpoint`) |

## Step-by-step trajectory

### 0. Baseline (before touching code)
- Copied every original `.py` file to a scratch folder.
- Built synthetic datasets in both layouts: brain (`Training/`, `Testing/`, 4 classes) and breast (flat, 3 classes, plus one `*_mask.png` and one corrupt file to exercise the filters).
- Ran the **original** `mlops_pipeline.py` on both tasks (`--epochs 2 --img_size 64 --batch 8`, seed 42) and kept every output as the reference.
- Confirmed four scripts were already broken:
  - `explain_brain_tumor.py`: `ModuleNotFoundError: train_brain_tumor`
  - `explain_breast_cancer.py`: `ModuleNotFoundError: train_breast_cancer`
  - `predict_app.py`: `ModuleNotFoundError: train_brain_tumor`
  - `brain_tumor_mlops.py`: `SyntaxError` (unfinished stub)

### 1. `model.py`: one home for inference helpers
- Added `eval_transform()`, `load_checkpoint()` and the `IMAGENET_MEAN/STD` constants.
- `build_model` is unchanged.
- Before this, `serve.py`, `patient_app.py`, `dashboard.py`, both explain scripts and the pipeline each carried their own copy of that code.

### 2. `mlops_pipeline.py`: split the 240-line `main()`
- Your uncommitted per-task gate thresholds (`TASK_META[...]["min_val_acc"]`, `--min_val_acc` defaulting to `None`) are kept as they were.
- `main()` now reads as the 10 stages. The training loop moved to `train_model()`. The three near-identical DataLoader blocks became `make_loader()`.
- All function names quoted in `documentation/` are kept, with the same signatures: `index_images(cfg)`, `build_splits(df, cfg)`, `EarlyStopping`, `run_epoch`, `evaluate_test_set`, `run_validation_gate`, `export_torchscript`.
- Removed:
  - unused imports (`sys`, `resnet50`, `ResNet50_Weights`)
  - the empty "Stage 6" section
  - the `triggered` flag in `EarlyStopping`
  - the duplicated forward pass in `run_epoch`
- **Behaviour change 1:** the script no longer runs `pip install` for `kagglehub` or `mlflow` at runtime. Both are in `requirements.txt` and the CI install step. Auto-installing inside a training script hides missing dependencies.
- **Behaviour change 2:** the pipeline now writes `test_images.csv` (the held-out split) for both tasks, so `explain.py` explains test images, not training images.
- The parameter log records one count for both `total_params` and `trainable_params`, because nothing in the model is frozen and the two were always equal.

### 3. `explain.py`: replaces both `explain_*.py` scripts
- The two scripts were about 95% identical (the diff was mostly spacing and default paths). Now there is one `explain.py --task brain_tumor|breast_cancer`, with defaults taken from `TASK_META`.
- This fixes the broken import: it uses `model.load_checkpoint`.
- The default checkpoint is now `artifacts_brain/`, not the old `artifacts/`.
- Simplified:
  - Integrated Gradients is built with one `linspace` rather than a Python list.
  - Removed the unused `denormalize()` and the unused return values.
  - A shared `normalise()` helper replaces four copies of the same code.

### 4. Apps and server
- `serve.py`, `patient_app.py` and `dashboard.py` now call `load_checkpoint`.
- Removed unused imports (`io`, `transforms`) and imports inside functions (`json` and `datetime` inside the button handler).
- `batch_predict.py` no longer mutates a `global SERVER`; the URL is passed as an argument.

### 5. Deleted files
Tracked files remain in git history (`git show HEAD:<file>`).

| File | Reason |
|---|---|
| `predict_app.py` | Crashed on import; `patient_app.py` replaced it |
| `train_brain_tumor_v2.py`, `train_breast_cancer_v2.py` | Line-for-line subsets of `mlops_pipeline.py` (same model, splits, loss and loop, minus MLflow and gate) |
| `explain_brain_tumor.py`, `explain_breast_cancer.py` | Merged into `explain.py` |
| `brain_tumor_mlops.py` (untracked) | Stub with `# ... existing argument parsing ...`; not valid Python |
| `README.md.bak` (untracked) | Byte-identical to `HEAD:README.md` |

### 6. Docs
- `README.md`: updated the explain commands and file layout. Removed two "Known issues" that are now fixed (`main()` too long; `_v2` suffixes).
- `CLAUDE.md`: rewritten. It described a single `train_brain_tumor.py` project that no longer exists.
- `documentation/`:
  - 08 and 14 now point to `explain.py`.
  - The 02 snippet no longer shows the auto-install.
  - The 06 snippet matches the simplified `EarlyStopping`.
- `DEPLOYMENT_GUIDE.md`: the two training commands now use `mlops_pipeline.py`.
- `MLOPS_GUIDE.md`: added a "historical note" at the top. It is a learning roadmap written around the removed scripts, so its content was left as is.

## Verification

| Check | Result |
|---|---|
| `py_compile` on all `.py` files | pass |
| New pipeline vs original, synthetic data, both tasks, same seed | **bit-identical**: `config.json`, `data_stats.json`, `history.csv` (excluding timing), `test_report.json` (excluding run_id) and all checkpoint weights |
| Gate-pass path (`--min_val_acc 0 --min_auroc 0`) | model registered in MLflow registry |
| Early stopping (`--patience 1 --epochs 6`) | stopped at epoch 3 |
| `EarlyStopping` new vs old, 2000 random val-acc sequences | 0 mismatches |
| `explain.py` vs original explain scripts (via an import shim), same checkpoint and image, both tasks | same prediction; figures match at 99.95% of pixels, the rest within 3/255 (float rounding) |
| `explain.py` batch mode | reads `test_images.csv`; fallback to indexing `data/Testing` works on the real brain checkpoint (meningioma, 95.8%, correct) |
| `serve.py` with real checkpoints | `/health` ok; unknown task → 404; non-image → 422; brain and breast predictions → 200 |
| `batch_predict.py` against the live server | 400/400 `data/Testing/pituitary` images classified, 100% predicted pituitary, avg confidence 95.8%, CSV written |
| `patient_app.py`, `dashboard.py` (Streamlit `AppTest`, both tasks, real artifacts) | no exceptions; dashboard metrics render, gate PASS for both |

Not verified: the Docker build (no Docker run here), the GitHub Actions run, the Kaggle download path (`kagglehub` isn't installed locally; the code there only lost the auto-install), and a real full-length GPU training run.

## Findings to decide on (not changed)

1. **README results don't match the artifacts on disk.** The README reports breast test accuracy 82.9%, AUROC 0.912 and malignant recall 0.677. `artifacts_breast/test_report.json` (26 Apr) shows 84.6%, 0.905 and **0.806**. One of them is from an older run. Pick the canonical run and make the README match it.
2. **The Kaggle key as a `workflow_dispatch` input** (your uncommitted `mlops.yml` change): inputs are not masked like secrets and appear in the run's UI and event payload. Prefer repository secrets only.
3. **Grad-CAM on the real brain model highlights the skull edge, not the tumour**, while occlusion does find the lesion. This is the same with the old code, so it is a model finding, not a refactor bug. It is worth a line in "Known issues", since reviewers ask about it.
4. Streamlit warns that `use_container_width` is deprecated in both apps. Harmless for now, but it will break on a future Streamlit version.
5. `MLOPS_GUIDE.md`, `DEPLOYMENT_GUIDE.md` and `AZURE_GUIDE.md` overlap heavily with `documentation/`. Consider folding them in or deleting them.
6. `documentation/` is still untracked. It's worth committing, since it's the best explanation of the project.
