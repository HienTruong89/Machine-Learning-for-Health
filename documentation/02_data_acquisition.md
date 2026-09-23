# Stage 2 — Data acquisition

**Goal:** pull raw imaging data into a known folder layout, idempotently, on a fresh machine or in CI, without leaking credentials.

## Use `kagglehub`, not the Kaggle CLI

The Python `kaggle` package and the `kaggle` CLI both authenticate at import / process start. `kagglehub` is the modern replacement: it caches downloads, deduplicates across runs, and works on Windows / Linux / macOS without `chmod` rituals or PATH gymnastics.

## Credential resolution order

Three sources, checked in order, so the same code works locally, in CI, and inside notebooks:

1. CLI flags `--kaggle_user` / `--kaggle_key`
2. Env vars `KAGGLE_USERNAME` / `KAGGLE_KEY` (used by GitHub Actions secrets)
3. `~/.kaggle/kaggle.json`

```python
import json, os
from pathlib import Path

def ensure_kaggle_credentials(cfg) -> None:
    user = cfg.kaggle_user or os.getenv("KAGGLE_USERNAME", "").strip()
    key  = cfg.kaggle_key  or os.getenv("KAGGLE_KEY",      "").strip()
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if user and key:
        cred.parent.mkdir(exist_ok=True)
        cred.write_text(json.dumps({"username": user, "key": key}))
        try:
            cred.chmod(0o600)        # POSIX only — wrap in try/except for Windows
        except Exception:
            pass
    elif not cred.exists():
        raise RuntimeError("No Kaggle credentials found.")
```

`chmod(0o600)` is wrapped because Windows raises on POSIX permissions. Don't print the credential file path in production logs — leave that to local debugging.

## Idempotent download

Always check whether the dataset is already present before hitting the network. CI runners and local re-runs both depend on this.

```python
import shutil, sys
from pathlib import Path

def download_dataset(cfg) -> None:
    meta     = TASK_META[cfg.task]
    data_dir = Path(cfg.data)
    check    = data_dir / meta["check_subdir"]   # e.g. "Training" or "benign"

    if check.exists() and any(check.rglob(meta["check_file_glob"])):
        return  # already downloaded

    ensure_kaggle_credentials(cfg)

    import kagglehub  # imported here: it authenticates on import, so credentials must exist first

    src = Path(kagglehub.dataset_download(meta["kaggle_dataset"]))
    data_dir.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest = data_dir / item.name
        if not dest.exists():
            (shutil.copytree if item.is_dir() else shutil.copy2)(str(item), str(dest))
```

## Flatten nested archives

Kaggle datasets often unzip into a single nested folder (`brain-tumor-mri-dataset/Training/...`). Flatten once during acquisition so every downstream stage can assume a clean root.

```python
for nested_name in ["brain-tumor-mri-dataset", "Dataset_BUSI_with_GT"]:
    nested = data_dir / nested_name
    if nested.exists():
        for item in nested.iterdir():
            shutil.move(str(item), str(data_dir / item.name))
        nested.rmdir()
```

## Two layouts to support

Most public medical-imaging datasets fall into one of two shapes — encode this in `TASK_META["layout"]` so the indexing function (Stage 3) knows where to look:

```
# "split" layout (brain tumor MRI)
data/Training/<class>/*.jpg
data/Testing/<class>/*.jpg

# "flat" layout (breast cancer ultrasound)
data_breast/<class>/*.png
```

If you hit a third shape, normalize it during acquisition rather than special-casing every later stage.

## What about DVC?

The repo has a `data.dvc` pointer file. DVC is useful when (a) the dataset is too big for git-LFS, or (b) you need to version a *specific* snapshot of the data instead of "whatever Kaggle returns today." For most starter projects, the `dataset_fingerprint` in Stage 3 is enough — DVC is the next upgrade if you need to reproduce a result months later.
