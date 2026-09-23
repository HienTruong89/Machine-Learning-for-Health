# Stage 3 — Data validation

**Goal:** detect corrupt or missing data *before* training burns GPU hours, and produce a fingerprint so you can prove which exact files trained which model.

## Index every image with PIL.verify

`PIL.Image.verify()` only parses the header, so it's fast — but it catches truncated files, the most common source of silent training failures.

```python
import pandas as pd
from pathlib import Path
from PIL import Image

def index_images(cfg) -> pd.DataFrame:
    meta     = TASK_META[cfg.task]
    data_dir = Path(cfg.data)
    exts     = meta["image_exts"]
    rows     = []

    if meta["layout"] == "split":
        scan_roots = [(data_dir / s, s) for s in ["Training", "Testing"]]
    else:
        scan_roots = [(data_dir, "all")]

    for root, split in scan_roots:
        if not root.exists():
            continue
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir():
                continue
            for p in cls_dir.glob("*"):
                if p.suffix.lower() not in exts:
                    continue
                if meta["mask_filter"] and "mask" in p.name.lower():
                    continue
                try:
                    with Image.open(p) as im:
                        im.verify()
                    rows.append({"path": str(p), "split": split,
                                 "label": cls_dir.name, "ok": True, "err": None})
                except Exception as exc:
                    rows.append({"path": str(p), "split": split,
                                 "label": cls_dir.name, "ok": False, "err": str(exc)})

    cols = ["path", "split", "label", "ok", "err"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
```

The DataFrame becomes the single source of truth for which files exist. Filter on `ok` later instead of crashing inside `__getitem__`.

## Hard fail vs. warn

Two thresholds worth checking:

- **Corruption rate > 5%** — fail loudly. Something is wrong with the download, not the data.
- **Smallest class < 10 images** — fail. Stratified split with k-fold CV would silently break, and the model will memorize.

```python
import hashlib

def validate_data(df: pd.DataFrame) -> dict:
    total       = len(df)
    corrupt     = int((~df["ok"]).sum())
    corrupt_pct = corrupt / total if total else 1.0
    if corrupt_pct > 0.05:
        raise ValueError(f"Too many corrupt images: {corrupt}/{total} ({corrupt_pct:.1%})")

    df_ok      = df[df["ok"]]
    per_class  = df_ok.groupby("label").size().to_dict()
    min_count  = min(per_class.values()) if per_class else 0
    if min_count < 10:
        raise ValueError(f"Smallest class has only {min_count} valid images.")

    fingerprint = hashlib.md5(
        "\n".join(sorted(df_ok["path"].tolist())).encode()
    ).hexdigest()

    return {
        "total_images":        total,
        "corrupt_images":      corrupt,
        "corrupt_pct":         round(corrupt_pct, 4),
        "per_class_counts":    {k: int(v) for k, v in per_class.items()},
        "min_class_samples":   int(min_count),
        "dataset_fingerprint": fingerprint,
    }
```

## The dataset fingerprint

A stable hash of the sorted file paths is enough to tell two runs apart "did the data change?" without storing the data twice. Save it next to the artifacts and log it as an MLflow tag (Stage 7). When a run later disagrees with another, you can compare fingerprints first before suspecting hyperparameters.

```python
out = Path(cfg.out)
(out / "data_stats.json").write_text(json.dumps(stats, indent=2))
```

For more rigour, hash file *contents* not paths — but path-hash is fast (~1s for 10k files) and catches the realistic failure mode (a file got added, removed, or moved).

## What this *doesn't* catch

- Label noise (a glioma scan filed under `meningioma/`)
- Distribution shift between Training/ and Testing/
- Duplicate images across splits (a real risk on Kaggle datasets)

For the next project, consider adding a perceptual-hash dedupe pass and a between-split overlap check before declaring the validation stage complete.
