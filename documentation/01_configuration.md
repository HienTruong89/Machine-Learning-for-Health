# Stage 1 — Configuration & reproducibility

**Goal:** lock every knob the pipeline cares about into a single, serializable object, and seed every RNG before any work happens.

## Why a dataclass beats a flat argparse

`argparse.Namespace` is fine until you need to (a) save the exact config alongside the model, (b) pass it through a dozen functions, (c) load it back for inference. A `@dataclass` gives you all three for free.

```python
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Config:
    task: str            # e.g. "brain_tumor"
    data: str            # raw data root
    out: str             # artifacts dir
    epochs: int
    batch: int
    lr: float
    img_size: int
    patience: int        # early-stopping patience
    min_val_acc: float   # quality gate threshold
    min_auroc: float     # quality gate threshold
    seed: int
    mlflow_uri: str
    experiment: str

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))
```

## Per-task metadata

Hard-code the dataset-specific bits in one dict so the rest of the pipeline stays generic:

```python
TASK_META = {
    "brain_tumor": {
        "kaggle_dataset": "masoudnickparvar/brain-tumor-mri-dataset",
        "default_data":   "data",
        "default_out":    "artifacts_brain",
        "layout":         "split",      # has Training/ and Testing/ subfolders
        "image_exts":     {".jpg", ".jpeg", ".png"},
        "mask_filter":    False,        # whether to skip *_mask.* files
        "experiment":     "brain-tumor-classification",
        "min_val_acc":    0.95,
        "min_auroc":      0.90,
    },
    # add more tasks here
}
```

For the next project, this is the *only* dict you need to extend.

## Reproducible seeding

Seed Python, NumPy, and PyTorch (CPU + CUDA) and disable cuDNN's nondeterministic kernels. This is enough to make a single-GPU run bit-reproducible.

```python
import random, numpy as np, torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
```

`cudnn.benchmark = False` costs a small amount of throughput but removes a real source of run-to-run variance — worth it for a clinical pipeline where you'll be asked "did the same data give the same model?"

## Save the config next to the artifacts

Always do this *before* training starts. If training crashes, you still know what was attempted.

```python
out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
cfg.save(out / "config.json")
```

The trained checkpoint embeds `classes`, `img_size`, `mean`, `std` (Stage 6), so `config.json` only needs to capture training-time choices, not inference-time ones.
