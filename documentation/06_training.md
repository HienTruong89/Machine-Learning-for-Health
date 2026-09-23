# Stage 6 — Training loop

**Goal:** train fast and stably, with class-imbalance protection, mixed precision when CUDA is available, and an early-stopping rule that doesn't waste epochs.

## One epoch function for train and eval

A single function with `optimizer is not None` switching modes is shorter, less error-prone, and guarantees both passes see the same data flow.

```python
import torch
from tqdm import tqdm

def run_epoch(model, loader, criterion, device,
              optimizer=None, scaler=None, desc=""):
    is_train = optimizer is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False, desc=desc):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type,
                                         enabled=scaler is not None):
                    out  = model(x)
                    loss = criterion(out, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                out  = model(x)
                loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            correct  += (out.argmax(1) == y).sum().item()
            total    += x.size(0)
    return loss_sum / total, correct / total
```

Notes:
- `non_blocking=True` is only useful with `pin_memory=True` (Stage 4). Together they overlap host→device transfers with compute.
- `torch.amp.autocast(device_type=device.type)` is the new (PyTorch ≥2.4) API replacing `torch.cuda.amp.autocast()` — works on CPU too (no-op there).

## Class-imbalance: weights + label smoothing

Medical datasets are almost always imbalanced. The cheap fix is two lines: inverse-frequency class weights in the loss, plus a small `label_smoothing` to prevent the network from getting overconfident on the majority class.

```python
import torch.nn as nn

counts = train_df["label"].value_counts().reindex(classes).values.astype(float)
class_weights = torch.tensor(
    counts.sum() / (len(classes) * counts),     # inverse frequency, normalized
    dtype=torch.float32,
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05,
)
```

`label_smoothing=0.05` is a safe default — bigger values (0.1+) start to hurt confident classes.

## Optimizer + schedule

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
```

- **AdamW** (decoupled weight decay) beats Adam on transfer-learning fine-tunes — use it by default.
- **`lr=3e-4`** ("Karpathy's constant") is a fine starting point with AdamW for fine-tuning.
- **CosineAnnealingLR** smoothly decays to ~0 by `T_max=epochs`. No tuning required.
- **`weight_decay=1e-4`** is enough for ResNet50 with dropout already in the head.

## Early stopping

A trivial state machine — breaks the training loop when `patience` consecutive epochs show no improvement.

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None

    def __call__(self, val_acc):
        """Record this epoch's val_acc; return True when training should stop."""
        if self.best is None or val_acc > self.best + self.min_delta:
            self.best    = val_acc
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
```

## The full loop

This is the heart of the pipeline. Every artifact downstream — checkpoints, history, MLflow metrics — is produced here.

```python
import pandas as pd
import time

early_stop = EarlyStopping(patience=cfg.patience)
best_val   = 0.0
history    = []
ckpt_path  = out / "best_model.pt"

for epoch in range(cfg.epochs):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                 optimizer=optimizer, scaler=scaler,
                                 desc=f"train e{epoch+1}")
    vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device,
                                 desc=f"val   e{epoch+1}")
    scheduler.step()

    history.append({
        "epoch":      epoch + 1,
        "train_loss": round(tr_loss, 4), "val_loss": round(vl_loss, 4),
        "train_acc":  round(tr_acc,  4), "val_acc":  round(vl_acc,  4),
        "elapsed_s":  round(time.time() - t0, 1),
    })

    if vl_acc > best_val:
        best_val = vl_acc
        torch.save({
            "state_dict": model.state_dict(),
            "classes":    classes,
            "img_size":   cfg.img_size,
            "mean":       IMAGENET_MEAN,
            "std":        IMAGENET_STD,
            "epoch":      epoch + 1,
            "val_acc":    vl_acc,
        }, ckpt_path)

    if early_stop(vl_acc):
        break

pd.DataFrame(history).to_csv(out / "history.csv", index=False)
```

## What goes in the checkpoint

Bundle everything inference needs — never just `state_dict`. The Streamlit app and the FastAPI server both read these fields directly:

- `state_dict` — the weights
- `classes` — alphabetical class names, defines argmax → label mapping
- `img_size`, `mean`, `std` — required to rebuild the eval transform
- `epoch`, `val_acc` — for debugging / reporting

This is the contract that makes `serve.py` and `patient_app.py` portable across tasks without hardcoded values.
