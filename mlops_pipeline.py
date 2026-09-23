"""
MLOps pipeline for medical image classification (PyTorch).

Supports two tasks, selected with --task:
  brain_tumor   — MRI scan classification  (glioma / meningioma / notumor / pituitary)
  breast_cancer — Ultrasound classification (benign / malignant / normal)

Ten explicit MLOps stages:
  1.  Configuration management  — dataclass config + reproducible seed
  2.  Data acquisition          — Kaggle download via kagglehub
  3.  Data validation           — integrity checks, class-distribution audit,
                                  dataset fingerprint
  4.  Feature engineering       — augmentation pipeline, stratified splits
  5.  Experiment tracking       — MLflow experiment + run (params, tags)
  6.  Model building            — ResNet50 transfer-learning head (model.py)
  7.  Model training            — AMP, AdamW, CosineAnnealingLR, early stopping
  8.  Model evaluation          — classification report, confusion matrix, AUROC
  9.  Model validation gate     — reject model below accuracy / AUROC thresholds
  10. Model export & registry   — TorchScript + MLflow Model Registry

Usage:
    python mlops_pipeline.py --task brain_tumor
    python mlops_pipeline.py --task breast_cancer --epochs 25 --batch 32
    python mlops_pipeline.py --task brain_tumor --kaggle_user NAME --kaggle_key KEY

View runs:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import IMAGENET_MEAN, IMAGENET_STD, build_model, eval_transform

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 · Configuration management
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    task: str           # "brain_tumor" | "breast_cancer"
    data: str           # raw data root
    out: str            # artifacts directory
    kaggle_user: str
    kaggle_key: str
    epochs: int
    batch: int
    lr: float
    img_size: int
    patience: int       # early-stopping patience (epochs without val_acc improvement)
    min_val_acc: float  # deployment gate — minimum validation accuracy
    min_auroc: float    # deployment gate — minimum macro OvR AUROC
    seed: int
    mlflow_uri: str
    experiment: str

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


# Per-task static metadata (dataset IDs, folder layouts, file-type filters, gates)
TASK_META = {
    "brain_tumor": {
        "kaggle_dataset": "masoudnickparvar/brain-tumor-mri-dataset",
        "default_data":   "data",
        "default_out":    "artifacts_brain",
        "layout":         "split",          # Training/ and Testing/ sub-dirs
        "image_exts":     {".jpg", ".jpeg", ".png"},
        "mask_filter":    False,
        "experiment":     "brain-tumor-classification",
        "check_file_glob": "*.jpg",
        "check_subdir":   "Training",
        "min_val_acc":    0.95,             # 4-class MRI — high bar
        "min_auroc":      0.90,
    },
    "breast_cancer": {
        "kaggle_dataset": "aryashah2k/breast-ultrasound-images-dataset",
        "default_data":   "data_breast",
        "default_out":    "artifacts_breast",
        "layout":         "flat",           # <class>/ directly under data root
        "image_exts":     {".png", ".jpg", ".jpeg"},
        "mask_filter":    True,             # skip filenames containing "mask"
        "experiment":     "breast-cancer-classification",
        "check_file_glob": "*.png",
        "check_subdir":   "benign",
        "min_val_acc":    0.90,             # 3-class ultrasound — adjusted threshold
        "min_auroc":      0.90,
    },
}


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="MLOps pipeline — brain tumor MRI or breast cancer ultrasound."
    )
    ap.add_argument("--task",        required=True, choices=list(TASK_META),
                    help="Which dataset / task to run.")
    ap.add_argument("--data",        type=str,   default="",
                    help="Root data directory (task default if omitted).")
    ap.add_argument("--out",         type=str,   default="",
                    help="Artifacts output directory (task default if omitted).")
    ap.add_argument("--kaggle_user", type=str,   default="")
    ap.add_argument("--kaggle_key",  type=str,   default="")
    ap.add_argument("--epochs",      type=int,   default=20)
    ap.add_argument("--batch",       type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--img_size",    type=int,   default=224)
    ap.add_argument("--patience",    type=int,   default=5,
                    help="Early-stopping patience in epochs.")
    ap.add_argument("--min_val_acc", type=float, default=None,
                    help="Deployment gate: minimum validation accuracy (default: per-task).")
    ap.add_argument("--min_auroc",   type=float, default=None,
                    help="Deployment gate: minimum macro OvR AUROC (default: per-task).")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--mlflow_uri",  type=str,   default="sqlite:///mlflow.db",
                    help="MLflow tracking URI.")
    a = ap.parse_args()

    meta = TASK_META[a.task]
    return Config(
        task=a.task,
        data=a.data or meta["default_data"],
        out=a.out   or meta["default_out"],
        kaggle_user=a.kaggle_user,
        kaggle_key=a.kaggle_key,
        epochs=a.epochs,
        batch=a.batch,
        lr=a.lr,
        img_size=a.img_size,
        patience=a.patience,
        min_val_acc=a.min_val_acc if a.min_val_acc is not None else meta["min_val_acc"],
        min_auroc=a.min_auroc     if a.min_auroc   is not None else meta["min_auroc"],
        seed=a.seed,
        mlflow_uri=a.mlflow_uri,
        experiment=meta["experiment"],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 · Data acquisition
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_kaggle_credentials(cfg: Config) -> None:
    user = cfg.kaggle_user or os.getenv("KAGGLE_USERNAME", "").strip()
    key  = cfg.kaggle_key  or os.getenv("KAGGLE_KEY",      "").strip()
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if user and key:
        cred.parent.mkdir(exist_ok=True)
        cred.write_text(json.dumps({"username": user, "key": key}))
        try:
            cred.chmod(0o600)   # no-op on Windows
        except Exception:
            pass
        log.info("Kaggle credentials written to %s", cred)
    elif cred.exists():
        log.info("Using existing Kaggle credentials at %s", cred)
    else:
        raise RuntimeError(
            "Kaggle credentials not found.\n"
            "Get your API token at: https://www.kaggle.com/settings -> API\n"
            f"Place kaggle.json at {cred}\n"
            "Or pass --kaggle_user NAME --kaggle_key KEY"
        )


def download_dataset(cfg: Config) -> None:
    meta     = TASK_META[cfg.task]
    data_dir = Path(cfg.data)
    check    = data_dir / meta["check_subdir"]

    if check.exists() and any(check.rglob(meta["check_file_glob"])):
        log.info("Dataset already present at '%s' — skipping download.", data_dir)
        return

    _ensure_kaggle_credentials(cfg)
    import kagglehub  # imported here: it authenticates on import, so credentials must exist first

    log.info("Downloading '%s' via kagglehub ...", meta["kaggle_dataset"])
    src = Path(kagglehub.dataset_download(meta["kaggle_dataset"]))

    data_dir.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest = data_dir / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # Flatten known nested archive sub-dirs
    for nested_name in ["brain-tumor-mri-dataset", "Dataset_BUSI_with_GT"]:
        nested = data_dir / nested_name
        if not nested.exists():
            continue
        for item in nested.iterdir():
            dest = data_dir / item.name
            if dest.is_dir():
                shutil.rmtree(dest)
            elif dest.exists():
                dest.unlink()
            shutil.move(str(item), str(dest))
        nested.rmdir()

    log.info("Dataset ready at '%s'.", data_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 · Data validation
# ══════════════════════════════════════════════════════════════════════════════

def index_images(cfg: Config) -> pd.DataFrame:
    """One row per image file: path, split, label, ok (opens cleanly), err."""
    meta     = TASK_META[cfg.task]
    data_dir = Path(cfg.data)

    if meta["layout"] == "split":
        scan_roots = [(data_dir / s, s) for s in ["Training", "Testing"]]
    else:
        scan_roots = [(data_dir, "all")]

    rows = []
    for root, split in scan_roots:
        if not root.exists():
            continue
        for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for p in cls_dir.glob("*"):
                if p.suffix.lower() not in meta["image_exts"]:
                    continue
                if meta["mask_filter"] and "mask" in p.name.lower():
                    continue
                try:
                    with Image.open(p) as im:
                        im.verify()
                    ok, err = True, None
                except Exception as exc:
                    ok, err = False, str(exc)
                rows.append({"path": str(p), "split": split,
                             "label": cls_dir.name, "ok": ok, "err": err})

    return pd.DataFrame(rows, columns=["path", "split", "label", "ok", "err"])


def validate_data(df: pd.DataFrame) -> dict:
    """
    Run data-quality checks. Raises ValueError on critical failures.
    Returns a stats dict that is logged to MLflow and saved as an artifact.
    """
    total       = len(df)
    corrupt     = int((~df["ok"]).sum())
    corrupt_pct = corrupt / total if total else 1.0

    if corrupt_pct > 0.05:
        raise ValueError(
            f"Too many corrupt images: {corrupt}/{total} ({corrupt_pct:.1%}). "
            "Inspect the dataset before continuing."
        )

    df_ok     = df[df["ok"]]
    per_class = df_ok.groupby("label").size().to_dict()
    min_count = min(per_class.values()) if per_class else 0

    if min_count < 10:
        raise ValueError(
            f"Smallest class has only {min_count} valid images — too few to train reliably."
        )

    # Fingerprint: stable hash of sorted file paths for reproducibility tracking
    fingerprint = hashlib.md5(
        "\n".join(sorted(df_ok["path"].tolist())).encode()
    ).hexdigest()

    stats = {
        "total_images":      int(total),
        "corrupt_images":    corrupt,
        "corrupt_pct":       round(corrupt_pct, 4),
        "per_class_counts":  {k: int(v) for k, v in per_class.items()},
        "min_class_samples": int(min_count),
        "dataset_fingerprint": fingerprint,
    }
    log.info("Data validation passed | classes=%s | fingerprint=%s",
             per_class, fingerprint)
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 · Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

def build_transforms(img_size: int):
    """Return (train_tfm, eval_tfm). Augmentation is applied to training only."""
    train_tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])
    return train_tfm, eval_transform(img_size)


def build_splits(df: pd.DataFrame, cfg: Config):
    """Return (train_df, val_df, test_df, classes, class_to_idx)."""
    df_ok = df[df["ok"]]

    if TASK_META[cfg.task]["layout"] == "split":
        # Brain tumor: pre-defined Training / Testing folders, val carved from Training
        train_all = df_ok[df_ok["split"] == "Training"]
        test_df   = df_ok[df_ok["split"] == "Testing"]
        train_df, val_df = train_test_split(
            train_all, test_size=0.15, stratify=train_all["label"], random_state=cfg.seed
        )
    else:
        # Breast cancer: stratified 70 / 15 / 15 from a single pool
        train_df, tmp = train_test_split(
            df_ok, test_size=0.30, stratify=df_ok["label"], random_state=cfg.seed
        )
        val_df, test_df = train_test_split(
            tmp, test_size=0.50, stratify=tmp["label"], random_state=cfg.seed
        )

    classes      = sorted(df_ok["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    log.info("Split sizes — train: %d  val: %d  test: %d  classes: %s",
             len(train_df), len(val_df), len(test_df), classes)
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True),
            test_df.reset_index(drop=True), classes, class_to_idx)


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict, tfm):
        self.df           = df
        self.class_to_idx = class_to_idx
        self.tfm          = tfm

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        r   = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tfm(img), self.class_to_idx[r["label"]]


def make_loader(df: pd.DataFrame, class_to_idx: dict, tfm, cfg: Config,
                shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ImageDataset(df, class_to_idx, tfm),
        batch_size=cfg.batch, shuffle=shuffle,
        num_workers=min(4, os.cpu_count() or 1), pin_memory=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stage 7 · Training
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Stop training when val_acc has not improved by min_delta for `patience` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None

    def __call__(self, val_acc: float) -> bool:
        """Record this epoch's val_acc; return True when training should stop."""
        if self.best is None or val_acc > self.best + self.min_delta:
            self.best    = val_acc
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def run_epoch(model, loader, criterion, device,
              optimizer=None, scaler=None, desc: str = "") -> tuple:
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates.
    Returns (mean loss, accuracy)."""
    is_train = optimizer is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(is_train):
        for x, y in tqdm(loader, leave=False, desc=desc or ("train" if is_train else "eval")):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # Mixed precision only while training on CUDA (scaler is None on CPU)
            with torch.amp.autocast(device_type=device.type,
                                    enabled=is_train and scaler is not None):
                out  = model(x)
                loss = criterion(out, y)
            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct  += (out.argmax(1) == y).sum().item()
            total    += x.size(0)
    return loss_sum / total, correct / total


def train_model(model, train_loader, val_loader, train_df, classes,
                cfg: Config, device, ckpt_path: Path, mlflow):
    """Train with early stopping, saving the best-val_acc checkpoint.
    Returns (best_val_acc, history rows)."""
    # Inverse-frequency class weights counter the class imbalance
    counts = train_df["label"].value_counts().reindex(classes).values.astype(float)
    class_weights = torch.tensor(
        counts.sum() / (len(classes) * counts), dtype=torch.float32
    ).to(device)

    criterion  = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer  = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler     = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    early_stop = EarlyStopping(patience=cfg.patience)
    best_val, history = 0.0, []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer=optimizer, scaler=scaler,
                                    desc=f"train e{epoch}")
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device,
                                    desc=f"val   e{epoch}")
        scheduler.step()
        elapsed = time.time() - t0

        history.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss, 4), "val_loss": round(vl_loss, 4),
            "train_acc":  round(tr_acc,  4), "val_acc":  round(vl_acc,  4),
            "elapsed_s":  round(elapsed,  1),
        })
        log.info("Epoch %02d/%02d  tr_loss=%.4f  tr_acc=%.4f  "
                 "vl_loss=%.4f  vl_acc=%.4f  (%.1fs)",
                 epoch, cfg.epochs, tr_loss, tr_acc, vl_loss, vl_acc, elapsed)
        mlflow.log_metrics({
            "train_loss": tr_loss, "val_loss": vl_loss,
            "train_acc":  tr_acc,  "val_acc":  vl_acc,
            "lr":         scheduler.get_last_lr()[0],
        }, step=epoch)

        if vl_acc > best_val:
            best_val = vl_acc
            # Everything needed for standalone inference lives in the checkpoint
            torch.save({
                "state_dict": model.state_dict(),
                "classes":    classes,
                "img_size":   cfg.img_size,
                "mean":       IMAGENET_MEAN,
                "std":        IMAGENET_STD,
                "epoch":      epoch,
                "val_acc":    vl_acc,
            }, ckpt_path)
            log.info("  [checkpoint] New best val_acc=%.4f saved.", best_val)

        if early_stop(vl_acc):
            log.info("Early stopping at epoch %d (no improvement for %d epochs).",
                     epoch, cfg.patience)
            break

    return best_val, history


# ══════════════════════════════════════════════════════════════════════════════
# Stage 8 · Model evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_test_set(model: nn.Module, loader, device, classes: list) -> tuple:
    """Return (y_true, y_pred, y_prob, report dict, confusion matrix, macro OvR AUROC)."""
    model.eval()
    all_y, all_prob = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, desc="test"):
            all_prob.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
            all_y.append(y.numpy())
    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_prob)
    y_pred = y_prob.argmax(1)
    report = classification_report(y_true, y_pred, target_names=classes,
                                   digits=4, output_dict=True)
    cm     = confusion_matrix(y_true, y_pred).tolist()
    auc    = roc_auc_score(y_true, y_prob, multi_class="ovr")
    return y_true, y_pred, y_prob, report, cm, auc


# ══════════════════════════════════════════════════════════════════════════════
# Stage 9 · Model validation gate
# ══════════════════════════════════════════════════════════════════════════════

def run_validation_gate(best_val_acc: float, test_auroc: float, cfg: Config) -> bool:
    """Return True if the model clears both quality thresholds."""
    acc_ok   = best_val_acc >= cfg.min_val_acc
    auroc_ok = test_auroc   >= cfg.min_auroc
    log.info("Gate — val_acc:    %.4f  (threshold %.2f) → %s",
             best_val_acc, cfg.min_val_acc, "PASS" if acc_ok else "FAIL")
    log.info("Gate — test_auroc: %.4f  (threshold %.2f) → %s",
             test_auroc, cfg.min_auroc, "PASS" if auroc_ok else "FAIL")
    return acc_ok and auroc_ok


# ══════════════════════════════════════════════════════════════════════════════
# Stage 10 · Model export
# ══════════════════════════════════════════════════════════════════════════════

def export_torchscript(model: nn.Module, img_size: int, out: Path) -> Path:
    """Trace the model to TorchScript for portable, framework-free inference."""
    model.eval().cpu()
    scripted = torch.jit.trace(model, torch.randn(1, 3, img_size, img_size))
    path     = out / "model.torchscript"
    torch.jit.save(scripted, str(path))
    log.info("TorchScript model saved → %s", path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("══ Stage 1 · Configuration management ══")
    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    cfg.save(out / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("task=%s | seed=%d | device=%s | epochs=%d | batch=%d | lr=%s",
             cfg.task, cfg.seed, device, cfg.epochs, cfg.batch, cfg.lr)

    log.info("══ Stage 2 · Data acquisition ══")
    download_dataset(cfg)

    log.info("══ Stage 3 · Data validation ══")
    df = index_images(cfg)
    if not df["ok"].any():
        raise FileNotFoundError(
            f"No valid images found under '{cfg.data}' even after download.\n"
            "Expected layouts:\n"
            "  brain_tumor:   data/Training/<class>/*.jpg\n"
            "  breast_cancer: data_breast/<class>/*.png"
        )
    data_stats = validate_data(df)
    (out / "data_stats.json").write_text(json.dumps(data_stats, indent=2))

    log.info("══ Stage 4 · Feature engineering ══")
    train_df, val_df, test_df, classes, class_to_idx = build_splits(df, cfg)
    test_df[["path", "label"]].to_csv(out / "test_images.csv", index=False)  # used by explain.py
    train_tfm, eval_tfm = build_transforms(cfg.img_size)
    train_loader = make_loader(train_df, class_to_idx, train_tfm, cfg, shuffle=True)
    val_loader   = make_loader(val_df,   class_to_idx, eval_tfm,  cfg)
    test_loader  = make_loader(test_df,  class_to_idx, eval_tfm,  cfg)

    log.info("══ Stage 5 · Experiment tracking (MLflow) ══")
    import mlflow
    import mlflow.pytorch
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.experiment)
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

        log.info("══ Stage 6 · Model building ══")
        model = build_model(len(classes)).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log.info("Parameters: %d", n_params)
        mlflow.log_params({"total_params": n_params, "trainable_params": n_params})

        log.info("══ Stage 7 · Training ══")
        ckpt_path = out / "best_model.pt"
        best_val, history = train_model(model, train_loader, val_loader, train_df,
                                        classes, cfg, device, ckpt_path, mlflow)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        mlflow.log_metric("best_val_acc",   best_val)
        mlflow.log_metric("epochs_trained", len(history))
        mlflow.log_artifact(str(out / "history.csv"))

        log.info("══ Stage 8 · Model evaluation ══")
        # Evaluate the best checkpoint, not the last epoch
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        y_true, y_pred, _, report, cm, auc = evaluate_test_set(
            model, test_loader, device, classes
        )
        report_path = out / "test_report.json"
        report_path.write_text(json.dumps({
            "task":                  cfg.task,
            "best_val_acc":          best_val,
            "test_macro_auroc":      auc,
            "test_accuracy":         report["accuracy"],
            "classification_report": report,
            "confusion_matrix":      {"labels": classes, "matrix": cm},
            "classes":               classes,
            "run_id":                run.info.run_id,
        }, indent=2))
        mlflow.log_metric("test_macro_auroc", auc)
        mlflow.log_metric("test_accuracy",    report["accuracy"])
        mlflow.log_artifact(str(report_path))
        log.info("\n%s", classification_report(y_true, y_pred,
                                               target_names=classes, digits=4))
        log.info("Test macro AUROC: %.4f", auc)

        log.info("══ Stage 9 · Model validation gate ══")
        gate_passed = run_validation_gate(best_val, auc, cfg)
        mlflow.log_metric("gate_passed", int(gate_passed))
        mlflow.set_tag("gate_passed", str(gate_passed))

        log.info("══ Stage 10 · Model export & registry ══")
        mlflow.log_artifact(str(export_torchscript(model, cfg.img_size, out)))
        # Always log the model; register it in the Model Registry only if the gate passes
        mlflow.pytorch.log_model(
            model,
            name="pytorch_model",
            registered_model_name=cfg.experiment if gate_passed else None,
            serialization_format="pickle",
        )
        if gate_passed:
            log.info("Model registered in MLflow Model Registry as '%s'.", cfg.experiment)
        else:
            log.warning("Model NOT registered — quality gate failed.")

        log.info("Pipeline complete | task=%s | run=%s | val_acc=%.4f | "
                 "test_auroc=%.4f | gate=%s | artifacts=%s/",
                 cfg.task, run.info.run_id, best_val, auc,
                 "PASS" if gate_passed else "FAIL", cfg.out)
        log.info("View UI: mlflow ui --backend-store-uri %s", cfg.mlflow_uri)


if __name__ == "__main__":
    main()
