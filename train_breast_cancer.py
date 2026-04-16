"""
End-to-end PyTorch training script for breast cancer ultrasound classification.

Dataset: Kaggle "Breast Ultrasound Images Dataset" (aryashah2k), 3 classes:
    benign, malignant, normal.

The dataset has no pre-defined train/test split.  A stratified 70 / 15 / 15
train / val / test split is applied automatically (random_state=42).

Kaggle credentials are resolved in this order:
  1. --kaggle_user / --kaggle_key command-line arguments
  2. Environment variables  KAGGLE_USERNAME  and  KAGGLE_KEY
  3. ~/.kaggle/kaggle.json  (standard Kaggle CLI credential file)

Folder layout after download:
    data_breast/benign/*.png
    data_breast/malignant/*.png
    data_breast/normal/*.png

Usage:
    # auto-download (credentials from env or ~/.kaggle/kaggle.json)
    python train_breast_cancer.py

    # pass credentials explicitly
    python train_breast_cancer.py --kaggle_user YOUR_NAME --kaggle_key YOUR_API_KEY

    # skip download if data already exists, override hyper-parameters
    python train_breast_cancer.py --data data_breast --epochs 20 --batch 32 --lr 3e-4
"""

import argparse, os, json, shutil, subprocess, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from PIL import Image
from tqdm import tqdm


# ========================= Kaggle helpers =================================== #

KAGGLE_DATASET = "aryashah2k/breast-ultrasound-images-dataset"


def get_kaggle_exe() -> str:
    """
    Find the kaggle CLI executable and return its path.
    If kaggle is not installed, installs it automatically via pip.

    We always call the kaggle CLI as a subprocess instead of importing the
    Python package, because kaggle v1.6+ removed KaggleApiExtended and broke
    the programmatic API.
    """
    # Places to look, in order of preference
    candidates = [
        shutil.which("kaggle"),                                              # system PATH
        shutil.which("kaggle.exe"),                                          # system PATH (Windows)
        str(Path(sys.executable).parent / "Scripts" / "kaggle.exe"),        # conda env Scripts/
        str(Path.home() / "AppData/Roaming/Python/Scripts/kaggle.exe"),     # pip --user on Windows
    ]
    exe = next((p for p in candidates if p and Path(p).exists()), None)

    if exe is None:
        # Not found anywhere — install it now, then search again
        print("kaggle CLI not found — installing via pip ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
        exe = shutil.which("kaggle") or shutil.which("kaggle.exe") or "kaggle"

    return exe


def setup_kaggle_credentials(args):
    """
    Make sure ~/.kaggle/kaggle.json exists with valid credentials.

    Credential lookup order:
      1. --kaggle_user / --kaggle_key  (command-line flags)
      2. KAGGLE_USERNAME / KAGGLE_KEY  (environment variables)
      3. ~/.kaggle/kaggle.json         (file already on disk — nothing to do)

    If none of the above are available, raises a RuntimeError with setup
    instructions so the user knows exactly what to do.
    """
    user = args.kaggle_user or os.getenv("KAGGLE_USERNAME", "").strip()
    key  = args.kaggle_key  or os.getenv("KAGGLE_KEY",      "").strip()
    cred = Path.home() / ".kaggle" / "kaggle.json"

    if user and key:
        # Write the credentials file so the kaggle CLI can read them
        cred.parent.mkdir(exist_ok=True)
        cred.write_text(json.dumps({"username": user, "key": key}))
        try:
            cred.chmod(0o600)   # owner-read-only; silently ignored on Windows
        except Exception:
            pass
        print(f"Kaggle credentials written to {cred}")

    elif cred.exists():
        print(f"Using existing Kaggle credentials at {cred}")

    else:
        raise RuntimeError(
            "Kaggle credentials not found.\n\n"
            "Option A — pass them on the command line:\n"
            "    python train_breast_cancer.py --kaggle_user NAME --kaggle_key KEY\n\n"
            "Option B — save them once (recommended):\n"
            "    1. Go to https://www.kaggle.com/settings  ->  API  ->  Create New Token\n"
            f"    2. Move the downloaded kaggle.json to:  {cred}\n"
            "    Then just run the script with no extra flags.\n"
        )


def download_dataset(data_dir: Path, args):
    """
    Download the Breast Ultrasound Images dataset from Kaggle into data_dir.
    Skips the download entirely if the class folders are already on disk.

    Steps:
      1. Find (or install) the kaggle CLI
      2. Ensure credentials are in place
      3. Download the dataset zip
      4. Extract the zip, then delete it to save space
      5. Flatten the nested sub-folder the archive creates (if present)

    After extraction the archive produces:
        data_breast/Dataset_BUSI_with_GT/benign/
        data_breast/Dataset_BUSI_with_GT/malignant/
        data_breast/Dataset_BUSI_with_GT/normal/
    which is flattened to:
        data_breast/benign/
        data_breast/malignant/
        data_breast/normal/
    """
    # Skip if the data is already there
    known_classes = ["benign", "malignant", "normal"]
    if any(
        (data_dir / cls).exists() and any((data_dir / cls).glob("*.png"))
        for cls in known_classes
    ):
        print(f"Dataset already present at '{data_dir}' — skipping download.")
        return

    print(f"Downloading dataset to '{data_dir}' ...")
    data_dir.mkdir(parents=True, exist_ok=True)

    exe = get_kaggle_exe()
    setup_kaggle_credentials(args)

    # Download via the kaggle CLI
    result = subprocess.run(
        [exe, "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(data_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed:\n{result.stderr}")
    print(result.stdout.strip())

    # Extract the zip, then remove it
    zips = list(data_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip found in '{data_dir}' after download.")
    print(f"Extracting {zips[0].name} ...")
    with zipfile.ZipFile(zips[0], "r") as zf:
        zf.extractall(data_dir)
    zips[0].unlink()

    # The archive extracts into Dataset_BUSI_with_GT/ — move contents up one level
    nested = data_dir / "Dataset_BUSI_with_GT"
    if nested.exists():
        for item in nested.iterdir():
            dest = data_dir / item.name
            if dest.exists():
                shutil.rmtree(str(dest))
            shutil.move(str(item), str(dest))
        nested.rmdir()

    print(f"Dataset ready at '{data_dir}'.")


# ========================= Dataset ========================================== #

class BreastDataset(Dataset):
    def __init__(self, df, class_to_idx, tfm):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tfm(img), self.class_to_idx[r["label"]]


def index_folder(root: Path):
    """
    Walk <root>/<class>/ subdirectories and record every image file.

    The BUSI dataset ships paired segmentation masks alongside every scan
    (e.g. 'benign (1) mask.png').  Those are filtered out by checking for
    the word 'mask' in the filename so they are never included as training
    samples.

    Returns a DataFrame with columns: path, label, ok, err.
    """
    rows = []
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.glob("*"):
            if p.suffix.lower() not in image_exts:
                continue
            if "mask" in p.name.lower():
                continue   # skip segmentation masks
            try:
                with Image.open(p) as im:
                    im.verify()
                rows.append({"path": str(p), "label": cls_dir.name,
                             "ok": True, "err": None})
            except Exception as e:
                rows.append({"path": str(p), "label": cls_dir.name,
                             "ok": False, "err": str(e)})
    COLS = ["path", "label", "ok", "err"]
    return pd.DataFrame(rows, columns=COLS) if rows else pd.DataFrame(columns=COLS)


# ========================= Model ============================================ #

def build_model(num_classes: int):
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512), nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return m


# ========================= Train / eval loop ================================ #

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    out = model(x); loss = criterion(out, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            else:
                out = model(x); loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            correct  += (out.argmax(1) == y).sum().item()
            total    += x.size(0)
    return loss_sum / total, correct / total


# ========================= Main ============================================= #

def main():
    ap = argparse.ArgumentParser(
        description="Breast cancer ultrasound classifier — auto-downloads data from Kaggle."
    )
    # data / output
    ap.add_argument("--data",    type=str, default="data_breast",
                    help="Root data directory (created & populated automatically).")
    ap.add_argument("--out",     type=str, default="artifacts_breast",
                    help="Directory for checkpoints, history, and reports.")
    # kaggle credentials
    ap.add_argument("--kaggle_user", type=str, default="",
                    help="Kaggle username  (or set KAGGLE_USERNAME env var).")
    ap.add_argument("--kaggle_key",  type=str, default="",
                    help="Kaggle API key   (or set KAGGLE_KEY env var).")
    # training hyper-parameters
    ap.add_argument("--epochs",   type=int,   default=20)
    ap.add_argument("--batch",    type=int,   default=32)
    ap.add_argument("--lr",       type=float, default=3e-4)
    ap.add_argument("--img_size", type=int,   default=224)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---- 1. Download dataset if needed -------------------------------------- #
    download_dataset(Path(args.data), args)

    # ---- 2. Index all images ------------------------------------------------ #
    df = index_folder(Path(args.data))
    if df.empty or not df["ok"].any():
        raise FileNotFoundError(
            f"No valid images found under '{args.data}' even after download.\n"
            "Expected layout:\n"
            "  data_breast/benign/*.png\n"
            "  data_breast/malignant/*.png\n"
            "  data_breast/normal/*.png"
        )
    corrupt = (~df["ok"]).sum()
    if corrupt:
        print(f"Warning: {corrupt} corrupt/unreadable files skipped.")
    df = df[df["ok"]].copy()

    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")
    print(df.groupby("label").size().to_string())

    # ---- 3. Stratified 70 / 15 / 15 splits ---------------------------------- #
    train_df, tmp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, stratify=tmp_df["label"], random_state=42
    )
    print(f"Split sizes — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    # Persist test paths so explain_breast_cancer.py can reuse the same hold-out
    test_df[["path", "label"]].to_csv(out / "test_images.csv", index=False)

    # ---- 4. Transforms ------------------------------------------------------ #
    MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
    train_tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])
    eval_tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # ---- 5. DataLoaders ----------------------------------------------------- #
    nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        BreastDataset(train_df, class_to_idx, train_tfm),
        batch_size=args.batch, shuffle=True, num_workers=nw, pin_memory=True
    )
    val_loader = DataLoader(
        BreastDataset(val_df, class_to_idx, eval_tfm),
        batch_size=args.batch, shuffle=False, num_workers=nw, pin_memory=True
    )
    test_loader = DataLoader(
        BreastDataset(test_df, class_to_idx, eval_tfm),
        batch_size=args.batch, shuffle=False, num_workers=nw, pin_memory=True
    )

    # ---- 6. Model, loss, optimizer ------------------------------------------ #
    counts = train_df["label"].value_counts().reindex(classes).values
    class_weights = torch.tensor(
        counts.sum() / (len(classes) * counts), dtype=torch.float32
    ).to(device)

    model     = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ---- 7. Training loop --------------------------------------------------- #
    best_val, history = 0.0, []
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer=optimizer, scaler=scaler)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {"epoch": epoch + 1,
               "train_loss": round(tr_loss, 4), "val_loss": round(vl_loss, 4),
               "train_acc":  round(tr_acc, 4),  "val_acc":  round(vl_acc, 4)}
        history.append(row)
        print(row)
        if vl_acc > best_val:
            best_val = vl_acc
            torch.save({"state_dict": model.state_dict(),
                        "classes": classes, "img_size": args.img_size,
                        "mean": MEAN, "std": STD},
                       out / "best_model.pt")
            print(f"  [saved] New best val_acc={best_val:.4f} — checkpoint saved.")

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # ---- 8. Test-set evaluation ---------------------------------------------- #
    ckpt = torch.load(out / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    all_y, all_p, all_prob = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            prob = torch.softmax(model(x), dim=1).cpu().numpy()
            all_prob.append(prob)
            all_p.append(prob.argmax(1))
            all_y.append(y.numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    y_prob = np.concatenate(all_prob)

    report = classification_report(y_true, y_pred, target_names=classes,
                                   digits=4, output_dict=True)
    cm  = confusion_matrix(y_true, y_pred).tolist()
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr")

    summary = {"best_val_acc": best_val, "test_macro_auroc": auc,
               "classification_report": report,
               "confusion_matrix": {"labels": classes, "matrix": cm},
               "classes": classes}
    (out / "test_report.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Final test results ===")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print(json.dumps({"best_val_acc": best_val, "test_macro_auroc": auc}, indent=2))
    print(f"\nArtifacts saved to '{out}/'")


if __name__ == "__main__":
    main()
