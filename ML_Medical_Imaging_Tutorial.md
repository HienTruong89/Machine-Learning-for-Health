# Machine Learning with PyTorch for Medical Imaging
## End-to-End Tutorial: Tumor/Cancer Prediction from MRI / CT / PET Scans

This tutorial walks through a complete, production-oriented machine learning pipeline in PyTorch for classifying medical images (example: **Brain Tumor MRI classification** — 4 classes: glioma, meningioma, pituitary, no tumor). The same pipeline generalizes to breast cancer (mammography / histopathology), lung CT, PET oncology scans, etc.

---

## Table of Contents

1. Project scoping and dataset selection
2. Environment setup
3. Data acquisition
4. Data cleaning and quality control
5. Exploratory data analysis (EDA) and visualization
6. Preprocessing and augmentation
7. Dataset and DataLoader design
8. Model architecture (transfer learning with CNNs)
9. Training loop with validation
10. Evaluation on the test set
11. Model interpretation (Grad-CAM, saliency)
12. Saving and exporting the model
13. Deployment (FastAPI + inference script)
14. Monitoring, ethics, and regulatory notes

---

## Step 1 — Project scoping and dataset selection

**Instructions.** Before writing any code, define:

- **Clinical task**: binary (tumor / no-tumor), multi-class (tumor type), segmentation (pixel-level mask), or survival prediction?
- **Imaging modality**: MRI (T1, T1c, T2, FLAIR), CT, PET, or multi-modal?
- **Target users**: radiologist decision-support vs. triage vs. research.
- **Success metrics**: accuracy is rarely enough. Use **AUROC, sensitivity, specificity, F1, and per-class recall**. In cancer, **sensitivity (recall for positive class)** is typically prioritized.
- **Ethics / regulation**: HIPAA / GDPR, IRB approval if using clinical data, FDA / CE requirements if deploying.

**Public datasets to choose from:**

| Task | Modality | Dataset |
|---|---|---|
| Brain tumor classification | MRI | Kaggle "Brain Tumor MRI Dataset" (Masoud Nickparvar) — 7,023 images, 4 classes |
| Brain tumor segmentation | MRI | BraTS (Brain Tumor Segmentation Challenge) |
| Breast cancer (histopathology) | Microscopy | BreakHis, PatchCamelyon |
| Breast cancer (mammography) | X-ray | CBIS-DDSM, RSNA Mammography |
| Lung nodule | CT | LUNA16, LIDC-IDRI |
| Chest pathology | X-ray | NIH ChestX-ray14, CheXpert |
| PET/CT oncology | PET+CT | AutoPET, TCIA collections |

This tutorial uses the **Brain Tumor MRI Dataset** (Kaggle). It is small, well-labeled, 2D, and ideal for a first end-to-end project.

---

## Step 2 — Environment setup

```bash
# Python 3.10+
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib seaborn scikit-learn
pip install pillow opencv-python albumentations
pip install pydicom nibabel SimpleITK        # for DICOM / NIfTI
pip install grad-cam                          # interpretation
pip install fastapi uvicorn python-multipart  # deployment
pip install tqdm tensorboard
```

Quick sanity check:

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

## Step 3 — Data acquisition

For Kaggle:

```bash
pip install kaggle
# put kaggle.json in ~/.kaggle/
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d data/
```

Resulting folder structure:

```
data/
  Training/
    glioma/        *.jpg
    meningioma/    *.jpg
    notumor/       *.jpg
    pituitary/     *.jpg
  Testing/
    glioma/ ...
```

For DICOM/NIfTI sources (CT, MRI volumes), use `pydicom` or `nibabel`:

```python
import pydicom, nibabel as nib
dcm = pydicom.dcmread("scan.dcm"); img = dcm.pixel_array
vol = nib.load("brain.nii.gz").get_fdata()   # shape (H, W, D)
```

---

## Step 4 — Data cleaning and quality control

Typical issues: corrupted files, duplicates, wrong labels, inconsistent sizes, patient leakage between train/test.

```python
import os, hashlib
from pathlib import Path
from PIL import Image
import pandas as pd

ROOT = Path("data")

def file_hash(p, chunk=1<<16):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()

records = []
for split in ["Training", "Testing"]:
    for cls_dir in (ROOT/split).iterdir():
        if not cls_dir.is_dir(): continue
        for img_path in cls_dir.glob("*"):
            try:
                with Image.open(img_path) as im:
                    im.verify()                      # corrupt check
                    w, h = im.size
                records.append({
                    "path": str(img_path), "split": split,
                    "label": cls_dir.name, "w": w, "h": h,
                    "hash": file_hash(img_path), "ok": True,
                })
            except Exception as e:
                records.append({"path": str(img_path), "ok": False, "err": str(e)})

df = pd.DataFrame(records)
print("Corrupt files:", (~df.ok).sum())

# Duplicates (same image in multiple classes = label noise)
dups = df[df.duplicated("hash", keep=False)].sort_values("hash")
print("Duplicate hashes:", len(dups))

# Patient leakage: if filename contains a patient id, ensure no id appears in both splits
# e.g. df["pid"] = df.path.str.extract(r"(P\d+)")
```

Cleaning actions: drop corrupt files, remove cross-split duplicates, and if labels were crowdsourced, spot-check with a clinician or consensus labels.

---

## Step 5 — Exploratory data analysis & visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Class balance
ax = sns.countplot(data=df[df.ok], x="label", hue="split")
ax.set_title("Images per class per split"); plt.show()

# Image size distribution
sns.scatterplot(data=df[df.ok], x="w", y="h", hue="label", alpha=0.5)
plt.title("Image resolutions"); plt.show()

# Sample grid
import random
fig, axes = plt.subplots(4, 6, figsize=(14, 10))
for row, cls in enumerate(sorted(df.label.dropna().unique())):
    samples = df[(df.label==cls)&(df.split=="Training")].sample(6, random_state=0)
    for col, (_, r) in enumerate(samples.iterrows()):
        img = Image.open(r.path).convert("L")
        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].set_title(cls, fontsize=9)
        axes[row, col].axis("off")
plt.tight_layout(); plt.show()

# Mean / std pixel intensity per class (feature bias check)
import numpy as np
stats = []
for _, r in df[df.ok].sample(min(len(df), 1500), random_state=0).iterrows():
    a = np.asarray(Image.open(r.path).convert("L"))/255.0
    stats.append({"label": r.label, "mean": a.mean(), "std": a.std()})
sns.boxplot(data=pd.DataFrame(stats), x="label", y="mean")
plt.title("Mean intensity by class"); plt.show()
```

Key things to look for: class imbalance (weight the loss), very small or elongated images (standardize size), bias (e.g., one class is systematically brighter — could be a shortcut the model exploits).

---

## Step 6 — Preprocessing and augmentation

Medical-image-appropriate augmentation: mild rotation, flips (be careful — for mammography, left/right flip changes anatomy labels), brightness/contrast, elastic deformation, random erasing. Avoid augmentation that creates unrealistic anatomy.

```python
from torchvision import transforms

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats (fine for transfer learning)
STD  = [0.229, 0.224, 0.225]

train_tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),          # replicate to 3ch for ImageNet backbone
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
])

eval_tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
```

---

## Step 7 — Dataset, splits, and DataLoader

We split Training into train/val (stratified) and keep Testing untouched as a true hold-out.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

CLASSES = sorted(os.listdir("data/Training"))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class MRIDataset(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True); self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.path).convert("RGB")
        return self.tfm(img), CLASS_TO_IDX[r.label]

train_df = df[(df.split=="Training") & df.ok].copy()
test_df  = df[(df.split=="Testing")  & df.ok].copy()

train_df, val_df = train_test_split(
    train_df, test_size=0.15, stratify=train_df.label, random_state=42
)

BATCH = 32
train_loader = DataLoader(MRIDataset(train_df, train_tfm), BATCH, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(MRIDataset(val_df,   eval_tfm),  BATCH, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(MRIDataset(test_df,  eval_tfm),  BATCH, shuffle=False, num_workers=4, pin_memory=True)
```

Handle class imbalance via a weighted loss:

```python
counts = train_df.label.value_counts().reindex(CLASSES).values
class_weights = torch.tensor(counts.sum()/(len(CLASSES)*counts), dtype=torch.float32)
```

---

## Step 8 — Model architecture

Transfer learning with a pretrained ResNet-50 is a strong baseline for 2D medical imaging. For 3D volumes, use MONAI's `DenseNet121` / `resnet3d`.

```python
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_model(num_classes, freeze_backbone=False):
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    if freeze_backbone:
        for p in m.parameters(): p.requires_grad = False
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512), nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return m

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(len(CLASSES)).to(device)
```

---

## Step 9 — Training with validation

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20)
scaler    = torch.cuda.amp.GradScaler()

EPOCHS = 20; best_val = 0.0
history = []

for epoch in range(EPOCHS):
    model.train(); tr_loss = tr_correct = tr_total = 0
    for x, y in tqdm(train_loader, desc=f"train {epoch+1}"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x); loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        tr_loss += loss.item()*x.size(0)
        tr_correct += (out.argmax(1)==y).sum().item(); tr_total += x.size(0)
    scheduler.step()

    # validation
    model.eval(); v_correct = v_total = 0; v_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x); v_loss += criterion(out, y).item()*x.size(0)
            v_correct += (out.argmax(1)==y).sum().item(); v_total += x.size(0)

    tr_acc, v_acc = tr_correct/tr_total, v_correct/v_total
    history.append({"epoch": epoch+1, "train_loss": tr_loss/tr_total,
                    "val_loss": v_loss/v_total, "train_acc": tr_acc, "val_acc": v_acc})
    print(history[-1])

    if v_acc > best_val:
        best_val = v_acc
        torch.save(model.state_dict(), "best_model.pt")

pd.DataFrame(history).to_csv("history.csv", index=False)
```

Plot learning curves:

```python
h = pd.DataFrame(history)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
h[["train_loss","val_loss"]].plot(ax=ax[0], title="Loss")
h[["train_acc","val_acc"]].plot(ax=ax[1], title="Accuracy"); plt.show()
```

---

## Step 10 — Evaluation on the test set

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

model.load_state_dict(torch.load("best_model.pt")); model.eval()
all_y, all_p, all_prob = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        prob = torch.softmax(model(x), dim=1).cpu().numpy()
        all_prob.append(prob); all_p.append(prob.argmax(1)); all_y.append(y.numpy())
import numpy as np
y_true = np.concatenate(all_y); y_pred = np.concatenate(all_p)
y_prob = np.concatenate(all_prob)

print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
plt.xlabel("Pred"); plt.ylabel("True"); plt.show()

# Multi-class AUROC (one-vs-rest)
auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
print("Macro AUROC:", auc)
```

Report per-class **sensitivity** (recall) and **specificity** — these are what clinicians care about.

---

## Step 11 — Model interpretation (Grad-CAM)

Grad-CAM shows which image regions drove the prediction — essential for clinical trust and for catching shortcut learning.

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

img_tensor, label = next(iter(test_loader))
input_t = img_tensor[:1].to(device)
pred = int(model(input_t).argmax(1).item())
grayscale_cam = cam(input_tensor=input_t, targets=[ClassifierOutputTarget(pred)])[0]

# Un-normalize for display
rgb = img_tensor[0].numpy().transpose(1,2,0)
rgb = (rgb*np.array(STD)+np.array(MEAN)).clip(0,1)
vis = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
plt.imshow(vis); plt.title(f"Pred: {CLASSES[pred]}  True: {CLASSES[label[0]]}"); plt.show()
```

Additional interpretation tools: Integrated Gradients and SHAP (via `captum`), occlusion maps, and for segmentation models, Dice per-region overlays.

---

## Step 12 — Saving and exporting the model

```python
# PyTorch native
torch.save({"state_dict": model.state_dict(),
            "classes": CLASSES,
            "img_size": IMG_SIZE,
            "mean": MEAN, "std": STD}, "mri_classifier.pt")

# TorchScript (portable)
scripted = torch.jit.script(model.cpu().eval())
scripted.save("mri_classifier_ts.pt")

# ONNX (for cross-framework / hardware accelerators)
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
torch.onnx.export(model.cpu().eval(), dummy, "mri_classifier.onnx",
                  input_names=["input"], output_names=["logits"],
                  opset_version=17, dynamic_axes={"input": {0: "batch"}})
```

---

## Step 13 — Deployment

### 13a — Stand-alone inference script

```python
# predict.py
import sys, torch
from PIL import Image
from torchvision import transforms

ckpt = torch.load("mri_classifier.pt", map_location="cpu")
CLASSES = ckpt["classes"]

from model import build_model     # same factory as training
model = build_model(len(CLASSES)); model.load_state_dict(ckpt["state_dict"]); model.eval()

tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(ckpt["mean"], ckpt["std"]),
])

img = Image.open(sys.argv[1]).convert("RGB")
with torch.no_grad():
    prob = torch.softmax(model(tfm(img).unsqueeze(0)), dim=1)[0]
for c, p in sorted(zip(CLASSES, prob.tolist()), key=lambda x: -x[1]):
    print(f"{c:>12s}  {p:.3f}")
```

### 13b — REST API with FastAPI

```python
# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io, torch
from torchvision import transforms
from model import build_model

app = FastAPI(title="MRI Tumor Classifier")

ckpt = torch.load("mri_classifier.pt", map_location="cpu")
CLASSES = ckpt["classes"]
model = build_model(len(CLASSES)); model.load_state_dict(ckpt["state_dict"]); model.eval()

tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(ckpt["mean"], ckpt["std"]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    with torch.no_grad():
        prob = torch.softmax(model(tfm(img).unsqueeze(0)), dim=1)[0].tolist()
    ranked = sorted(zip(CLASSES, prob), key=lambda x: -x[1])
    return JSONResponse({
        "prediction": ranked[0][0],
        "confidence": ranked[0][1],
        "probabilities": dict(ranked),
        "disclaimer": "Research use only. Not a medical device."
    })

# run:  uvicorn app:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl -F "file=@scan.jpg" http://localhost:8000/predict
```

### 13c — Containerize

```dockerfile
FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

For production scale: TorchServe, Triton Inference Server, or ONNX Runtime behind Kubernetes, with GPU nodes autoscaled.

---

## Step 14 — Monitoring, ethics, and regulatory notes

- **Data drift / performance drift**: log prediction distributions, track per-site performance, re-evaluate quarterly on fresh labeled data.
- **Bias audit**: stratify test performance by sex, age, scanner vendor, site — a model that works well on Siemens and poorly on GE is unsafe.
- **Uncertainty**: expose predicted probability and consider MC-Dropout or deep ensembles for calibrated uncertainty; route low-confidence cases to a human.
- **Explainability logs**: store Grad-CAMs alongside predictions for audit.
- **Regulation**: in the US, diagnostic use likely requires FDA clearance (SaMD). In the EU, the AI Act + MDR. This tutorial's model is **research-only**.
- **Privacy**: de-identify DICOM headers; never commit PHI to version control; use encrypted storage at rest and in transit.
- **Clinical validation**: retrospective internal test → retrospective external (multi-site) → prospective silent trial → prospective interventional trial.

---

## Appendix — Quick checklist

```
[ ] Task & metrics defined with clinical stakeholder
[ ] Dataset licensed and de-identified
[ ] Patient-level split (no leakage)
[ ] Class balance handled
[ ] Augmentation is anatomically valid
[ ] Baseline (transfer learning) established
[ ] Validation curves show no severe overfit
[ ] Per-class sensitivity / specificity / AUROC reported
[ ] Grad-CAM sanity-checked for shortcut learning
[ ] External test set evaluated
[ ] Model exported (TorchScript / ONNX)
[ ] Inference API tested with latency + memory budget
[ ] Monitoring, drift alerts, bias audit in place
[ ] Ethics / regulatory pathway documented
```

---

*This tutorial is for educational and research purposes only and is not a medical device.*
