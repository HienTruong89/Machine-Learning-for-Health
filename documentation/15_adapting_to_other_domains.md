# Adapting the template to non-image domains

The 10-stage MLOps skeleton (Config → Acquire → Validate → Features → Track → Build → Train → Evaluate → Gate → Export) is domain-agnostic. Only four stages have image-classification-specific code: **2 (acquisition)**, **4 (features/dataset)**, **5/6 (model)**, and **8 (metrics)**.

This doc shows the concrete swap for four common domains. For each, only the changed pieces are shown — everything else from `mlops_pipeline.py` reuses verbatim.

## 1. Tabular classification (e.g. patient outcome from EHR)

### Stage 2 — Acquisition

Replace `kagglehub.dataset_download` with whatever produces a CSV / Parquet:

```python
import pandas as pd

def load_tabular(cfg) -> pd.DataFrame:
    return pd.read_csv(cfg.data)        # or read_parquet, read_sql, ...
```

### Stage 3 — Validation

Different hard-fail rules:

```python
def validate_tabular(df: pd.DataFrame) -> dict:
    null_pct = df.isna().mean().to_dict()
    high_null = {k: v for k, v in null_pct.items() if v > 0.30}
    if high_null:
        raise ValueError(f"Columns >30% null: {high_null}")
    if df.duplicated().any():
        raise ValueError(f"{df.duplicated().sum()} duplicate rows")
    return {
        "n_rows":    len(df),
        "n_cols":    df.shape[1],
        "null_pct":  {k: round(v, 4) for k, v in null_pct.items()},
        "label_dist": df["target"].value_counts().to_dict(),
    }
```

### Stage 4 — Features

```python
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

NUMERIC = ["age", "bmi", "lab_result"]
CATEGOR = ["sex", "diagnosis_code"]

def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGOR),
    ])

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]
```

The fitted preprocessor goes in the checkpoint alongside the state_dict so serving uses the *exact same* transformation:

```python
import joblib
joblib.dump(preprocessor, out / "preprocessor.joblib")
torch.save({
    "state_dict": model.state_dict(),
    "classes":    classes,
    "feature_names": preprocessor.get_feature_names_out().tolist(),
    "preprocessor_path": "preprocessor.joblib",
}, out / "best_model.pt")
```

### Stage 5/6 — Model

```python
import torch.nn as nn

def build_model(in_features: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128),         nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
```

For mixed numeric + categorical, consider [TabNet](https://github.com/dreamquark-ai/tabnet) or [FT-Transformer](https://github.com/yandex-research/rtdl-revisiting-models) — but XGBoost is often the right answer for pure tabular and beats deep nets at this scale. If you go XGBoost, drop PyTorch entirely; the rest of the pipeline (config, validation, MLflow, gate) still applies.

### Stage 8 — Metrics

Same as image classification (`classification_report`, `roc_auc_score`). Add **calibration** — tabular models often need Platt scaling or isotonic regression before deployment:

```python
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_true, y_prob[:, 1], n_bins=10)
```

## 2. Text classification (e.g. clinical-note triage)

### Stage 4 — Features

```python
from transformers import AutoTokenizer
from torch.utils.data import Dataset

TOK = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

class TextDataset(Dataset):
    def __init__(self, texts, labels, max_len=256):
        self.texts, self.labels, self.max_len = texts, labels, max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc = TOK(self.texts[i], truncation=True, padding="max_length",
                  max_length=self.max_len, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, self.labels[i]
```

Augmentation for text is harder — skip it for v1, add back-translation or [nlpaug](https://github.com/makcedward/nlpaug) only if you're underfitting.

### Stage 5/6 — Model

```python
from transformers import AutoModel
import torch.nn as nn

class ClinicalClassifier(nn.Module):
    def __init__(self, num_classes: int, ckpt="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ckpt)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.encoder.config.hidden_size, num_classes),
        )
    def forward(self, input_ids, attention_mask, **_):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0])      # [CLS] token
```

The training loop changes only at `model(**batch)` — adapt `run_epoch` to unpack the dict:

```python
for batch, y in loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    out   = model(**batch)
```

### Stage 8 — Metrics

For multi-class same as before. For multi-label, switch to `BCEWithLogitsLoss`, threshold at 0.5, report micro/macro F1 instead of accuracy.

### Stage 11 — UI

Streamlit text input replaces file upload:

```python
text = st.text_area("Clinical note", height=200)
if st.button("Classify") and text:
    enc = TOK(text, truncation=True, padding="max_length",
              max_length=256, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc), dim=1)[0].tolist()
```

## 3. Semantic segmentation (e.g. tumor mask from MRI)

The biggest jump from classification — labels become *images*, not integers.

### Stage 4 — Dataset

```python
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths, tfm):
        self.imgs, self.masks, self.tfm = img_paths, mask_paths, tfm
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img  = np.array(Image.open(self.imgs[i]).convert("RGB"))
        mask = np.array(Image.open(self.masks[i]).convert("L"))    # 0=bg, 1=lesion
        out  = self.tfm(image=img, mask=mask)
        return out["image"], out["mask"].long()
```

Use `albumentations` instead of `torchvision.transforms` — it applies the same geometric ops to image *and* mask in one call.

```python
train_tfm = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(),
    A.RandomRotate90(),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
```

### Stage 5/6 — Model

```python
import segmentation_models_pytorch as smp

def build_model(num_classes: int):
    return smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,        # for binary use 1 + sigmoid; for multi-class use N + softmax
    )

# Loss: combination of Dice (handles class imbalance) + CE (sharp boundaries)
criterion = smp.losses.DiceLoss(mode="multiclass") + nn.CrossEntropyLoss()
```

### Stage 8 — Metrics

Throw out accuracy. Use IoU and Dice — both per-class and mean:

```python
def iou(pred, target, num_classes):
    pred = pred.argmax(1)
    ious = []
    for c in range(num_classes):
        intersection = ((pred == c) & (target == c)).sum().item()
        union        = ((pred == c) | (target == c)).sum().item()
        ious.append(intersection / union if union else float("nan"))
    return ious
```

The quality gate becomes "mean IoU > 0.6" or "Dice > 0.7" instead of accuracy / AUROC.

### Stage 11 — UI

Visualize the predicted mask overlaid on the input:

```python
import numpy as np
overlay = np.array(img).copy()
overlay[pred_mask == 1] = [255, 0, 0]   # red where lesion predicted
st.image(overlay, caption="Predicted mask")
```

## 4. Regression (e.g. ICU length-of-stay)

### Stage 5/6 — Model + loss

```python
def build_model(in_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 64),          nn.ReLU(),
        nn.Linear(64, 1),                                  # single scalar output
    )

criterion = nn.SmoothL1Loss()      # robust to outliers; or MSELoss / L1Loss
```

### Stage 7 — Training loop

`accuracy = (out.argmax(1) == y).sum()` no longer makes sense. Track loss only inside `run_epoch`, compute regression metrics at the end of evaluation:

```python
def run_epoch_reg(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
            if is_train:
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward(); optimizer.step()
            else:
                loss = criterion(model(x), y)
            loss_sum += loss.item() * x.size(0); n += x.size(0)
    return loss_sum / n
```

### Stage 8 — Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
```

Quality gate: `mae < <clinical_acceptable_error>` and `r2 > 0.5`.

### Stage 11 — UI

Show the prediction with a confidence interval (fit a quantile-regression head, or train an ensemble and report std):

```python
st.metric("Predicted LOS", f"{pred:.1f} days", delta=f"±{std:.1f} CI")
```

## What stays unchanged across all four

Reuse from the original docs without modification:

- **Stage 1** — `Config` dataclass + `set_seed`. Always.
- **Stage 3** — fingerprint pattern (md5 of sorted record IDs/paths). Always.
- **Stage 5** — MLflow `start_run` block with tags / params / metrics / artifacts. Always.
- **Stage 6 (loop shape)** — train one epoch, eval one epoch, save best checkpoint, early-stop. Always.
- **Stage 9** — gate: pass two thresholds → register; fail → log artifacts but skip registry.
- **Stage 10** — FastAPI server with lazy load + `/health` + `/predict/{task}`.
- **Stage 11** — Streamlit shell, HF Hub model download, `@st.cache_resource`.
- **Stage 12** — GitHub Actions: install → smoke train → assert minimum metric → build Docker → upload artifacts.
- **Stage 13** — every CLI command (Git, Docker, Azure, HF) is task-agnostic.

## Decision matrix — which template piece to swap

| Domain | Stage 2 | Stage 4 | Stage 5 | Stage 8 |
|--------|---------|---------|---------|---------|
| Image classification | kagglehub | torchvision tfms | ResNet50 + MLP head | report + AUROC |
| Tabular classification | DB / CSV | scaler + encoder | MLP / XGBoost | report + AUROC + calibration |
| Text classification | csv / API | HF tokenizer | HF encoder + linear | report + F1 |
| Segmentation | image+mask pairs | albumentations | U-Net (smp) | IoU / Dice |
| Regression | DB / CSV | scaler + encoder | MLP, single output | MAE / RMSE / R² |
| Object detection | COCO-style JSON | torchvision detection tfms | Faster R-CNN / DETR | mAP @ IoU |
| Audio classification | wav files | torchaudio mel-spec | EfficientNet on spec | report + AUROC |

For object detection and audio, the same template applies — different Stage 4/5/8, identical Stages 1/3/7/9/10/11/12.

## When to abandon the template

The template assumes:
1. A **fixed dataset** that fits on disk and can be split train/val/test.
2. A **supervised** signal (labels exist).
3. A **batch training** loop with epochs.
4. A **pointwise prediction** at serving time (one input → one output).

Drop or rewrite the template if you're doing:
- **Generative modeling** (diffusion, LLM fine-tuning) — different loop, different metrics, streaming serving.
- **Reinforcement learning** — no static dataset; the data IS the policy.
- **Online / streaming learning** — model updates continuously; gate becomes shadow-mode comparison.
- **Recommender systems at scale** — user×item matrix, retrieval + re-ranking, A/B tests instead of held-out test sets.

For those, the *philosophy* (config-as-code, fingerprint your data, gate before promote, separate training and serving deps) still applies — just not the literal `mlops_pipeline.py`.
