# Stage 4 — Dataset, transforms, splits

**Goal:** turn the validated DataFrame into PyTorch `DataLoader`s that augment training data, leave eval data alone, and never leak between splits.

## Stratified splits

Brain-tumor MRI ships with `Training/` and `Testing/` predefined; carve a stratified val out of `Training/`. Breast-cancer ultrasound is one flat pool — do a 70/15/15 stratified split with a fixed seed.

```python
from sklearn.model_selection import train_test_split

def build_splits(df, cfg):
    meta  = TASK_META[cfg.task]
    df_ok = df[df["ok"]].copy()

    if meta["layout"] == "split":
        train_all = df_ok[df_ok["split"] == "Training"]
        test_df   = df_ok[df_ok["split"] == "Testing"]
        train_df, val_df = train_test_split(
            train_all, test_size=0.15,
            stratify=train_all["label"], random_state=cfg.seed,
        )
    else:
        train_df, tmp = train_test_split(
            df_ok, test_size=0.30,
            stratify=df_ok["label"], random_state=cfg.seed,
        )
        val_df, test_df = train_test_split(
            tmp, test_size=0.50,
            stratify=tmp["label"], random_state=cfg.seed,
        )

    classes      = sorted(df_ok["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
            classes, class_to_idx)
```

`stratify=` is non-negotiable on imbalanced medical data: without it, your val set may not even contain every class.

## Two transform pipelines, never one

Augmentations belong in training only. Even `RandomHorizontalFlip` will silently shift your reported val accuracy if you apply it at eval time.

```python
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(img_size: int):
    train_tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),     # MRI is single-channel
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
    eval_tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfm, eval_tfm
```

Why grayscale → 3-channel? Most medical scans are single-channel (MRI, ultrasound, X-ray) but `ResNet50` with ImageNet weights expects 3. Replicating across channels lets the pretrained filters apply directly.

Anatomy notes for the augmentations:
- **Vertical flip is dataset-dependent.** Fine for ultrasound, often *wrong* for MRI orientation conventions — set `p=0.2` or remove if your domain expert says no.
- **`RandomErasing`** acts as a learned cutout; small `scale` so it doesn't blot out the lesion entirely.

## A minimal Dataset

```python
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, df, class_to_idx, tfm):
        self.df          = df
        self.class_to_idx = class_to_idx
        self.tfm         = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r   = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tfm(img), self.class_to_idx[r["label"]]
```

Convert to RGB even if the source is grayscale — combined with the `Grayscale(3)` transform this handles every input shape uniformly.

## DataLoaders with sane defaults

```python
import os
from torch.utils.data import DataLoader

nw = min(4, os.cpu_count() or 1)
train_loader = DataLoader(
    ImageDataset(train_df, class_to_idx, train_tfm),
    batch_size=cfg.batch, shuffle=True,
    num_workers=nw, pin_memory=True,
)
val_loader  = DataLoader(ImageDataset(val_df,  class_to_idx, eval_tfm),
                         batch_size=cfg.batch, shuffle=False,
                         num_workers=nw, pin_memory=True)
test_loader = DataLoader(ImageDataset(test_df, class_to_idx, eval_tfm),
                         batch_size=cfg.batch, shuffle=False,
                         num_workers=nw, pin_memory=True)
```

- `num_workers=4` is the right default for one-GPU box. Higher only helps if your data is cold (S3, HDD) — on local SSD it costs RAM.
- `pin_memory=True` matters once `device.type == "cuda"`; harmless on CPU.
- `shuffle=False` for val/test so batch indices line up with the DataFrame for debugging.
