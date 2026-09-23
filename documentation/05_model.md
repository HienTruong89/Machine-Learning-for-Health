# Stage 5 — Model architecture

**Goal:** start from a pretrained backbone, swap the classifier head, keep the definition tiny so training and serving share it.

## Why ResNet50 + ImageNet V2

Medical imaging datasets are small (low thousands of images). Training a CNN from scratch overfits in <5 epochs. Transfer learning from ImageNet works because the early conv filters (edges, textures, contrast) generalize to any natural-image-like domain — including grayscale scans replicated to 3 channels.

`ResNet50_Weights.IMAGENET1K_V2` is a recipe-improved checkpoint that beats V1 by ~1-2% on ImageNet at zero cost to you. Use V2 unless you have a specific reason not to.

## The whole model in 10 lines

Keep this in its own module (`model.py`) so the pipeline, the API server, and the Streamlit app all import the *same* function. If they drift, your inference will silently use a different graph than training.

```python
# model.py
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

def build_model(num_classes: int) -> nn.Module:
    m    = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_f = m.fc.in_features                    # 2048 for ResNet50
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512), nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return m
```

A 2-layer MLP head is the right default:
- **Single linear layer** under-fits when classes are visually similar (glioma vs. meningioma).
- **3+ layers** start overfitting on small datasets.
- **Dropout 0.3** twice — enough regularization without killing capacity.

## What's *not* here

- **No backbone freezing.** Full fine-tuning works better than freezing for medical imaging because the late-stage filters (object parts) need to adapt — ImageNet doesn't know what a tumor looks like. The cost is more memory; mitigated by AMP (Stage 6).
- **No model-specific config object.** `num_classes` is the only thing that varies; it comes from `len(classes)` after Stage 4.
- **No pretrained-weights download flag.** `weights=ResNet50_Weights.IMAGENET1K_V2` always uses the cached download; if the host has no internet, pre-bake the weights into the Docker image with `torch.hub.set_dir`.

## Loading a saved model for inference

The training stage saves `state_dict + classes + img_size + mean + std` in one file. Inference is symmetric:

```python
import torch
from model import build_model

ckpt  = torch.load("artifacts_brain/best_model.pt",
                   map_location="cpu", weights_only=True)
model = build_model(len(ckpt["classes"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

`weights_only=True` is important — it refuses to unpickle arbitrary Python objects, which is the right default for any checkpoint that came from outside your machine.

## Swapping the backbone

If you need a smaller / faster model later:

```python
# Drop-in replacements with the same interface
from torchvision.models import (
    resnet18, ResNet18_Weights,                   # ~10× fewer params
    efficientnet_v2_s, EfficientNet_V2_S_Weights, # better accuracy/param trade
    convnext_tiny, ConvNeXt_Tiny_Weights,         # modern arch
)
```

For each, find the classifier attribute (`m.fc` for ResNet, `m.classifier[-1]` for EfficientNet/ConvNeXt) and replace it with the same MLP head. Keep the `build_model(num_classes)` signature so nothing downstream changes.
