"""Minimal model definition used by both mlops_pipeline.py and serve.py."""
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_model(num_classes: int) -> nn.Module:
    m    = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512), nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return m
