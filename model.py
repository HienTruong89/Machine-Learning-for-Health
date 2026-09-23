"""Model definition and checkpoint loading, shared by training, serving and the apps."""
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


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


def eval_transform(img_size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Grayscale -> 3 channels, resize, normalise. Used for val, test and inference."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_checkpoint(path, device="cpu"):
    """Load a best_model.pt checkpoint. Returns (model in eval mode, classes, transform)."""
    ckpt  = torch.load(path, map_location=device, weights_only=True)
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    tfm = eval_transform(ckpt["img_size"], ckpt["mean"], ckpt["std"])
    return model, ckpt["classes"], tfm
