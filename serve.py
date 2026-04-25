"""
FastAPI inference server for brain tumor and breast cancer classifiers.

Loads best_model.pt from artifacts_brain/ or artifacts_breast/ at startup.
Each endpoint accepts an image file upload and returns class probabilities.

Usage:
    uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health                    — liveness probe
    POST /predict/brain_tumor       — brain MRI classification (4 classes)
    POST /predict/breast_cancer     — breast ultrasound classification (3 classes)
"""

import io
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import build_model

app = FastAPI(
    title="Medical Imaging Classifier",
    description="Brain tumor MRI and breast cancer ultrasound classification API.",
    version="1.0.0",
)

# ── Model configuration per task ──────────────────────────────────────────────

TASK_CONFIG = {
    "brain_tumor": {
        "checkpoint": "artifacts_brain/best_model.pt",
        "description": "4-class brain MRI: glioma / meningioma / notumor / pituitary",
    },
    "breast_cancer": {
        "checkpoint": "artifacts_breast/best_model.pt",
        "description": "3-class breast ultrasound: benign / malignant / normal",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_cache: dict = {}


def _load_model(task: str):
    """Load and cache a checkpoint. Returns (model, classes, transform)."""
    if task in _model_cache:
        return _model_cache[task]

    cfg  = TASK_CONFIG[task]
    path = Path(cfg["checkpoint"])
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Checkpoint not found: {path}. Run: python mlops_pipeline.py --task {task}",
        )

    ckpt  = torch.load(path, map_location=DEVICE, weights_only=True)
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])

    _model_cache[task] = (model, ckpt["classes"], tfm)
    return _model_cache[task]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE.type, "tasks": list(TASK_CONFIG)}


@app.post("/predict/{task}")
async def predict(task: str, file: UploadFile):
    if task not in TASK_CONFIG:
        raise HTTPException(status_code=404,
                            detail=f"Unknown task '{task}'. Choose: {list(TASK_CONFIG)}")

    model, classes, tfm = _load_model(task)

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image file.")

    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()

    top_idx = int(torch.tensor(probs).argmax())
    return {
        "task":        task,
        "prediction":  classes[top_idx],
        "confidence":  round(probs[top_idx], 4),
        "probabilities": {c: round(p, 4) for c, p in zip(classes, probs)},
    }
