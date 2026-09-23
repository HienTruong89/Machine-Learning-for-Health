# Stage 10 — Inference API (FastAPI + Docker)

**Goal:** wrap the trained checkpoint in a tiny HTTP service that any client (web, mobile, batch script) can hit.

## Single endpoint per task, dynamic dispatch

Don't write one server per task — pass `task` in the path and look up the checkpoint at request time. Adding a new dataset is a one-line change to `TASK_CONFIG`.

```python
# serve.py
import io
from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import build_model

app = FastAPI(title="Medical Imaging Classifier", version="1.0.0")

TASK_CONFIG = {
    "brain_tumor":   {"checkpoint": "artifacts_brain/best_model.pt"},
    "breast_cancer": {"checkpoint": "artifacts_breast/best_model.pt"},
    # add new tasks here
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_cache: dict = {}
```

## Lazy-load + cache

Loading ResNet50 takes ~1s. Cache per task; the first request pays the cost, the rest are free.

```python
def _load_model(task: str):
    if task in _cache:
        return _cache[task]
    cfg  = TASK_CONFIG[task]
    path = Path(cfg["checkpoint"])
    if not path.exists():
        raise HTTPException(503, f"Checkpoint not found: {path}")
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
    _cache[task] = (model, ckpt["classes"], tfm)
    return _cache[task]
```

The `mean`, `std`, `img_size` come from the checkpoint — that's why we bundled them in Stage 6. The serving code has zero hardcoded preprocessing constants.

## Two endpoints: health + predict

```python
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE.type, "tasks": list(TASK_CONFIG)}

@app.post("/predict/{task}")
async def predict(task: str, file: UploadFile):
    if task not in TASK_CONFIG:
        raise HTTPException(404, f"Unknown task '{task}'")
    model, classes, tfm = _load_model(task)
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(422, "Could not decode image file.")
    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    top = int(torch.tensor(probs).argmax())
    return {
        "task":          task,
        "prediction":    classes[top],
        "confidence":    round(probs[top], 4),
        "probabilities": {c: round(p, 4) for c, p in zip(classes, probs)},
    }
```

`/health` is mandatory — Docker, Kubernetes, and Azure App Service all need a liveness probe. Make it cheap (no model loading).

## Dockerfile: training-free serving image

The serving image must NOT install MLflow, scikit-learn, or kagglehub. They're hundreds of megabytes you don't need at inference time.

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        Pillow fastapi "uvicorn[standard]" python-multipart

COPY model.py serve.py ./
COPY artifacts_brain/best_model.pt   artifacts_brain/best_model.pt
COPY artifacts_breast/best_model.pt  artifacts_breast/best_model.pt

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

Notes:
- `--index-url cpu` keeps the image at ~1 GB instead of ~5 GB CUDA build. If you actually need GPU inference, use `pytorch/pytorch:<tag>-cuda` as the base.
- COPY each checkpoint explicitly. `COPY artifacts_*/` would silently miss new task dirs.
- `HEALTHCHECK` uses stdlib `urllib`, no extra deps.

## Run + smoke test

```bash
docker build -t medical-imaging-api:v1 .
docker run -p 8000:8000 medical-imaging-api:v1

# Health
curl http://localhost:8000/health

# Predict
curl -F "file=@scan.jpg" http://localhost:8000/predict/brain_tumor
```

## Batch client

For backfilling a folder of scans, hit the API in a loop. Keeps the server stateless and lets you parallelize across machines later.

```python
# batch_predict.py
import requests
from pathlib import Path
from tqdm import tqdm

SERVER = "http://localhost:8000"

def classify(image_path: Path, task: str) -> dict:
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{SERVER}/predict/{task}",
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )
    r.raise_for_status()
    return r.json()
```

## Adding auth

The current API has no authentication — fine for localhost or a private VPC. For public deployment, add a header check:

```python
from fastapi import Header

API_KEY = os.environ["API_KEY"]

async def require_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")

@app.post("/predict/{task}", dependencies=[Depends(require_key)])
async def predict(...):
    ...
```

Keep the API key in `docker run -e API_KEY=...`, never bake it into the image.
