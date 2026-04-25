# Deployment Guide: Docker, FastAPI & REST APIs
## From Zero to Production — Beginner to Application

All examples are tied to the medical imaging project in this repository.

---

## Table of Contents

1. [What is Deployment?](#1-what-is-deployment)
2. [REST APIs — The Foundation](#2-rest-apis--the-foundation)
3. [FastAPI — Building the API](#3-fastapi--building-the-api)
4. [Docker — Packaging the App](#4-docker--packaging-the-app)
5. [Putting It All Together](#5-putting-it-all-together)
6. [Learning Path](#6-learning-path)

---

## 1. What is Deployment?

Training a model produces a file (`best_model.pt`). Deployment makes that model useful to the outside world — a web app, a mobile app, another service, or a hospital system.

```
Before deployment:
  model.pt  ←  only you can use it (run Python locally)

After deployment:
  model.pt → FastAPI server → REST API → anyone can call it
                                          from any language
                                          on any device
```

**The three tools we use:**

| Tool | Role | Analogy |
|---|---|---|
| **REST API** | The contract — defines how to talk to your model | A menu at a restaurant |
| **FastAPI** | The waiter — receives requests, calls the model, returns results | The waiter who takes your order |
| **Docker** | The kitchen + building — makes everything portable and reproducible | The entire restaurant, packaged to go |

---

## 2. REST APIs — The Foundation

### What is an API?

An **API** (Application Programming Interface) is a set of rules for how two programs communicate. A **REST API** uses HTTP — the same protocol your browser uses — to send and receive data.

### HTTP Basics

Every REST request has:
- A **method** — what action to take
- A **URL** — what resource to act on
- A **body** (optional) — data to send
- A **response** — the result (usually JSON)

```
HTTP Methods:
  GET     → read/retrieve data
  POST    → send data / trigger an action
  PUT     → update existing data
  DELETE  → remove data
```

### REST API Example — No Code Yet

Imagine your brain tumor model is already deployed. Here is how a doctor's app would use it:

```
Request:
  POST  http://your-server.com/predict
  Body: { image: <MRI scan file> }

Response:
  {
    "prediction":   "glioma",
    "confidence":   0.9732,
    "probabilities": {
      "glioma":      0.9732,
      "meningioma":  0.0141,
      "notumor":     0.0098,
      "pituitary":   0.0029
    }
  }
```

The doctor's app does not need Python, PyTorch, or any ML knowledge. It just sends an image over HTTP and reads the JSON result.

### JSON — The Language of REST APIs

JSON (JavaScript Object Notation) is the standard format for REST responses.

```json
{
  "prediction": "glioma",
  "confidence": 0.9732,
  "model_version": "v2",
  "classes": ["glioma", "meningioma", "notumor", "pituitary"],
  "probabilities": {
    "glioma": 0.9732,
    "meningioma": 0.0141,
    "notumor": 0.0098,
    "pituitary": 0.0029
  }
}
```

**Rules:**
- Keys are always strings in double quotes
- Values can be: string, number, boolean, array `[]`, object `{}`, or null
- No trailing commas

### Testing REST APIs with curl

`curl` is a command-line tool to send HTTP requests. You will use it constantly.

```bash
# GET request
curl http://localhost:8000/health

# POST with a file upload
curl -X POST http://localhost:8000/predict \
     -F "file=@data/Testing/glioma/Te-gl_0010.jpg"

# POST with JSON body
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"patient_id": "P001", "scan_type": "MRI"}'
```

---

## 3. FastAPI — Building the API

### What is FastAPI?

FastAPI is a modern Python web framework for building REST APIs. It is:
- **Fast to write** — less boilerplate than Flask or Django
- **Fast to run** — built on async Python (one of the fastest Python frameworks)
- **Self-documenting** — generates interactive docs at `/docs` automatically

```bash
pip install fastapi uvicorn python-multipart
```

`uvicorn` is the server that runs FastAPI applications.

---

### Example 1 — Hello World API

The simplest possible API. Run this first to understand the basics.

**`hello_api.py`:**
```python
from fastapi import FastAPI

app = FastAPI(title="Hello API")

@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"Hello, {name}!"}
```

**Run it:**
```bash
uvicorn hello_api:app --reload
```

**Test it:**
```bash
curl http://localhost:8000/
# {"message":"Hello, World!"}

curl http://localhost:8000/greet/Alice
# {"message":"Hello, Alice!"}

# Or open http://localhost:8000/docs in your browser
# FastAPI generates a full interactive UI automatically
```

---

### Example 2 — Brain Tumor Prediction API

A real API that loads the trained model and serves predictions.

**`serve_brain_tumor.py`:**
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import io
import json
from pathlib import Path


# ── Model setup ──────────────────────────────────────────────────────────── #

def build_model(num_classes: int):
    m = resnet50(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512), nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return m


# ── Startup: load model once when the server starts ──────────────────────── #

model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt_path = Path("artifacts/best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    model_state["model"]   = model
    model_state["classes"] = ckpt["classes"]
    model_state["tfm"] = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])
    print(f"Model loaded. Classes: {ckpt['classes']}")
    yield  # server runs here
    model_state.clear()


# ── App ───────────────────────────────────────────────────────────────────── #

app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description="Classifies MRI scans into: glioma, meningioma, notumor, pituitary.",
    version="2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Check if the server and model are ready."""
    return {
        "status": "ok",
        "model_loaded": "model" in model_state,
        "classes": model_state.get("classes", []),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an MRI image and receive a tumor classification.

    - **file**: JPG or PNG brain MRI scan
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    x = model_state["tfm"](img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model_state["model"](x), dim=1)[0]

    classes = model_state["classes"]
    top_idx = probs.argmax().item()

    return {
        "prediction":    classes[top_idx],
        "confidence":    round(probs[top_idx].item(), 4),
        "probabilities": {c: round(p.item(), 4) for c, p in zip(classes, probs)},
        "filename":      file.filename,
    }


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Upload multiple MRI images and get predictions for all of them."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch.")

    results = []
    for file in files:
        contents = await file.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            x = model_state["tfm"](img).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model_state["model"](x), dim=1)[0]
            classes = model_state["classes"]
            top_idx = probs.argmax().item()
            results.append({
                "filename":   file.filename,
                "prediction": classes[top_idx],
                "confidence": round(probs[top_idx].item(), 4),
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"results": results, "total": len(results)}
```

**Run it:**
```bash
uvicorn serve_brain_tumor:app --reload --port 8000
```

**Test single prediction:**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@data/Testing/glioma/Te-gl_0010.jpg"
```

**Response:**
```json
{
  "prediction": "glioma",
  "confidence": 0.9732,
  "probabilities": {
    "glioma": 0.9732,
    "meningioma": 0.0141,
    "notumor": 0.0098,
    "pituitary": 0.0029
  },
  "filename": "Te-gl_0010.jpg"
}
```

**Test batch prediction:**
```bash
curl -X POST http://localhost:8000/predict/batch \
     -F "files=@data/Testing/glioma/Te-gl_0010.jpg" \
     -F "files=@data/Testing/notumor/Te-no_0001.jpg"
```

**Interactive docs** — open `http://localhost:8000/docs` to test every endpoint in your browser.

---

### Example 3 — Breast Cancer API

The same pattern applied to the breast cancer model.

**`serve_breast_cancer.py`:**
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import io
from pathlib import Path


def build_model(num_classes):
    m = resnet50(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512), nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return m


model_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = torch.load("artifacts_breast/best_model.pt", map_location="cpu")
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model_state["model"]   = model
    model_state["classes"] = ckpt["classes"]
    model_state["tfm"] = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])
    yield
    model_state.clear()


app = FastAPI(
    title="Breast Cancer Ultrasound Classifier",
    description="Classifies ultrasound images into: benign, malignant, normal.",
    version="2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "classes": model_state.get("classes", [])}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload a breast ultrasound image and receive a classification."""
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    x = model_state["tfm"](img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model_state["model"](x), dim=1)[0]

    classes = model_state["classes"]
    top_idx = probs.argmax().item()

    return {
        "prediction":    classes[top_idx],
        "confidence":    round(probs[top_idx].item(), 4),
        "probabilities": {c: round(p.item(), 4) for c, p in zip(classes, probs)},
        "risk_flag":     classes[top_idx] == "malignant",
    }
```

**Unique addition:** `risk_flag` — a boolean that is `true` when the model predicts malignant, making it easy for downstream apps to trigger an alert.

---

### Example 4 — Combined API (Both Models)

Serve both models from one server on different routes.

**`serve_combined.py`:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import both routers (split each model into its own file)
# from serve_brain_tumor import app as brain_app   ← for larger projects
# Here we mount them as sub-applications

from serve_brain_tumor  import app as brain_app
from serve_breast_cancer import app as breast_app

app = FastAPI(title="Medical Imaging API — Combined")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/brain",  brain_app)
app.mount("/breast", breast_app)

@app.get("/")
def root():
    return {
        "services": {
            "brain_tumor":    "/brain/predict",
            "breast_cancer":  "/breast/predict",
            "brain_docs":     "/brain/docs",
            "breast_docs":    "/breast/docs",
        }
    }
```

```bash
uvicorn serve_combined:app --port 8000

# Brain tumor prediction
curl -X POST http://localhost:8000/brain/predict \
     -F "file=@data/Testing/glioma/Te-gl_0010.jpg"

# Breast cancer prediction
curl -X POST http://localhost:8000/breast/predict \
     -F "file=@data_breast/malignant/malignant (1).png"
```

---

### FastAPI Features You Will Use Daily

**Path parameters** — part of the URL:
```python
@app.get("/models/{model_name}/info")
def model_info(model_name: str):
    return {"model": model_name}
```

**Query parameters** — after the `?`:
```python
@app.get("/predict")
def predict(threshold: float = 0.5, top_k: int = 3):
    # called as: /predict?threshold=0.8&top_k=2
    ...
```

**Request body** — structured input using Pydantic:
```python
from pydantic import BaseModel

class PredictRequest(BaseModel):
    patient_id: str
    scan_date:  str
    notes:      str = ""   # optional field with default

@app.post("/predict/structured")
def predict_structured(req: PredictRequest):
    return {"patient_id": req.patient_id, "status": "queued"}
```

**Error responses:**
```python
from fastapi import HTTPException

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.size > 10 * 1024 * 1024:   # 10 MB limit
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")
```

**Common HTTP status codes:**

| Code | Meaning | When to use |
|---|---|---|
| 200 | OK | Successful request |
| 201 | Created | Resource was created |
| 400 | Bad Request | Client sent invalid data |
| 401 | Unauthorized | Missing/invalid credentials |
| 404 | Not Found | Resource does not exist |
| 413 | Payload Too Large | File too big |
| 422 | Unprocessable Entity | Validation failed (FastAPI default) |
| 500 | Internal Server Error | Something crashed on the server |

---

## 4. Docker — Packaging the App

### What is Docker?

Docker packages your application, its Python version, and all dependencies into a single portable unit called a **container**. It runs identically on any machine that has Docker installed.

```
Without Docker:
  "Works on my machine" → fails on the server
  (different Python version, missing package, OS difference)

With Docker:
  Same image → same result on every machine, every time
```

### Key Concepts

| Term | What it is | Analogy |
|---|---|---|
| **Image** | A blueprint — a read-only snapshot | A recipe |
| **Container** | A running instance of an image | The actual dish made from the recipe |
| **Dockerfile** | Instructions to build an image | The recipe card itself |
| **Registry** | A place to store and share images | A recipe book library (e.g. Docker Hub) |

---

### Example 1 — Dockerfile for the Brain Tumor API

Create this file at the root of the project:

**`Dockerfile`:**
```dockerfile
# ── Stage 1: base image ───────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# ── Stage 2: install dependencies ────────────────────────────────────────
# Copy requirements first — Docker caches this layer.
# If only your code changes, pip install won't re-run.
COPY requirements_serve.txt .
RUN pip install --no-cache-dir -r requirements_serve.txt

# ── Stage 3: copy application code ───────────────────────────────────────
COPY serve_brain_tumor.py .

# ── Stage 4: copy the trained model ──────────────────────────────────────
COPY artifacts/best_model.pt artifacts/best_model.pt

# ── Stage 5: expose port and set startup command ─────────────────────────
EXPOSE 8000

CMD ["uvicorn", "serve_brain_tumor:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`requirements_serve.txt`:**
```
torch==2.3.0
torchvision==0.18.0
fastapi==0.111.0
uvicorn==0.30.0
python-multipart==0.0.9
Pillow==10.3.0
```

**Build the image:**
```bash
docker build -t brain-tumor-api:v2 .
```

**Run a container from it:**
```bash
docker run -p 8000:8000 brain-tumor-api:v2
```

**Test it (same as before):**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@data/Testing/glioma/Te-gl_0010.jpg"
```

---

### Example 2 — Dockerfile for Both Models

**`Dockerfile.combined`:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_serve.txt .
RUN pip install --no-cache-dir -r requirements_serve.txt

# Copy both API scripts
COPY serve_brain_tumor.py .
COPY serve_breast_cancer.py .
COPY serve_combined.py .

# Copy both model checkpoints
COPY artifacts/best_model.pt           artifacts/best_model.pt
COPY artifacts_breast/best_model.pt    artifacts_breast/best_model.pt

EXPOSE 8000

CMD ["uvicorn", "serve_combined:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -f Dockerfile.combined -t medical-imaging-api:v2 .
docker run -p 8000:8000 medical-imaging-api:v2
```

---

### Example 3 — Docker Compose (Run Multiple Services Together)

`docker-compose.yml` lets you define and run multiple containers together — useful when you want to run an API server alongside a database or a monitoring service.

**`docker-compose.yml`:**
```yaml
version: "3.9"

services:

  brain-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8001:8000"
    volumes:
      - ./artifacts:/app/artifacts   # mount model — no rebuild needed to swap model
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  breast-api:
    build:
      context: .
      dockerfile: Dockerfile.breast
    ports:
      - "8002:8000"
    volumes:
      - ./artifacts_breast:/app/artifacts_breast
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - brain-api
      - breast-api
```

**Run everything with one command:**
```bash
docker compose up --build

# Brain tumor API  → http://localhost:8001
# Breast cancer API → http://localhost:8002
# Nginx (load balancer) → http://localhost:80
```

**Stop everything:**
```bash
docker compose down
```

---

### Essential Docker Commands

```bash
# Build
docker build -t myapp:v1 .                  # build image from Dockerfile
docker build -f Dockerfile.prod -t myapp .  # use a specific Dockerfile

# Run
docker run -p 8000:8000 myapp:v1            # run container, map port
docker run -d -p 8000:8000 myapp:v1         # run in background (detached)
docker run -v ./artifacts:/app/artifacts myapp:v1  # mount local folder

# Inspect
docker ps                                   # list running containers
docker ps -a                                # list all containers (including stopped)
docker logs <container_id>                  # view container output
docker exec -it <container_id> bash         # open a shell inside a running container

# Clean up
docker stop <container_id>                  # stop a running container
docker rm <container_id>                    # delete a stopped container
docker rmi myapp:v1                         # delete an image
docker system prune                         # remove all stopped containers + unused images
```

---

### Dockerfile Best Practices

```dockerfile
# 1. Use a specific version tag — never "latest" in production
FROM python:3.11.9-slim          # good
FROM python:latest               # bad — breaks when new version releases

# 2. Copy requirements before code — Docker caches pip install
COPY requirements.txt .          # cached unless requirements change
RUN pip install -r requirements.txt
COPY . .                         # code changes don't re-trigger pip install

# 3. Use .dockerignore to keep images small
# .dockerignore:
#   data/
#   __pycache__/
#   *.pyc
#   .git/
#   .env

# 4. Run as non-root for security
RUN useradd -m appuser
USER appuser

# 5. Set PYTHONDONTWRITEBYTECODE to keep image clean
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
```

---

## 5. Putting It All Together

### Full Deployment Stack Diagram

```
Developer's laptop
       │
       │  python train_brain_tumor_v2.py
       │  → artifacts/best_model.pt
       │
       ▼
  Docker build
  ─────────────────────────────────────
  │  python:3.11-slim                 │
  │  + torch, fastapi, uvicorn        │
  │  + serve_brain_tumor.py           │
  │  + artifacts/best_model.pt        │
  ─────────────────────────────────────
       │
       │  docker run -p 8000:8000
       ▼
  Running Container
  ┌─────────────────────────────────┐
  │  FastAPI Server (uvicorn)       │
  │  ├── GET  /health               │
  │  ├── POST /predict              │
  │  └── POST /predict/batch        │
  └─────────────────────────────────┘
       │
       │  HTTP/REST
       ▼
  Clients
  ├── curl (terminal testing)
  ├── Web app (JavaScript fetch)
  ├── Mobile app (Swift / Kotlin)
  └── Hospital system (any language)
```

### End-to-End Workflow

**Step 1 — Train the model:**
```bash
python train_brain_tumor_v2.py --epochs 20
# produces: artifacts/best_model.pt
```

**Step 2 — Test the API locally (no Docker):**
```bash
uvicorn serve_brain_tumor:app --reload --port 8000
curl -X POST http://localhost:8000/predict \
     -F "file=@data/Testing/glioma/Te-gl_0010.jpg"
```

**Step 3 — Build the Docker image:**
```bash
docker build -t brain-tumor-api:v2 .
```

**Step 4 — Test the containerized API:**
```bash
docker run -p 8000:8000 brain-tumor-api:v2
curl -X POST http://localhost:8000/predict \
     -F "file=@data/Testing/glioma/Te-gl_0010.jpg"
# Same result — the container is working
```

**Step 5 — Push to a registry (to deploy on a server):**
```bash
# Docker Hub
docker tag brain-tumor-api:v2 yourusername/brain-tumor-api:v2
docker push yourusername/brain-tumor-api:v2

# Pull and run on any server
docker pull yourusername/brain-tumor-api:v2
docker run -d -p 8000:8000 yourusername/brain-tumor-api:v2
```

### Calling the API from Python (Client Side)

Once deployed, any Python script can call it without importing PyTorch:

```python
import requests

url = "http://localhost:8000/predict"

with open("data/Testing/glioma/Te-gl_0010.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Calling the API from JavaScript (Web App)

```javascript
async function predictTumor(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    body: formData,
  });

  const result = await response.json();
  console.log(`Prediction: ${result.prediction} (${(result.confidence * 100).toFixed(1)}%)`);
  return result;
}

// Usage (in a browser with a file input)
const input = document.getElementById("imageInput");
input.addEventListener("change", () => predictTumor(input.files[0]));
```

---

## 6. Learning Path

Work through these stages in order. Each stage should take 2–4 hours.

### Stage 1 — REST API Fundamentals (Day 1)
- [ ] Install `httpie` or use `curl`, send GET/POST requests to a public API
- [ ] Read a JSON response and understand its structure
- [ ] Understand HTTP status codes 200, 400, 404, 500

```bash
# Practice with a free public API
curl https://api.github.com/users/octocat
curl https://httpbin.org/post -X POST -d "hello=world"
```

### Stage 2 — First FastAPI App (Day 1–2)
- [ ] Install FastAPI and uvicorn
- [ ] Run the Hello World example above
- [ ] Open `/docs` and test endpoints in the browser
- [ ] Add a `/predict` route that returns a hardcoded JSON response

### Stage 3 — Real ML API (Day 2–3)
- [ ] Copy `serve_brain_tumor.py` from this guide
- [ ] Run it against your trained `artifacts/best_model.pt`
- [ ] Test with `curl` using a real MRI image
- [ ] Add the `/predict/batch` endpoint

### Stage 4 — First Dockerfile (Day 3–4)
- [ ] Install Docker Desktop
- [ ] Write the `Dockerfile` from this guide
- [ ] Build the image and run it
- [ ] Verify the API gives the same response inside Docker as it did outside

### Stage 5 — Docker Compose (Day 4–5)
- [ ] Write `docker-compose.yml` to run both models
- [ ] Add a health check
- [ ] Use a volume mount so you can swap models without rebuilding

### Stage 6 — Deploy to a Cloud VM (Day 5–7)
- [ ] Create a free VM on Azure (B1s), AWS EC2 (t2.micro), or Google Cloud (e2-micro)
- [ ] Install Docker on the VM
- [ ] Push your image to Docker Hub, pull it on the VM, and run it
- [ ] Call your API from your laptop using the VM's public IP

```bash
# On the VM
docker pull yourusername/brain-tumor-api:v2
docker run -d -p 8000:8000 yourusername/brain-tumor-api:v2

# From your laptop
curl http://<VM_PUBLIC_IP>:8000/health
```

---

## Quick Reference Card

```
REST API
  GET  /health           → check if server is alive
  POST /predict          → send image, get prediction
  POST /predict/batch    → send multiple images

FastAPI
  uvicorn app:app --reload          → run dev server
  localhost:8000/docs               → interactive API docs
  @app.get("/route")                → handle GET
  @app.post("/route")               → handle POST
  raise HTTPException(400, detail)  → return error

Docker
  docker build -t name:tag .        → build image
  docker run -p 8000:8000 name:tag  → run container
  docker ps                         → list containers
  docker logs <id>                  → view output
  docker compose up --build         → run all services
  docker compose down               → stop all services
```

---

## Recommended Resources

- **FastAPI official docs** — [fastapi.tiangolo.com](https://fastapi.tiangolo.com) — best framework docs available, read the tutorial start to finish
- **Docker getting started** — [docs.docker.com/get-started](https://docs.docker.com/get-started)
- **Play with Docker** — [labs.play-with-docker.com](https://labs.play-with-docker.com) — free in-browser Docker playground, no install needed
- **HTTP Cats** — [http.cat](https://http.cat) — learn HTTP status codes in a memorable way
- **Postman** — GUI tool for testing REST APIs, easier than curl for beginners
