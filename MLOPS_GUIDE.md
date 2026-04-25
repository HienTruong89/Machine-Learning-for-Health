# MLOps Fundamentals — Beginner's Guide

A practical introduction to MLOps using the medical imaging project in this repository as a running example.

---

## What is MLOps?

**MLOps** (Machine Learning Operations) is the practice of deploying, monitoring, and maintaining ML models in production reliably and efficiently. Think of it as DevOps, but with extra challenges unique to ML:

- Models degrade silently when real-world data shifts
- Experiments need to be reproducible
- Data and model versions must be tracked together
- Retraining pipelines must be automated

**Without MLOps** → you train a model locally, copy the file to a server, and hope it keeps working.  
**With MLOps** → you have automated pipelines, version control for data + models, and alerts when things go wrong.

---

## The MLOps Lifecycle

```
    ┌─────────────┐
    │   Data      │  ← collect, validate, version
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Training   │  ← experiment tracking, hyperparameter tuning
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Evaluation │  ← metrics, comparison with previous model
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Deployment │  ← containerize, serve via API
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Monitoring │  ← data drift, model drift, alerts
    └──────┬──────┘
           │
           └──── triggers retraining ────┐
                                         │
                               (back to Training)
```

---

## Core Concepts

### 1. Experiment Tracking

Track every training run so you can compare results and reproduce the best one.

**Key questions it answers:**
- Which hyperparameters gave the best accuracy?
- What data version was used?
- How did model v3 compare to model v2?

**Tool: MLflow**

```bash
pip install mlflow
```

**Example** — adding MLflow to `train_brain_tumor_v2.py`:

```python
import mlflow
import mlflow.pytorch

def main():
    # ... existing argument parsing ...

    mlflow.set_experiment("brain-tumor-classification")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "epochs":   args.epochs,
            "batch":    args.batch,
            "lr":       args.lr,
            "img_size": args.img_size,
        })

        # ... training loop ...

        # Log metrics each epoch
        for row in history:
            mlflow.log_metrics({
                "train_loss": row["train_loss"],
                "val_loss":   row["val_loss"],
                "train_acc":  row["train_acc"],
                "val_acc":    row["val_acc"],
            }, step=row["epoch"])

        # Log the best model
        mlflow.log_metric("best_val_acc", best_val)
        mlflow.log_metric("test_macro_auroc", auc)
        mlflow.pytorch.log_model(model, "model")
```

**View the UI:**
```bash
mlflow ui
# Open http://localhost:5000
```

You will see a dashboard comparing every run — hyperparameters, metrics, and saved models side by side.

---

### 2. Data Versioning

Code versioning (Git) tracks your scripts. Data versioning tracks your datasets so you can reproduce results exactly.

**Tool: DVC (Data Version Control)**

```bash
pip install dvc
dvc init
```

**Example** — version the data folder:

```bash
# Tell DVC to track the data folder (not Git)
dvc add data/

# Push data to remote storage (e.g. Google Drive, S3, Azure Blob)
dvc remote add -d myremote gdrive://YOUR_FOLDER_ID
dvc push

# Git tracks only the small .dvc pointer file
git add data.dvc .gitignore
git commit -m "Add dataset v1"
```

**To restore data on another machine:**
```bash
git pull          # gets the .dvc pointer
dvc pull          # downloads the actual data
```

Now your data and code versions are linked. Anyone who checks out a specific Git commit gets the exact matching dataset.

---

### 3. Model Registry

A central store that tracks model versions, their status (staging / production / archived), and associated metadata.

**Tool: MLflow Model Registry** (built into MLflow)

```python
# After a training run, register the model
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="brain-tumor-classifier"
)
```

**Promote to production via UI or code:**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="brain-tumor-classifier",
    version=3,
    stage="Production"
)
```

**Load the production model anywhere:**
```python
model = mlflow.pytorch.load_model(
    "models:/brain-tumor-classifier/Production"
)
```

---

### 4. Containerization

Package your model and its dependencies into a container so it runs identically everywhere (your laptop, a server, the cloud).

**Tool: Docker**

**Example** — `Dockerfile` for serving the brain tumor model:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY artifacts/best_model.pt .
COPY predict_app.py .

EXPOSE 8000

CMD ["python", "predict_app.py"]
```

**`requirements.txt`:**
```
torch==2.3.0
torchvision==0.18.0
Pillow==10.3.0
fastapi==0.111.0
uvicorn==0.30.0
```

**Build and run:**
```bash
docker build -t brain-tumor-api .
docker run -p 8000:8000 brain-tumor-api
```

The model is now accessible as an HTTP API, isolated from your local environment.

---

### 5. CI/CD for ML

Automatically test, validate, and deploy your model whenever code or data changes.

**Tool: GitHub Actions**

**Example** — `.github/workflows/train_and_validate.yml`:

```yaml
name: Train and Validate Model

on:
  push:
    branches: [master]
    paths:
      - "train_brain_tumor_v2.py"

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install torch torchvision kagglehub scikit-learn pandas tqdm mlflow

      - name: Download data and train (smoke test — 1 epoch)
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
        run: python train_brain_tumor_v2.py --epochs 1 --batch 16

      - name: Validate test accuracy threshold
        run: |
          python - <<'EOF'
          import json
          report = json.load(open("artifacts/test_report.json"))
          acc = report["classification_report"]["accuracy"]
          assert acc > 0.50, f"Accuracy {acc:.2%} below threshold — rejecting model"
          print(f"Accuracy check passed: {acc:.2%}")
          EOF

      - name: Upload model artifact
        uses: actions/upload-artifact@v4
        with:
          name: best_model
          path: artifacts/best_model.pt
```

Every push triggers a training run. If accuracy drops below the threshold, the pipeline fails and the model is not deployed.

---

### 6. Model Serving

Expose your model as an API so applications can send images and receive predictions.

**Tool: FastAPI**

**Example** — `serve.py`:

```python
from fastapi import FastAPI, UploadFile
from PIL import Image
import torch
from torchvision import transforms
import io

app = FastAPI()

# Load model at startup
ckpt = torch.load("artifacts/best_model.pt", map_location="cpu")
model = ...  # build_model(len(ckpt["classes"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()

tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(ckpt["mean"], ckpt["std"]),
])

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    classes = ckpt["classes"]
    return {
        "prediction": classes[probs.argmax().item()],
        "confidence": round(probs.max().item(), 4),
        "probabilities": {c: round(p.item(), 4) for c, p in zip(classes, probs)},
    }
```

**Run and call:**
```bash
uvicorn serve:app --reload

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
  }
}
```

---

### 7. Monitoring

Detect when your model starts performing poorly in production — before users notice.

**Two types of drift to watch:**

| Type | What it means | Example |
|---|---|---|
| **Data drift** | Input distribution has changed | Hospital switches to a different MRI scanner |
| **Model drift** (concept drift) | Relationship between inputs and labels has changed | New tumor subtypes appear |

**Tool: Evidently**

```bash
pip install evidently
```

**Example** — compare training data against recent production data:

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

train_data = pd.read_csv("artifacts/train_features.csv")
prod_data  = pd.read_csv("artifacts/production_features.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_data, current_data=prod_data)
report.save_html("drift_report.html")
```

Open `drift_report.html` to see which features have drifted and by how much. If drift is detected, trigger a retraining pipeline automatically.

---

## Putting It All Together

Here is how the full MLOps stack looks for this project:

```
Developer pushes code
        │
        ▼
GitHub Actions (CI/CD)
  ├── install deps
  ├── download data via kagglehub
  ├── train model (1-epoch smoke test)
  ├── validate accuracy threshold
  └── push passing model to MLflow Registry
              │
              ▼
        MLflow Registry
          (staging → production promotion)
              │
              ▼
        Docker container
          (FastAPI serving /predict endpoint)
              │
              ▼
        Production traffic
              │
              ▼
        Evidently monitoring
          (data drift alerts → triggers retraining)
```

---

## Beginner Learning Path

Follow these steps in order — each builds on the previous:

| Step | What to do | Time estimate |
|---|---|---|
| 1 | Add MLflow tracking to `train_brain_tumor_v2.py` and compare 3 runs | 1–2 hours |
| 2 | Set up DVC and version the `data/` folder | 1 hour |
| 3 | Write a `Dockerfile` and run the model as a local API | 2–3 hours |
| 4 | Add a GitHub Actions workflow that runs a 1-epoch smoke test | 1–2 hours |
| 5 | Register the model in MLflow Model Registry and load it by stage name | 1 hour |
| 6 | Add Evidently drift monitoring on a sample of "production" images | 2 hours |

---

## Recommended Free Resources

- **MLflow docs** — [mlflow.org/docs/latest](https://mlflow.org/docs/latest)
- **DVC docs** — [dvc.org/doc/start](https://dvc.org/doc/start)
- **Docker getting started** — [docs.docker.com/get-started](https://docs.docker.com/get-started)
- **GitHub Actions** — [docs.github.com/actions](https://docs.github.com/en/actions)
- **Made With ML** (free MLOps course) — [madewithml.com](https://madewithml.com)
- **Full Stack Deep Learning** (free) — [fullstackdeeplearning.com](https://fullstackdeeplearning.com)
