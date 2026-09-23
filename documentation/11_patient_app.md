# Stage 11 — Patient-facing UI (Streamlit)

**Goal:** a single-file web app that loads checkpoints from Hugging Face Hub on first run, serves clinical-style results, and is deployable to Streamlit Cloud with zero infrastructure.

## Why Streamlit (and not the FastAPI server alone)

The FastAPI server (Stage 10) is for *machine* clients — batch jobs, mobile apps, downstream pipelines. Patients and clinicians want a web page. Streamlit gives you the page in ~200 lines, with file upload, charts, and reactive state. For the next project, keep both: the API for integration, the Streamlit app for demo / clinic-side UX.

## Lazy model download from Hugging Face Hub

Don't bake checkpoints into the Streamlit Cloud repo. Streamlit Cloud has a 1 GB repo limit and your checkpoints are large — host them on Hugging Face Hub and pull on first run.

```python
from pathlib import Path
import streamlit as st

HF_REPO = "<your-username>/medical-imaging-models"   # one HF repo, many models

def ensure_model(artifacts: Path) -> bool:
    model_path = artifacts / "best_model.pt"
    if model_path.exists():
        return True
    try:
        from huggingface_hub import hf_hub_download
        with st.spinner("Downloading model from Hugging Face Hub..."):
            hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{artifacts.name}/best_model.pt",
                local_dir=".",
            )
        return model_path.exists()
    except Exception as e:
        st.error(f"Could not download model: {e}")
        return False
```

The HF repo layout mirrors the local one:

```
<your-repo>/
├── artifacts_brain/best_model.pt
├── artifacts_breast/best_model.pt
└── artifacts_<new_task>/best_model.pt   # just upload here for a new dataset
```

Upload via:

```bash
huggingface-cli upload <user>/medical-imaging-models  artifacts_brain/best_model.pt  artifacts_brain/best_model.pt
```

Public models don't need an HF token; private ones do (`huggingface-cli login`).

## Cache the model with `@st.cache_resource`

Without caching, the model is reloaded on every interaction (every slider drag, every upload). `@st.cache_resource` keeps the model in memory across reruns of the script.

```python
import torch
from torchvision import transforms
from model import build_model

@st.cache_resource
def load_model(artifacts: Path):
    ckpt_path = artifacts / "best_model.pt"
    if not ckpt_path.exists():
        return None, None, None
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])
    return model, ckpt["classes"], tfm
```

## Task config dict — the only place to add a new dataset

When you add a new dataset, only this dict needs editing:

```python
TASKS = {
    "Brain Tumor MRI": {
        "artifacts":       Path("artifacts_brain"),
        "concern_classes": {"glioma", "meningioma", "pituitary"},
        "safe_class":      "notumor",
        "description":     "Classifies brain MRI scans into: Glioma, Meningioma, No Tumor, Pituitary",
        "upload_label":    "Upload Brain MRI scan (JPG or PNG)",
        "color":           "#4C78A8",
    },
    # add new entries here, mirroring the structure
}

FRIENDLY_NAMES = {                       # raw class id → display label
    "glioma":     "Glioma",
    "notumor":    "No Tumor Detected",
    # ...
}

RISK_FLAGS = {                           # raw class id → (banner text, colour, css class)
    "glioma":     ("HIGH CONCERN",     "#dc3545", "result-positive"),
    "notumor":    ("NO FINDING",       "#28a745", "result-negative"),
    # ...
}
```

## The full inference flow

```python
import numpy as np
from PIL import Image

uploaded = st.file_uploader(cfg["upload_label"], type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an image to get a classification result.")
    st.stop()

img   = Image.open(uploaded).convert("RGB")
x     = tfm(img).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0].tolist()

top         = int(np.argmax(probs))
prediction  = classes[top]
confidence  = probs[top]
risk_label, risk_color, css = RISK_FLAGS[prediction]
```

## Clinical UX rules

- **Always show a friendly name and a confidence percentage**, never the raw class id.
- **Always show all class probabilities**, not just the top one — clinicians want to see the runner-up.
- **Always show a disclaimer.** Make it clear the tool is research-only.
- **Use colour coding consistently.** Green = no finding, yellow/orange = needs review, red = high concern. Don't invent new colours per page.
- **Don't expose the raw model output.** Logits or top-5 with 4-decimal floats look like noise to a non-ML user.

## Deployment to Streamlit Cloud

1. Push the repo to GitHub. Don't commit the checkpoints — they live on HF Hub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with the GitHub account.
3. Point the app at `patient_app.py` and the `master` branch.
4. Add `requirements-app.txt` with the runtime deps (Streamlit, torch CPU, torchvision, Pillow, plotly, huggingface_hub).
5. First load downloads the checkpoint from HF; subsequent loads use the cache.

## Local dev

```bash
streamlit run patient_app.py
# → http://localhost:8501
```

Streamlit auto-reloads on file save — fast iteration loop for tweaking copy and visuals.
