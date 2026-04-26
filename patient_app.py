"""
Patient-facing Medical Imaging Classifier
Clean clinical UI for MRI and ultrasound classification.

Run with:
    streamlit run patient_app.py
"""

import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import build_model

# ── HF Hub model download ──────────────────────────────────────────────────────
# Set HF_REPO to your Hugging Face model repo, e.g. "YourUsername/medical-imaging-models"
# Upload artifacts_brain/best_model.pt and artifacts_breast/best_model.pt there.
HF_REPO = "Slakje89/medical-imaging-models"

def ensure_model(artifacts: Path) -> bool:
    model_path = artifacts / "best_model.pt"
    if model_path.exists():
        return True
    try:
        from huggingface_hub import hf_hub_download
        with st.spinner(f"Downloading model from Hugging Face Hub..."):
            hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{artifacts.name}/best_model.pt",
                local_dir=".",
            )
        return model_path.exists()
    except Exception as e:
        st.error(f"Could not download model: {e}")
        return False

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Medical Imaging Classifier",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { max-width: 800px; margin: auto; }
    .result-box {
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        text-align: center;
    }
    .result-positive { background: #fff3cd; border: 2px solid #ffc107; }
    .result-negative { background: #d4edda; border: 2px solid #28a745; }
    .result-label { font-size: 2rem; font-weight: bold; margin-bottom: 8px; }
    .confidence-text { font-size: 1.1rem; color: #555; }
    .disclaimer {
        background: #f8f9fa;
        border-left: 4px solid #6c757d;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #6c757d;
        margin-top: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ── Task config ───────────────────────────────────────────────────────────────

TASKS = {
    "Brain Tumor MRI": {
        "artifacts":    Path("artifacts_brain"),
        "concern_classes": {"glioma", "meningioma", "pituitary"},
        "safe_class":   "notumor",
        "description":  "Classifies brain MRI scans into: Glioma, Meningioma, No Tumor, Pituitary",
        "upload_label": "Upload Brain MRI scan (JPG or PNG)",
        "color":        "#4C78A8",
    },
    "Breast Cancer Ultrasound": {
        "artifacts":    Path("artifacts_breast"),
        "concern_classes": {"malignant"},
        "safe_class":   "normal",
        "description":  "Classifies breast ultrasound images into: Benign, Malignant, Normal",
        "upload_label": "Upload breast ultrasound image (JPG or PNG)",
        "color":        "#E45756",
    },
}

FRIENDLY_NAMES = {
    "glioma":     "Glioma",
    "meningioma": "Meningioma",
    "notumor":    "No Tumor Detected",
    "pituitary":  "Pituitary Tumor",
    "benign":     "Benign",
    "malignant":  "Malignant",
    "normal":     "Normal",
}

RISK_FLAGS = {
    "glioma":     ("HIGH CONCERN", "#dc3545", "result-positive"),
    "meningioma": ("MODERATE CONCERN", "#fd7e14", "result-positive"),
    "pituitary":  ("MODERATE CONCERN", "#fd7e14", "result-positive"),
    "notumor":    ("NO FINDING", "#28a745", "result-negative"),
    "malignant":  ("HIGH CONCERN", "#dc3545", "result-positive"),
    "benign":     ("LOW CONCERN", "#ffc107", "result-positive"),
    "normal":     ("NO FINDING", "#28a745", "result-negative"),
}

# ── Model loader ──────────────────────────────────────────────────────────────

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

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Medical Imaging Classifier")
st.caption("AI-assisted medical image analysis — for research and educational use only")
st.markdown("---")

# ── Task selector ─────────────────────────────────────────────────────────────

task_label = st.radio(
    "Select scan type",
    list(TASKS.keys()),
    horizontal=True,
)
cfg = TASKS[task_label]

st.caption(cfg["description"])
st.markdown("")

# ── Model check ───────────────────────────────────────────────────────────────

if not ensure_model(cfg["artifacts"]):
    st.error(
        f"No trained model found for {task_label}. "
        f"Either upload the model to Hugging Face Hub or run: "
        f"`python mlops_pipeline.py --task {cfg['artifacts'].name.replace('artifacts_', '')}`"
    )
    st.stop()

model, classes, tfm = load_model(cfg["artifacts"])

# ── Upload ────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    cfg["upload_label"],
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.info("Upload an image above to get a classification result.")

    with st.expander("How it works"):
        st.markdown("""
        1. Select the scan type (Brain MRI or Breast Ultrasound)
        2. Upload a scan image (JPG or PNG)
        3. The AI model analyses the image and returns a classification
        4. Review the result and confidence level

        **This tool uses a ResNet50 deep learning model trained on publicly
        available medical imaging datasets. Always consult a qualified
        radiologist or clinician before making any medical decision.**
        """)
    st.stop()

# ── Inference ─────────────────────────────────────────────────────────────────

img = Image.open(uploaded).convert("RGB")
x   = tfm(img).unsqueeze(0)

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0].tolist()

top_idx    = int(np.argmax(probs))
prediction = classes[top_idx]
confidence = probs[top_idx]

risk_label, risk_color, result_css = RISK_FLAGS.get(
    prediction, ("UNKNOWN", "#6c757d", "result-positive")
)
friendly = FRIENDLY_NAMES.get(prediction, prediction.upper())

# ── Display image + result ────────────────────────────────────────────────────

col_img, col_result = st.columns([1, 1])

with col_img:
    st.image(img, caption=uploaded.name, use_container_width=True)

with col_result:
    st.markdown(f"""
    <div class="result-box {result_css}">
        <div class="result-label" style="color:{risk_color}">
            {friendly}
        </div>
        <div class="confidence-text">
            Confidence: <strong>{confidence:.1%}</strong>
        </div>
        <br>
        <span style="font-size:1.1rem; font-weight:600; color:{risk_color}">
            {risk_label}
        </span>
    </div>
    """, unsafe_allow_html=True)

    if risk_label in ("HIGH CONCERN", "MODERATE CONCERN"):
        st.warning("Please consult a qualified radiologist or clinician.")
    else:
        st.success("No concerning findings detected by the AI model.")

# ── Probability breakdown ─────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Full breakdown")

prob_data = sorted(
    zip(classes, probs),
    key=lambda x: x[1],
)
labels = [FRIENDLY_NAMES.get(c, c) for c, _ in prob_data]
values = [p for _, p in prob_data]
colors = [
    cfg["color"] if c == prediction else "#d3d3d3"
    for c, _ in prob_data
]

fig = go.Figure(go.Bar(
    x=values,
    y=labels,
    orientation="h",
    marker_color=colors,
    text=[f"{v:.1%}" for v in values],
    textposition="outside",
))
fig.update_layout(
    xaxis=dict(range=[0, 1], tickformat=".0%"),
    height=220,
    margin=dict(l=10, r=60, t=10, b=10),
    plot_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)

# ── Patient info (optional) ───────────────────────────────────────────────────

with st.expander("Add patient details (optional)"):
    col_a, col_b = st.columns(2)
    patient_id  = col_a.text_input("Patient ID")
    scan_date   = col_b.date_input("Scan date")
    notes       = st.text_area("Clinical notes", height=80)

    if st.button("Save report"):
        import json, datetime
        report = {
            "patient_id":  patient_id,
            "scan_date":   str(scan_date),
            "task":        task_label,
            "filename":    uploaded.name,
            "prediction":  prediction,
            "confidence":  round(confidence, 4),
            "risk_label":  risk_label,
            "probabilities": {c: round(p, 4) for c, p in zip(classes, probs)},
            "notes":       notes,
            "timestamp":   datetime.datetime.now().isoformat(),
        }
        out_path = Path(f"report_{patient_id or 'patient'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        out_path.write_text(json.dumps(report, indent=2))
        st.success(f"Report saved to {out_path}")
        st.json(report)

# ── Disclaimer ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="disclaimer">
    <strong>Medical Disclaimer:</strong> This tool is intended for research and
    educational purposes only. It is not a certified medical device and must not
    be used as the sole basis for any clinical decision. Always seek the advice
    of a qualified healthcare professional for diagnosis and treatment.
</div>
""", unsafe_allow_html=True)
