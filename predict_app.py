"""
Streamlit deployment app for breast cancer and brain tumor prediction.

Loads both trained checkpoints and serves a browser-based UI where you
can upload an image and get class probabilities from either model.

Usage:
    streamlit run predict_app.py
    # Opens http://localhost:8501 in your browser automatically
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---- make sure local train scripts are importable --------------------------
sys.path.insert(0, str(Path(__file__).parent))
from train_brain_tumor import build_model as build_brain_model
from train_breast_cancer import build_model as build_breast_model

# ========================= Config =========================================== #

MODELS = {
    "Brain Tumor MRI": {
        "checkpoint":   "artifacts/best_model.pt",
        "build_fn":     build_brain_model,
        "description":  "Classifies brain MRI scans into 4 categories.",
        "classes_hint": "glioma · meningioma · no tumor · pituitary",
        "color":        "#4e8cff",
    },
    "Breast Cancer Ultrasound": {
        "checkpoint":   "artifacts_breast/best_model.pt",
        "build_fn":     build_breast_model,
        "description":  "Classifies breast ultrasound images into 3 categories.",
        "classes_hint": "benign · malignant · normal",
        "color":        "#ff6b6b",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================= Model loading ==================================== #

@st.cache_resource(show_spinner="Loading model ...")
def load_model(model_name: str):
    """
    Load a checkpoint and return (model, classes, transform).
    Cached so the model is only loaded once per session.
    """
    cfg  = MODELS[model_name]
    ckpt = torch.load(cfg["checkpoint"], map_location=DEVICE, weights_only=False)

    model = cfg["build_fn"](len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])

    return model, ckpt["classes"], tfm


# ========================= Inference ======================================== #

def run_inference(model, classes, tfm, image: Image.Image) -> dict:
    """Return {class_name: probability} for every class."""
    tensor = tfm(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
    return {cls: float(prob) for cls, prob in zip(classes, probs)}


# ========================= UI =============================================== #

st.set_page_config(
    page_title="Medical Imaging Predictor",
    page_icon="🩺",
    layout="wide",
)

# ---- Sidebar: model selector -----------------------------------------------
st.sidebar.title("🩺 Medical Imaging Predictor")
st.sidebar.markdown("---")
model_name = st.sidebar.radio(
    "Select model",
    list(MODELS.keys()),
    help="Choose which trained model to use for prediction.",
)
cfg = MODELS[model_name]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{model_name}**")
st.sidebar.caption(cfg["description"])
st.sidebar.caption(f"Classes: {cfg['classes_hint']}")
st.sidebar.markdown("---")
st.sidebar.caption(f"Device: `{DEVICE}`")

# ---- Main area -------------------------------------------------------------
st.title(model_name)
st.caption(cfg["description"])

# Check checkpoint exists before trying to load
ckpt_path = Path(cfg["checkpoint"])
if not ckpt_path.exists():
    st.error(
        f"Checkpoint not found: `{ckpt_path}`\n\n"
        f"Run the training script first:\n"
        f"```\npython {'train_brain_tumor.py' if 'Brain' in model_name else 'train_breast_cancer.py'}\n```"
    )
    st.stop()

# Load model (cached after first load)
model, classes, tfm = load_model(model_name)

# ---- Upload + predict ------------------------------------------------------
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("Upload Image")
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=uploaded.name, use_container_width=True)

with col_result:
    st.subheader("Prediction")

    if not uploaded:
        st.info("Upload an image on the left to see the prediction.")
    else:
        with st.spinner("Running inference ..."):
            scores = run_inference(model, classes, tfm, image)

        # Sort by probability descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_class, top_prob = sorted_scores[0]

        # Top prediction badge
        st.markdown(
            f"""
            <div style="
                background-color:{cfg['color']}22;
                border-left: 5px solid {cfg['color']};
                padding: 16px 20px;
                border-radius: 6px;
                margin-bottom: 20px;
            ">
                <div style="font-size:0.85rem; color:#aaa; margin-bottom:4px;">
                    Top prediction
                </div>
                <div style="font-size:2rem; font-weight:700; color:{cfg['color']};">
                    {top_class.upper()}
                </div>
                <div style="font-size:1.1rem; color:#ccc;">
                    Confidence: <b>{top_prob:.1%}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence bar for every class
        st.markdown("**All class probabilities**")
        for cls, prob in sorted_scores:
            bar_color = cfg["color"] if cls == top_class else "#555"
            st.markdown(
                f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between;
                                font-size:0.9rem; margin-bottom:3px;">
                        <span style="font-weight:{'700' if cls == top_class else '400'}">
                            {cls}
                        </span>
                        <span>{prob:.1%}</span>
                    </div>
                    <div style="background:#333; border-radius:4px; height:10px;">
                        <div style="
                            width:{prob*100:.1f}%;
                            background:{bar_color};
                            height:10px;
                            border-radius:4px;
                            transition: width 0.4s ease;
                        "></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Clinical note for malignant predictions
        if top_class == "malignant" and top_prob > 0.5:
            st.warning(
                "**Note:** This prediction suggests a malignant finding. "
                "This tool is for research purposes only — always consult a "
                "qualified radiologist for clinical decisions."
            )
