"""
MLOps Dashboard — Medical Imaging Classifier
Unified view of model status, training history, evaluation, predictions, and drift.

Run locally:  streamlit run dashboard.py
Deployed at:  Streamlit Cloud (reads artifacts from Hugging Face Hub)
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import build_model

# ── Config ────────────────────────────────────────────────────────────────────

HF_REPO = "Slakje89/medical-imaging-models"

ARTIFACT_FILES = [
    "best_model.pt",
    "test_report.json",
    "history.csv",
    "config.json",
    "data_stats.json",
    "train_features.csv",
    "prod_features.csv",
]

TASKS = {
    "Brain Tumor MRI": {
        "key":       "brain_tumor",
        "artifacts": Path("artifacts_brain"),
        "color":     "#4C78A8",
    },
    "Breast Cancer Ultrasound": {
        "key":       "breast_cancer",
        "artifacts": Path("artifacts_breast"),
        "color":     "#E45756",
    },
}

st.set_page_config(
    page_title="Medical Imaging MLOps",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── HF Hub download ───────────────────────────────────────────────────────────

def ensure_artifacts(artifacts: Path):
    from huggingface_hub import hf_hub_download
    missing = [f for f in ARTIFACT_FILES if not (artifacts / f).exists()]
    if not missing:
        return
    with st.spinner(f"Downloading artifacts from Hugging Face Hub..."):
        for fname in missing:
            try:
                hf_hub_download(
                    repo_id=HF_REPO,
                    filename=f"{artifacts.name}/{fname}",
                    local_dir=".",
                )
            except Exception:
                pass

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def load_model_for_task(artifacts_str: str):
    artifacts = Path(artifacts_str)
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

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Medical Imaging MLOps")
st.sidebar.markdown("---")
selected_label = st.sidebar.radio("Select task", list(TASKS.keys()))
cfg       = TASKS[selected_label]
artifacts = cfg["artifacts"]

st.sidebar.markdown("---")
st.sidebar.caption("Data sourced from Hugging Face Hub model repository.")

# ── Download artifacts ────────────────────────────────────────────────────────

ensure_artifacts(artifacts)

report     = load_json(artifacts / "test_report.json")
history    = load_csv(artifacts  / "history.csv")
config     = load_json(artifacts / "config.json")
data_stats = load_json(artifacts / "data_stats.json")

if not (artifacts / "best_model.pt").exists():
    st.warning(f"No trained model found for {selected_label}.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────

st.title(f"MLOps Dashboard — {selected_label}")

# ── Row 1: Model Status ───────────────────────────────────────────────────────

st.subheader("Model Status")

col1, col2, col3, col4 = st.columns(4)

if report:
    gate = report.get("gate_passed")
    col1.metric("Test accuracy",  f"{report.get('test_accuracy',    0):.2%}")
    col2.metric("Best val acc",   f"{report.get('best_val_acc',     0):.2%}")
    col3.metric("Test AUROC",     f"{report.get('test_macro_auroc', 0):.4f}")
    col4.metric("Quality gate",   "PASS" if gate else "FAIL" if gate is not None else "N/A")

if config:
    with st.expander("Run config"):
        st.json(config)

# ── Row 2: Training History ───────────────────────────────────────────────────

st.markdown("---")
st.subheader("Training History")

if not history.empty:
    col_loss, col_acc = st.columns(2)

    with col_loss:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["epoch"], y=history["train_loss"],
                                 name="Train loss", line=dict(color=cfg["color"])))
        fig.add_trace(go.Scatter(x=history["epoch"], y=history["val_loss"],
                                 name="Val loss", line=dict(color=cfg["color"], dash="dash")))
        fig.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss",
                          height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_acc:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["epoch"], y=history["train_acc"],
                                 name="Train acc", line=dict(color=cfg["color"])))
        fig.add_trace(go.Scatter(x=history["epoch"], y=history["val_acc"],
                                 name="Val acc", line=dict(color=cfg["color"], dash="dash")))
        fig.update_layout(title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy",
                          height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No training history found.")

# ── Row 3: Evaluation ─────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Test Set Evaluation")

if report:
    col_cm, col_cr = st.columns(2)

    with col_cm:
        cm_data = report.get("confusion_matrix", {})
        labels  = cm_data.get("labels", [])
        matrix  = cm_data.get("matrix", [])
        if matrix:
            fig = px.imshow(
                matrix,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=labels, y=labels,
                color_continuous_scale="Blues",
                text_auto=True,
                title="Confusion Matrix",
            )
            fig.update_layout(height=350, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with col_cr:
        cr   = report.get("classification_report", {})
        rows = []
        for cls in labels:
            if cls in cr:
                rows.append({
                    "Class":     cls,
                    "Precision": round(cr[cls]["precision"], 3),
                    "Recall":    round(cr[cls]["recall"],    3),
                    "F1":        round(cr[cls]["f1-score"],  3),
                    "Support":   int(cr[cls]["support"]),
                })
        if rows:
            st.markdown("**Per-class report**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            macro = cr.get("macro avg", {})
            st.caption(
                f"Macro F1: {macro.get('f1-score', 0):.3f}  |  "
                f"Test accuracy: {report.get('test_accuracy', 0):.2%}  |  "
                f"AUROC: {report.get('test_macro_auroc', 0):.4f}"
            )

# ── Row 4: Live Prediction ────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Live Prediction")

model, classes, tfm = load_model_for_task(str(artifacts))

uploaded = st.file_uploader(
    "Upload an image to classify",
    type=["jpg", "jpeg", "png"],
)

if uploaded and model is not None:
    col_img, col_pred = st.columns([1, 2])

    with col_img:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption=uploaded.name, use_container_width=True)

    with col_pred:
        x = tfm(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].tolist()

        top_idx = int(np.argmax(probs))
        st.markdown(f"### {classes[top_idx].upper()}")
        st.markdown(f"Confidence: **{probs[top_idx]:.1%}**")

        prob_df = pd.DataFrame({
            "Class":       classes,
            "Probability": [round(p, 4) for p in probs],
        }).sort_values("Probability", ascending=True)

        fig = px.bar(
            prob_df, x="Probability", y="Class",
            orientation="h",
            color="Probability",
            color_continuous_scale=[[0, "#d3d3d3"], [1, cfg["color"]]],
            range_x=[0, 1],
        )
        fig.update_layout(
            height=250, showlegend=False,
            coloraxis_showscale=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Row 5: Drift Monitoring ───────────────────────────────────────────────────

st.markdown("---")
st.subheader("Data Drift Monitoring")

train_feat = load_csv(artifacts / "train_features.csv")
prod_feat  = load_csv(artifacts / "prod_features.csv")

if train_feat.empty or prod_feat.empty:
    st.info("No drift data available.")
else:
    features = ["mean_pixel", "std_pixel", "min_pixel", "max_pixel", "aspect_ratio"]
    features = [f for f in features if f in train_feat.columns]

    col_drift, col_dist = st.columns([1, 2])

    with col_drift:
        st.markdown("**Feature statistics**")
        stats = pd.DataFrame({
            "Feature":    features,
            "Train mean": [round(train_feat[f].mean(), 2) for f in features],
            "Prod mean":  [round(prod_feat[f].mean(),  2) for f in features],
            "Delta":      [round(prod_feat[f].mean() - train_feat[f].mean(), 2) for f in features],
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)

    with col_dist:
        selected_feat = st.selectbox("Feature distribution", features)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=train_feat[selected_feat], name="Training (reference)",
            opacity=0.65, marker_color=cfg["color"], histnorm="probability",
        ))
        fig.add_trace(go.Histogram(
            x=prod_feat[selected_feat], name="Production (current)",
            opacity=0.65, marker_color="#FECB52", histnorm="probability",
        ))
        fig.update_layout(
            barmode="overlay", title=f"Distribution: {selected_feat}",
            height=300, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("Medical Imaging MLOps Dashboard · artifacts from Hugging Face Hub · Slakje89/medical-imaging-models")
