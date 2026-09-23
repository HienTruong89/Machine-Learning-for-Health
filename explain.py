"""
Explainable AI for the brain tumor MRI and breast cancer ultrasound classifiers.

Loads the best checkpoint produced by mlops_pipeline.py and generates four
visual explanations per image:
  - Grad-CAM / Grad-CAM++   (class-discriminative saliency via gradients
                             on the last convolutional layer)
  - Integrated Gradients    (attribution by interpolating from a baseline)
  - Occlusion Sensitivity   (sliding-patch perturbation map)

Test images come from <artifacts>/test_images.csv (written by mlops_pipeline.py),
so explanations are drawn from the held-out set. Outputs go to
<artifacts>/explanations/ as PNG figures plus an HTML report.

Usage:
    python explain.py --task brain_tumor                  # 8 random test images
    python explain.py --task breast_cancer --n 16
    python explain.py --task brain_tumor --image path/to/mri.jpg
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from mlops_pipeline import TASK_META, index_images
from model import load_checkpoint

TITLES = {"brain_tumor": "Brain Tumor MRI", "breast_cancer": "Breast Cancer Ultrasound"}


# ========================= Explanation methods ============================== #

class GradCAM:
    """Grad-CAM and Grad-CAM++ for a target conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self._hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx: int, plus: bool = False):
        """Return a [h, w] heatmap in [0, 1]. plus=True uses Grad-CAM++ weighting."""
        self.model.zero_grad()
        score = self.model(input_tensor)[0, class_idx]
        score.backward()

        grads = self.gradients    # [1, C, h, w]
        acts  = self.activations  # [1, C, h, w]
        if plus:
            denom   = 2.0 * grads ** 2 + acts * grads ** 3 + 1e-8
            alpha   = grads ** 2 / denom * torch.relu(score.detach() * grads)
            weights = alpha.sum(dim=(2, 3), keepdim=True)
        else:
            # Standard Grad-CAM: global-average-pool the gradients
            weights = grads.mean(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * acts).sum(dim=1)).squeeze().cpu().numpy()
        return normalise(cam)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def integrated_gradients(model, input_tensor, class_idx: int, steps: int = 50):
    """Integrated Gradients from a black baseline. Returns a [H, W] map in [0, 1]."""
    baseline = torch.zeros_like(input_tensor)
    alphas   = torch.linspace(0, 1, steps + 1, device=input_tensor.device).view(-1, 1, 1, 1)
    scaled   = (baseline + alphas * (input_tensor - baseline)).requires_grad_(True)

    model(scaled)[:, class_idx].sum().backward()

    grads     = scaled.grad                                       # [steps+1, C, H, W]
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0, keepdim=True) / 2  # trapezoidal rule
    ig        = (input_tensor - baseline) * avg_grads
    return normalise(ig.squeeze(0).abs().sum(dim=0).cpu().numpy())


def occlusion_sensitivity(model, input_tensor, class_idx: int,
                          patch_size: int = 16, stride: int = 8):
    """Slide a zero patch over the image and record the drop in the class
    probability. Returns a [H, W] map in [0, 1]."""
    def prob(x):
        with torch.no_grad():
            return torch.softmax(model(x), dim=1)[0, class_idx].item()

    base_prob = prob(input_tensor)
    _, _, H, W = input_tensor.shape
    sens  = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            masked = input_tensor.clone()
            masked[:, :, y:y + patch_size, x:x + patch_size] = 0.0
            sens [y:y + patch_size, x:x + patch_size] += max(base_prob - prob(masked), 0.0)
            count[y:y + patch_size, x:x + patch_size] += 1.0

    return normalise(sens / np.maximum(count, 1.0))


def normalise(heatmap: np.ndarray) -> np.ndarray:
    """Scale a non-negative heatmap to [0, 1]."""
    return heatmap / heatmap.max() if heatmap.max() > 0 else heatmap


# ========================= Visualisation ==================================== #

def overlay_heatmap(raw_img, heatmap, alpha=0.5, cmap="jet"):
    """Overlay a [h, w] heatmap on a [H, W, 3] float RGB image. Returns uint8."""
    h, w = raw_img.shape[:2]
    hm = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h))
    ).astype(np.float32) / 255.0
    hm_color = matplotlib.colormaps[cmap](hm)[..., :3]  # drop alpha channel
    blended  = (1 - alpha) * raw_img + alpha * hm_color
    return (blended * 255).clip(0, 255).astype(np.uint8)


def save_figure(raw_img, maps: dict, pred_label, true_label, prob, path):
    """One row: the original image followed by every explanation map."""
    fig, axes = plt.subplots(1, 1 + len(maps), figsize=(4 * (1 + len(maps)), 4))

    colour = "green" if pred_label == true_label else "red"
    axes[0].imshow(raw_img)
    axes[0].set_title(f"True: {true_label}\nPred: {pred_label} ({prob:.1%})",
                      fontsize=10, color=colour)
    for ax, (name, heatmap) in zip(axes[1:], maps.items()):
        ax.imshow(overlay_heatmap(raw_img, heatmap))
        ax.set_title(name, fontsize=10)
    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_html_report(entries: list, out_dir: Path, title: str):
    """Write an HTML page showing every explanation figure."""
    cards = []
    for e in entries:
        colour = "green" if e["correct"] else "red"
        cards.append(
            f'<div class="card">'
            f'<img src="{os.path.relpath(e["figure"], out_dir)}" />'
            f'<p><b>True:</b> {e["true_label"]} &nbsp; '
            f'<b style="color:{colour}">Pred:</b> {e["pred_label"]} ({e["prob"]:.1%})</p>'
            f'<p class="file">{e["image_path"]}</p>'
            f'</div>'
        )
    n_correct = sum(e["correct"] for e in entries)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>{title} XAI Report</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee;
         max-width: 1400px; margin: auto; padding: 20px; }}
  h1 {{ text-align: center; }}
  .card {{ background: #16213e; border-radius: 8px; padding: 12px; margin: 18px 0; }}
  .card img {{ width: 100%; border-radius: 4px; }}
  .card p {{ margin: 6px 0; }}
  .file {{ font-size: 0.8em; color: #888; word-break: break-all; }}
</style></head><body>
<h1>Explainable AI &mdash; {title}</h1>
<p>Methods shown per image: <b>Grad-CAM</b>, <b>Grad-CAM++</b>,
   <b>Integrated Gradients</b>, <b>Occlusion Sensitivity</b></p>
<p>Total images: {len(entries)} &nbsp;|&nbsp; Correct: {n_correct}
   &nbsp;|&nbsp; Wrong: {len(entries) - n_correct}</p>
{''.join(cards)}
</body></html>"""
    report_path = out_dir / "xai_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"HTML report saved to {report_path}")


# ========================= Main ============================================= #

def explain_single(model, tfm, classes, img_path, device, out_dir,
                   true_label=None, idx=0) -> dict:
    """Run all XAI methods on one image and save the figure."""
    img          = Image.open(img_path).convert("RGB")
    input_tensor = tfm(img).unsqueeze(0).to(device)
    size         = input_tensor.shape[-1]
    raw_img      = np.asarray(img.resize((size, size)), dtype=np.float32) / 255.0

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]
    pred_idx   = int(probs.argmax())
    pred_label = classes[pred_idx]
    pred_prob  = probs[pred_idx].item()
    true_label = true_label or pred_label

    # Target layer: last bottleneck block of ResNet layer4
    gc = GradCAM(model, model.layer4[-1].conv3)
    maps = {
        "Grad-CAM":              gc(input_tensor, pred_idx),
        "Grad-CAM++":            gc(input_tensor, pred_idx, plus=True),
        "Integrated Gradients":  integrated_gradients(model, input_tensor, pred_idx),
        "Occlusion Sensitivity": occlusion_sensitivity(model, input_tensor, pred_idx),
    }
    gc.remove_hooks()

    fig_path = str(out_dir / f"xai_{idx:03d}_{pred_label}.png")
    save_figure(raw_img, maps, pred_label, true_label, pred_prob, fig_path)
    print(f"  [{idx}] {Path(img_path).name}  true={true_label}  "
          f"pred={pred_label} ({pred_prob:.1%})")

    return {
        "image_path": str(img_path),
        "true_label": true_label,
        "pred_label": pred_label,
        "prob":       pred_prob,
        "correct":    pred_label == true_label,
        "figure":     fig_path,
    }


def load_test_images(task: str, artifacts: Path, data: str) -> pd.DataFrame:
    """Held-out test images: test_images.csv if the pipeline wrote one,
    otherwise re-index the data folder (Testing/ split for brain_tumor)."""
    test_csv = artifacts / "test_images.csv"
    if test_csv.exists():
        print(f"Loading test images from {test_csv}")
        return pd.read_csv(test_csv)

    print(f"{test_csv} not found — indexing images under '{data}'")
    df = index_images(SimpleNamespace(task=task, data=data))
    df = df[df["ok"]]
    if TASK_META[task]["layout"] == "split":
        df = df[df["split"] == "Testing"]
    return df


def main():
    ap = argparse.ArgumentParser(description="Explainable AI for the medical imaging classifiers.")
    ap.add_argument("--task", required=True, choices=list(TASK_META))
    ap.add_argument("--checkpoint", default="",
                    help="Model checkpoint (default: <task artifacts>/best_model.pt).")
    ap.add_argument("--data", default="",
                    help="Data root, used only if test_images.csv is missing.")
    ap.add_argument("--image", default="",
                    help="Explain a single image instead of sampling the test set.")
    ap.add_argument("--n", type=int, default=8, help="Number of random test images.")
    ap.add_argument("--out", default="",
                    help="Output folder (default: <task artifacts>/explanations).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta       = TASK_META[args.task]
    artifacts  = Path(meta["default_out"])
    checkpoint = args.checkpoint or str(artifacts / "best_model.pt")
    out_dir    = Path(args.out or artifacts / "explanations")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | loading checkpoint: {checkpoint}")
    model, classes, tfm = load_checkpoint(checkpoint, device)
    print(f"Classes: {classes}")

    if args.image:
        entries = [explain_single(model, tfm, classes, args.image, device, out_dir)]
    else:
        test_df = load_test_images(args.task, Path(checkpoint).parent,
                                   args.data or meta["default_data"])
        if test_df.empty:
            print("No test images found. Run mlops_pipeline.py first, or pass --image.")
            sys.exit(1)
        sample = test_df.sample(n=min(args.n, len(test_df)), random_state=args.seed)
        print(f"\nExplaining {len(sample)} test images ...")
        entries = [
            explain_single(model, tfm, classes, row["path"], device, out_dir,
                           true_label=row["label"], idx=i)
            for i, (_, row) in enumerate(sample.iterrows())
        ]

    print(f"\nResults: {sum(e['correct'] for e in entries)}/{len(entries)} correct")
    summary_path = out_dir / "xai_summary.json"
    summary_path.write_text(json.dumps(entries, indent=2))
    print(f"JSON summary: {summary_path}")
    build_html_report(entries, out_dir, TITLES.get(args.task, args.task))


if __name__ == "__main__":
    main()
