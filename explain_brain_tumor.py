"""
Explainable AI for brain tumor MRI classification.

Loads the best checkpoint produced by train_brain_tumor.py and generates
visual explanations using multiple techniques:
  - Grad-CAM / Grad-CAM++  (class-discriminative saliency via gradients
    on the last convolutional layer)
  - Integrated Gradients    (attribution by interpolating from a baseline)
  - Occlusion Sensitivity   (sliding-patch perturbation map)

Outputs are saved to  artifacts/explanations/  as individual image files
and a summary HTML report that can be opened in any browser.

Usage:
    # Explain 8 random test images (default)
    python explain_brain_tumor.py

    # Explain a specific image
    python explain_brain_tumor.py --image path/to/mri.jpg

    # Use more test samples, custom checkpoint
    python explain_brain_tumor.py --data data --checkpoint artifacts/best_model.pt --n 16
"""

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from train_brain_tumor import build_model, index_folder


# ========================= Utilities ========================================= #

def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load model + metadata from the training checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]
    mean = ckpt["mean"]
    std = ckpt["std"]

    model = build_model(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, classes, img_size, mean, std


def get_eval_transform(img_size, mean, std):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_image(path: str, img_size: int, mean: list, std: list, device: torch.device):
    """Load a single image and return (input_tensor [1,3,H,W], raw_rgb [H,W,3])."""
    img = Image.open(path).convert("RGB")
    tfm = get_eval_transform(img_size, mean, std)
    tensor = tfm(img).unsqueeze(0).to(device)

    # Keep a normalised-back version for display
    raw = Image.open(path).convert("RGB").resize((img_size, img_size))
    raw_np = np.array(raw).astype(np.float32) / 255.0
    return tensor, raw_np


def denormalize(tensor, mean, std):
    """Undo ImageNet normalisation for visualisation (expects [C,H,W])."""
    t = tensor.clone()
    for c, m, s in zip(t, mean, std):
        c.mul_(s).add_(m)
    return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


# ========================= Grad-CAM ========================================= #

class GradCAM:
    """Grad-CAM and Grad-CAM++ for a target conv layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None,
                 plus: bool = False):
        """
        Returns a [H, W] heatmap in [0, 1].
        If plus=True uses Grad-CAM++ weighting.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients   # [1, C, h, w]
        acts  = self.activations # [1, C, h, w]

        if plus:
            # Grad-CAM++ weights
            grad_pow2 = grads ** 2
            grad_pow3 = grads ** 3
            denom = 2.0 * grad_pow2 + acts * grad_pow3 + 1e-8
            alpha = grad_pow2 / denom
            alpha = alpha * torch.relu(score.detach() * grads)
            weights = alpha.sum(dim=(2, 3), keepdim=True)
        else:
            # Standard Grad-CAM: global-average-pool the gradients
            weights = grads.mean(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx, output

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ========================= Integrated Gradients ============================== #

def integrated_gradients(model, input_tensor, class_idx=None, steps=50,
                         baseline=None):
    """
    Compute Integrated Gradients attribution for *input_tensor* w.r.t. class_idx.
    Returns an attribution map [H, W] in [0, 1].
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    # Forward on the original to pick class
    with torch.no_grad():
        out = model(input_tensor)
    if class_idx is None:
        class_idx = out.argmax(dim=1).item()

    # Interpolate
    scaled = [baseline + (float(i) / steps) * (input_tensor - baseline)
              for i in range(steps + 1)]
    scaled = torch.cat(scaled, dim=0).requires_grad_(True)

    # Batch forward
    outputs = model(scaled)
    scores = outputs[:, class_idx]
    scores.sum().backward()

    grads = scaled.grad  # [steps+1, C, H, W]
    # Trapezoidal rule
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0, keepdim=True)
    ig = (input_tensor - baseline) * avg_grads  # [1, C, H, W]

    # Collapse to single channel: absolute sum over channels
    attr = ig.squeeze(0).abs().sum(dim=0).cpu().numpy()
    if attr.max() > 0:
        attr = attr / attr.max()
    return attr, class_idx


# ========================= Occlusion Sensitivity ============================ #

def occlusion_sensitivity(model, input_tensor, class_idx=None,
                          patch_size=16, stride=8):
    """
    Slide a grey patch over the image and record the predicted-class
    probability drop.  Returns a sensitivity map [H, W] in [0, 1].
    """
    device = input_tensor.device
    with torch.no_grad():
        base_out = torch.softmax(model(input_tensor), dim=1)
    if class_idx is None:
        class_idx = base_out.argmax(dim=1).item()
    base_prob = base_out[0, class_idx].item()

    _, C, H, W = input_tensor.shape
    sens = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            masked = input_tensor.clone()
            masked[:, :, y:y+patch_size, x:x+patch_size] = 0.0
            with torch.no_grad():
                prob = torch.softmax(model(masked), dim=1)[0, class_idx].item()
            drop = max(base_prob - prob, 0.0)
            sens[y:y+patch_size, x:x+patch_size] += drop
            count[y:y+patch_size, x:x+patch_size] += 1.0

    count[count == 0] = 1.0
    sens = sens / count
    if sens.max() > 0:
        sens = sens / sens.max()
    return sens, class_idx


# ========================= Visualisation ===================================== #

def overlay_heatmap(raw_img, heatmap, alpha=0.5, cmap="jet"):
    """Overlay a [h,w] heatmap on a [H,W,3] RGB image. Returns [H,W,3] uint8."""
    h, w = raw_img.shape[:2]
    hm_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h))
    ).astype(np.float32) / 255.0
    colormap = matplotlib.colormaps.get_cmap(cmap)
    hm_color = colormap(hm_resized)[..., :3]  # drop alpha channel
    blended = (1 - alpha) * raw_img + alpha * hm_color
    return (blended * 255).clip(0, 255).astype(np.uint8)


def make_explanation_figure(raw_img, maps: dict, pred_label: str,
                            true_label: str, prob: float, path: str):
    """
    Create a single figure with the original image + all explanation maps.
    maps: {method_name: heatmap_2d}
    """
    n = 1 + len(maps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    # Original
    axes[0].imshow(raw_img)
    colour = "green" if pred_label == true_label else "red"
    axes[0].set_title(f"True: {true_label}\nPred: {pred_label} ({prob:.1%})",
                      fontsize=10, color=colour)
    axes[0].axis("off")

    for ax, (name, heatmap) in zip(axes[1:], maps.items()):
        overlay = overlay_heatmap(raw_img, heatmap)
        ax.imshow(overlay)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ========================= HTML report ======================================= #

def build_html_report(entries: list, out_dir: Path):
    """Write a self-contained HTML file linking all explanation images."""
    rows_html = []
    for e in entries:
        rel = os.path.relpath(e["figure"], out_dir)
        colour = "green" if e["correct"] else "red"
        rows_html.append(
            f'<div class="card">'
            f'<img src="{rel}" />'
            f'<p><b>True:</b> {e["true_label"]} &nbsp; '
            f'<b style="color:{colour}">Pred:</b> {e["pred_label"]} '
            f'({e["prob"]:.1%})</p>'
            f'<p class="file">{e["image_path"]}</p>'
            f'</div>'
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Brain Tumor XAI Report</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee;
         max-width: 1400px; margin: auto; padding: 20px; }}
  h1 {{ text-align: center; }}
  .card {{ background: #16213e; border-radius: 8px; padding: 12px;
           margin: 18px 0; }}
  .card img {{ width: 100%; border-radius: 4px; }}
  .card p {{ margin: 6px 0; }}
  .file {{ font-size: 0.8em; color: #888; word-break: break-all; }}
</style></head><body>
<h1>Explainable AI &mdash; Brain Tumor MRI</h1>
<p>Methods shown per image: <b>Grad-CAM</b>, <b>Grad-CAM++</b>,
   <b>Integrated Gradients</b>, <b>Occlusion Sensitivity</b></p>
<p>Total images: {len(entries)} &nbsp;|&nbsp;
   Correct: {sum(e['correct'] for e in entries)} &nbsp;|&nbsp;
   Wrong: {sum(not e['correct'] for e in entries)}</p>
{''.join(rows_html)}
</body></html>"""
    report_path = out_dir / "xai_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"HTML report saved to {report_path}")


# ========================= Main ============================================= #

def explain_single(model, img_path, classes, img_size, mean, std, device,
                   out_dir, true_label=None, idx=0):
    """Run all XAI methods on one image and save the figure."""
    input_tensor, raw_img = load_image(img_path, img_size, mean, std, device)

    # Prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
    pred_idx = probs.argmax(dim=1).item()
    pred_label = classes[pred_idx]
    pred_prob = probs[0, pred_idx].item()
    if true_label is None:
        true_label = pred_label

    # Target conv layer = last block of ResNet layer4
    target_layer = model.layer4[-1].conv3

    # 1. Grad-CAM
    gc = GradCAM(model, target_layer)
    cam, _, _ = gc(input_tensor, class_idx=pred_idx, plus=False)
    cam_pp, _, _ = gc(input_tensor, class_idx=pred_idx, plus=True)
    gc.remove_hooks()

    # 2. Integrated Gradients
    ig_map, _ = integrated_gradients(model, input_tensor, class_idx=pred_idx,
                                     steps=50)

    # 3. Occlusion Sensitivity
    occ_map, _ = occlusion_sensitivity(model, input_tensor, class_idx=pred_idx,
                                       patch_size=16, stride=8)

    maps = {
        "Grad-CAM": cam,
        "Grad-CAM++": cam_pp,
        "Integrated Gradients": ig_map,
        "Occlusion Sensitivity": occ_map,
    }

    fig_path = str(out_dir / f"xai_{idx:03d}_{pred_label}.png")
    make_explanation_figure(raw_img, maps, pred_label, true_label,
                           pred_prob, fig_path)
    print(f"  [{idx}] {Path(img_path).name}  true={true_label}  "
          f"pred={pred_label} ({pred_prob:.1%})")

    return {
        "image_path": str(img_path),
        "true_label": true_label,
        "pred_label": pred_label,
        "prob": pred_prob,
        "correct": pred_label == true_label,
        "figure": fig_path,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Explainable AI for brain tumor MRI classification."
    )
    ap.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt",
                    help="Path to the best model checkpoint.")
    ap.add_argument("--data", type=str, default="data",
                    help="Root data directory (same as train_brain_tumor.py).")
    ap.add_argument("--image", type=str, default="",
                    help="Path to a single image to explain (overrides --n).")
    ap.add_argument("--n", type=int, default=8,
                    help="Number of random test images to explain.")
    ap.add_argument("--out", type=str, default="artifacts/explanations",
                    help="Output directory for explanation figures.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, classes, img_size, mean, std = load_checkpoint(args.checkpoint, device)
    print(f"Classes: {classes}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []

    if args.image:
        # Single image mode
        entry = explain_single(model, args.image, classes, img_size, mean, std,
                               device, out_dir, true_label=None, idx=0)
        entries.append(entry)
    else:
        # Sample from test set
        df = index_folder(Path(args.data))
        test_df = df[(df["split"] == "Testing") & (df["ok"])].copy()
        if test_df.empty:
            print("No test images found. Provide --image or --data.")
            sys.exit(1)

        n = min(args.n, len(test_df))
        sample = test_df.sample(n=n, random_state=args.seed)
        print(f"\nExplaining {n} test images …")

        for i, (_, row) in enumerate(sample.iterrows()):
            entry = explain_single(model, row["path"], classes, img_size,
                                   mean, std, device, out_dir,
                                   true_label=row["label"], idx=i)
            entries.append(entry)

    # Summary
    correct = sum(e["correct"] for e in entries)
    print(f"\nResults: {correct}/{len(entries)} correct")

    # Save JSON summary
    summary_path = out_dir / "xai_summary.json"
    summary_path.write_text(json.dumps(entries, indent=2))
    print(f"JSON summary: {summary_path}")

    # Build HTML report
    build_html_report(entries, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
