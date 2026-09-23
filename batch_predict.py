"""
Batch MRI classifier — calls the FastAPI server for every image in a folder.

Usage:
    python batch_predict.py --folder data/Testing/glioma --task brain_tumor
    python batch_predict.py --folder data_breast/malignant --task breast_cancer
    python batch_predict.py --folder scans/ --task brain_tumor --out results.csv

The server must be running:
    docker run -p 8000:8000 medical-imaging-api:v1
    OR
    uvicorn serve:app --port 8000
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_SERVER = "http://localhost:8000"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def classify_image(server: str, image_path: Path, task: str) -> dict:
    """Send one image to the API and return its JSON result. Raises on HTTP errors."""
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{server}/predict/{task}",
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )
    response.raise_for_status()
    return response.json()


def check_server(server: str) -> bool:
    try:
        return requests.get(f"{server}/health", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def batch_predict(server: str, folder: Path, task: str, out_csv: Path) -> None:
    images = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not images:
        print(f"No images found in {folder}")
        sys.exit(1)

    print(f"Found {len(images)} images in '{folder}'")
    print(f"Task: {task}  |  Server: {server}")
    print()

    results   = []
    errors    = []
    t_start   = time.time()

    for img_path in tqdm(images, desc="Classifying"):
        try:
            result = classify_image(server, img_path, task)
            results.append({
                "file":       img_path.name,
                "path":       str(img_path),
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                **{f"prob_{cls}": prob
                   for cls, prob in result.get("probabilities", {}).items()},
                "status": "ok",
            })
        except Exception as e:
            errors.append(img_path.name)
            results.append({
                "file":       img_path.name,
                "path":       str(img_path),
                "prediction": "ERROR",
                "confidence": 0.0,
                "status":     str(e),
            })

    elapsed = time.time() - t_start

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    # ── Print summary ─────────────────────────────────────────────────────────
    ok_results = [r for r in results if r["status"] == "ok"]

    print()
    print("=" * 50)
    print(f"BATCH COMPLETE — {len(ok_results)}/{len(images)} successful")
    print(f"Time: {elapsed:.1f}s  ({elapsed/len(images):.2f}s per image)")
    print()

    if ok_results:
        counts = Counter(r["prediction"] for r in ok_results)
        avg_conf = sum(r["confidence"] for r in ok_results) / len(ok_results)

        print("Prediction breakdown:")
        for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "#" * int(count / len(ok_results) * 30)
            pct = count / len(ok_results) * 100
            print(f"  {cls:<15} {bar:<30} {count:>4} ({pct:.1f}%)")

        print()
        print(f"Average confidence: {avg_conf:.1%}")
        print(f"Results saved to:   {out_csv}")

    if errors:
        print()
        print(f"Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    print("=" * 50)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch MRI classifier via REST API.")
    ap.add_argument("--folder", required=True,  help="Folder containing images.")
    ap.add_argument("--task",   required=True,  choices=["brain_tumor", "breast_cancer"])
    ap.add_argument("--out",    default="",     help="Output CSV path (default: <folder>_results.csv)")
    ap.add_argument("--server", default=DEFAULT_SERVER, help=f"API server URL (default: {DEFAULT_SERVER})")
    args = ap.parse_args()

    server  = args.server.rstrip("/")
    folder  = Path(args.folder)
    out_csv = Path(args.out) if args.out else Path(f"{folder.name}_results.csv")

    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    if not check_server(server):
        print(f"Server not reachable at {server}")
        print("Start it with:")
        print("  docker run -p 8000:8000 medical-imaging-api:v1")
        print("  OR: uvicorn serve:app --port 8000")
        sys.exit(1)

    batch_predict(server, folder, args.task, out_csv)


if __name__ == "__main__":
    main()
