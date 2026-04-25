# ── Stage 1: base image ───────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# ── Stage 2: install serving dependencies only (no sklearn/mlflow/dvc) ────────
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        Pillow \
        fastapi \
        "uvicorn[standard]" \
        python-multipart

# ── Stage 3: copy source code and trained model checkpoints ───────────────────
COPY model.py .
COPY serve.py .
COPY artifacts_brain/best_model.pt  artifacts_brain/best_model.pt
COPY artifacts_breast/best_model.pt artifacts_breast/best_model.pt

# ── Stage 4: expose port and run ──────────────────────────────────────────────
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
