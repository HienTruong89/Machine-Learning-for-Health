# Stage 12 — CI/CD with GitHub Actions

**Goal:** every push that touches the pipeline retrains a smoke-sized model, validates a basic accuracy threshold, and verifies the Docker image still builds. Failures block merging.

## What CI should and shouldn't do

- **Should:** install deps deterministically, run a 1-epoch smoke train on a small dataset, assert a minimum accuracy, build the Docker image, upload artifacts.
- **Shouldn't:** run a full 20-epoch training (too slow, too expensive), push to a registry without explicit approval, run on every commit if you have a busy repo (use `paths:` filters).

CI verifies the *pipeline* still works, not that the *model* is production-quality. Production training happens on a beefier runner triggered by `workflow_dispatch` or a release tag.

## The workflow file

```yaml
# .github/workflows/mlops.yml
name: MLOps Pipeline

on:
  push:
    branches: [master]
    paths:
      - mlops_pipeline.py
      - model.py
      - serve.py
      - requirements.txt
      - Dockerfile
  workflow_dispatch:
    inputs:
      kaggle_user:
        description: "Kaggle username"
        required: true
      kaggle_key:
        description: "Kaggle API key"
        required: true

jobs:
  train-and-validate:
    name: Train · Validate · Register
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: |
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
          pip install mlflow kagglehub scikit-learn pandas numpy tqdm Pillow

      - name: Smoke train
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME || inputs.kaggle_user }}
          KAGGLE_KEY:      ${{ secrets.KAGGLE_KEY      || inputs.kaggle_key }}
        run: |
          python mlops_pipeline.py \
            --task brain_tumor \
            --epochs 1 \
            --batch 16 \
            --min_val_acc 0.0 \
            --min_auroc   0.0

      - name: Smoke-test threshold
        run: |
          python - <<'EOF'
          import json, sys
          r = json.load(open("artifacts_brain/test_report.json"))
          acc = r["test_accuracy"]
          print(f"Test accuracy: {acc:.2%}")
          if acc < 0.30:
              sys.exit(f"FAIL: {acc:.2%} below 30% smoke threshold")
          print("PASS")
          EOF

      - name: Upload trained artifacts
        uses: actions/upload-artifact@v4
        with:
          name: brain-tumor-model
          path: |
            artifacts_brain/best_model.pt
            artifacts_brain/model.torchscript
            artifacts_brain/test_report.json
            artifacts_brain/history.csv

  docker-build:
    name: Docker Build (smoke test)
    runs-on: ubuntu-latest
    needs: train-and-validate
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: brain-tumor-model
          path: artifacts_brain/
      - name: Placeholder for second model
        run: mkdir -p artifacts_breast && dd if=/dev/zero bs=1 count=1 of=artifacts_breast/best_model.pt
      - run: docker build -t medical-imaging-api:ci .
```

## Two thresholds, two purposes

CI uses a *smoke threshold* (30% — anything above random guessing for 4 classes) so a 1-epoch run still passes. Production thresholds (95% val acc / 0.90 AUROC) live in `TASK_META` and are enforced by `mlops_pipeline.py` itself when you do a real training run with `--min_val_acc` / `--min_auroc` left at their defaults.

CI's job: catch regressions in the *code*. Production gate's job: catch regressions in the *model*. Don't conflate them.

## Secrets vs. inputs

Two trigger paths, two credential sources:

- **`push` trigger** uses repository secrets (`secrets.KAGGLE_USERNAME` / `secrets.KAGGLE_KEY`). Set these once at *Settings → Secrets and variables → Actions*.
- **`workflow_dispatch`** lets a human paste credentials into the GitHub UI. Useful for one-off training runs without committing or storing credentials.

The `${{ secrets.X || inputs.X }}` fallback handles both seamlessly.

## Adding a new dataset to CI

1. Add the new task to `TASK_META` in `mlops_pipeline.py`.
2. Duplicate the `Smoke train` step with `--task <new_task>` and a new `min_val_acc` of `0.0` for CI.
3. Duplicate the `Upload trained artifacts` step with the new `artifacts_<task>` path.
4. Drop the `Placeholder for second model` step — once both real models are produced by CI, the placeholder isn't needed.

## Beyond GitHub Actions

The same pattern transposes to other runners:

- **Azure DevOps Pipelines** — replace `uses:` with `task:`, the rest of the YAML is similar. Use Azure Service Connection for ACR push.
- **GitLab CI** — `.gitlab-ci.yml` with `image: python:3.11-slim`, otherwise identical commands.
- **Self-hosted with GPU** — `runs-on: [self-hosted, gpu]`, increase `--epochs`, drop the smoke threshold.

For CD (push to a registry, deploy to App Service / Kubernetes), see the [CLI tools doc](13_cli_tools.md) — it covers Azure CLI / `docker push` / `az webapp` end-to-end.
