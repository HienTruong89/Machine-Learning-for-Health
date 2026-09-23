# CLI tools setup — Git, Docker, Azure (and Kaggle, Hugging Face)

End-to-end CLI plumbing for this pipeline: install, authenticate, and the exact commands you'll use during training, packaging, and deployment.

The recommended order on a fresh machine is **Git → Kaggle → Hugging Face → Docker → Azure**. Earlier tools are needed by later ones (e.g. you can't push your repo before Git is configured).

## 1. Git CLI — source control

### Install

- **Windows:** download from [git-scm.com](https://git-scm.com/download/win), or `winget install --id Git.Git -e`.
- **macOS:** `brew install git` (Xcode CLT also ships one).
- **Linux:** `sudo apt-get install git` / `sudo dnf install git`.

### Configure once per machine

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch master
git config --global pull.rebase false
```

### Authentication options (pick one)

- **GitHub CLI (recommended):** `gh auth login` — handles HTTPS auth, SSH keys, and 2FA in one prompt.
- **Personal Access Token:** create at *github.com → Settings → Developer settings → Personal access tokens (fine-grained)*; paste it when Git prompts for the password on `git push`.
- **SSH keys:** `ssh-keygen -t ed25519 -C "you@example.com"`, then add `~/.ssh/id_ed25519.pub` at *GitHub → Settings → SSH and GPG keys*.

### Day-to-day commands you'll actually use

```bash
git clone https://github.com/<user>/<repo>.git
git status
git switch -c feature/new-dataset            # create + switch to new branch
git add mlops_pipeline.py model.py
git commit -m "Add <new_dataset> task to pipeline"
git push -u origin feature/new-dataset
gh pr create --fill                           # via GitHub CLI
```

### Adding a new project / dataset to an existing repo

```bash
git switch master && git pull
git switch -c feature/<new_task>
# edit TASK_META, model code, add tests
git add -A && git commit -m "Add <new_task> classifier"
git push -u origin feature/<new_task>
```

## 2. Kaggle CLI — dataset access

This pipeline uses `kagglehub` (a Python library) instead of the `kaggle` CLI for downloading. But the `kaggle` CLI is still useful for searching and inspecting datasets.

### Install

```bash
pip install kaggle           # CLI
pip install kagglehub        # what mlops_pipeline.py uses
```

### Authenticate

1. Visit [kaggle.com/settings → API → Create New Token](https://www.kaggle.com/settings).
2. Save the downloaded `kaggle.json` to:
   - **Windows:** `C:\Users\<you>\.kaggle\kaggle.json`
   - **macOS / Linux:** `~/.kaggle/kaggle.json`, then `chmod 600 ~/.kaggle/kaggle.json`.

Or set env vars instead — handy in CI:

```bash
export KAGGLE_USERNAME=<user>
export KAGGLE_KEY=<key>
```

### Useful commands when scouting datasets

```bash
kaggle datasets list -s "medical imaging"
kaggle datasets files <user/dataset-slug>
kaggle datasets download -d <user/dataset-slug> -p data/
```

For new tasks, find the dataset slug, then add it to `TASK_META["<new_task>"]["kaggle_dataset"]` and let `kagglehub` handle the actual download.

## 3. Hugging Face CLI — model hosting

This is how the patient-facing Streamlit app gets its weights without bloating the repo.

### Install + auth

```bash
pip install -U huggingface_hub
huggingface-cli login                  # paste a token from huggingface.co/settings/tokens
```

The token needs `write` access if you'll be uploading models; `read` is enough for downloading public ones.

### Create a model repo (once per project)

Easier in the web UI: [huggingface.co/new](https://huggingface.co/new) → choose "Model" → name it `medical-imaging-models`. Or:

```bash
huggingface-cli repo create medical-imaging-models --type model
```

### Upload a trained checkpoint

After every passing training run:

```bash
# General form
huggingface-cli upload <user>/medical-imaging-models  \
    artifacts_<task>/best_model.pt                    \
    artifacts_<task>/best_model.pt

# Concrete examples
huggingface-cli upload Slakje89/medical-imaging-models \
    artifacts_brain/best_model.pt artifacts_brain/best_model.pt
huggingface-cli upload Slakje89/medical-imaging-models \
    artifacts_breast/best_model.pt artifacts_breast/best_model.pt
```

The two-arg form is `<local_path> <repo_path>`. Match the repo path to what `patient_app.py` requests (`<artifacts_dir>/best_model.pt`).

### Verify

```bash
huggingface-cli download <user>/medical-imaging-models  artifacts_<task>/best_model.pt
```

## 4. Docker CLI — container packaging

### Install

- **Windows / macOS:** [Docker Desktop](https://www.docker.com/products/docker-desktop). Includes the CLI and a working daemon.
- **Linux:** [docker.com/install](https://docs.docker.com/engine/install/) — distro-specific.

Verify with `docker version` and `docker run hello-world`.

### Build, run, test the serving image

```bash
# Build (uses the Dockerfile in repo root)
docker build -t medical-imaging-api:v1 .

# Run, mapping host:container ports
docker run -d --name medical-api -p 8000:8000 medical-imaging-api:v1

# Test the API
curl http://localhost:8000/health
curl -F "file=@scan.jpg" http://localhost:8000/predict/brain_tumor

# Tail logs / stop / remove
docker logs -f medical-api
docker stop medical-api && docker rm medical-api
```

### Tag and push to a registry

Generic Docker Hub:

```bash
docker login                         # paste DH password / token
docker tag medical-imaging-api:v1 <dockerhub-user>/medical-imaging-api:v1
docker push <dockerhub-user>/medical-imaging-api:v1
```

For Azure Container Registry, see the Azure section below — the auth is different (`az acr login`).

### Adding a new dataset to the image

The Dockerfile copies one checkpoint per task explicitly. When you add `<new_task>`:

1. Train the model (`python mlops_pipeline.py --task <new_task>`).
2. Add a new `COPY` line:
   ```dockerfile
   COPY artifacts_<new_task>/best_model.pt artifacts_<new_task>/best_model.pt
   ```
3. The serve.py `TASK_CONFIG` dict needs the matching entry — pointing at the same path.

Avoid `COPY artifacts_*/` glob — it silently includes whatever's left over from older tasks.

## 5. Azure CLI — cloud deploy

For a managed deployment of the FastAPI server to Azure App Service (Linux containers).

### Install

- **Windows:** `winget install -e --id Microsoft.AzureCLI`
- **macOS:** `brew install azure-cli`
- **Linux:** `curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash`

Verify with `az version`.

### Sign in

```bash
az login                              # opens a browser; for headless: az login --use-device-code
az account set --subscription "<sub-name-or-id>"
az account show                       # confirm context
```

### One-time resource setup

Pick names you'll reuse — they're scoped to the subscription:

```bash
RG=ml-medical-rg
LOC=westeurope
ACR=mlmedicalacr$RANDOM               # must be globally unique, lowercase
PLAN=ml-medical-plan
APP=medical-imaging-api-$RANDOM       # must be globally unique

az group create --name $RG --location $LOC
az acr create  --resource-group $RG --name $ACR --sku Basic --admin-enabled true
az appservice plan create --name $PLAN --resource-group $RG --is-linux --sku B1
```

### Push the Docker image to ACR

```bash
az acr login --name $ACR

docker tag  medical-imaging-api:v1   $ACR.azurecr.io/medical-imaging-api:v1
docker push                          $ACR.azurecr.io/medical-imaging-api:v1

az acr repository list --name $ACR --output table
```

### Create the web app pointing at the image

```bash
az webapp create \
  --resource-group $RG \
  --plan $PLAN \
  --name $APP \
  --deployment-container-image-name $ACR.azurecr.io/medical-imaging-api:v1

# Tell App Service which port your container listens on
az webapp config appsettings set \
  --resource-group $RG --name $APP \
  --settings WEBSITES_PORT=8000

# Hit the live URL
echo "https://$APP.azurewebsites.net/health"
curl  https://$APP.azurewebsites.net/health
```

### Update on every release

```bash
docker build -t medical-imaging-api:v2 .
docker tag  medical-imaging-api:v2   $ACR.azurecr.io/medical-imaging-api:v2
docker push                          $ACR.azurecr.io/medical-imaging-api:v2
az webapp config container set \
  --resource-group $RG --name $APP \
  --container-image-name $ACR.azurecr.io/medical-imaging-api:v2
```

### Tear down (avoid surprise bills)

```bash
az group delete --name $RG --yes --no-wait
```

## 6. Putting it all together — one flow

After CI passes, the human steps to ship a new model are:

```bash
# Local: train + verify
python mlops_pipeline.py --task <task>

# 1. Push code via Git
git add -A && git commit -m "Improve <task> model" && git push

# 2. Push model weights to HF Hub (for Streamlit)
huggingface-cli upload <user>/medical-imaging-models \
    artifacts_<task>/best_model.pt artifacts_<task>/best_model.pt

# 3. Build + push container (for FastAPI deploy)
docker build -t medical-imaging-api:v$(date +%Y%m%d) .
docker tag  medical-imaging-api:v$(date +%Y%m%d) $ACR.azurecr.io/medical-imaging-api:v$(date +%Y%m%d)
docker push $ACR.azurecr.io/medical-imaging-api:v$(date +%Y%m%d)

# 4. Roll the Azure web app
az webapp config container set --resource-group $RG --name $APP \
    --container-image-name $ACR.azurecr.io/medical-imaging-api:v$(date +%Y%m%d)

# 5. Confirm
curl https://$APP.azurewebsites.net/health
```

Three artifact destinations, three CLIs:

- **Code → GitHub** (`git push`)
- **Model weights → Hugging Face Hub** (`huggingface-cli upload`)
- **Container → Azure Container Registry → App Service** (`docker push` + `az webapp`)

Each is independent, so you can iterate on any one without retouching the others.
