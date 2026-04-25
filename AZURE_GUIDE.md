# Azure for Beginners — Deploying the Disease Prediction Model

A practical, project-tailored guide. Covers account setup, core concepts, and three concrete ways to deploy the `predict_v3.py` pipeline to Azure, plus shorter overviews of other Azure services you'll meet along the way.

> **Cost warning up front.** Azure charges by the second for most compute. A GPU VM left running overnight by accident can cost $20–$100. Always **stop** or **delete** resources when done, and set a monthly spending cap (see §9).

---

## 1. The 30-second mental model

Azure is Microsoft's cloud. Three things to know before you click anything:

- **Subscription** = the bill-paying account. You have one (free tier or pay-as-you-go).
- **Resource Group** = a folder that holds related resources (VMs, storage, models). Delete the group and everything inside it goes with it — this is how you avoid forgotten charges.
- **Region** = a physical datacenter (e.g. `East US`, `West Europe`). Pick one close to you and keep using it; cross-region traffic costs extra.

Everything else is a **resource** (a VM, a storage account, an ML workspace, etc.) that lives inside a resource group.

---

## 2. One-time setup

### 2.1 Create an account

1. Go to https://azure.microsoft.com/free — sign up with a Microsoft account.
2. Free tier gives you **$200 credit for 30 days** plus always-free services. You need a credit card for identity, but it is not charged unless you explicitly upgrade.

### 2.2 Install the Azure CLI (recommended over clicking in the portal)

```powershell
# Windows (PowerShell as admin)
winget install -e --id Microsoft.AzureCLI
az login
az account show              # verify you're logged into the right subscription
```

Most steps below use the CLI because it's copy-pasteable. The web portal (https://portal.azure.com) does the same things with buttons.

### 2.3 Create a resource group

```bash
az group create --name diseasepred-rg --location eastus
```

Everything in this guide goes into `diseasepred-rg`. When you're done experimenting:

```bash
az group delete --name diseasepred-rg --yes
```

That single command deletes all resources and stops all charges from this project.

---

## 3. Store your model files in Blob Storage

Your trained checkpoint (`resmlp_3c_v3_best.pth`, ~few MB) plus scaler and encoder files need to live somewhere the deployed app can read them. Don't bake them into your container image — they change too often.

```bash
# Create a storage account (name must be globally unique, lowercase, 3-24 chars)
az storage account create \
  --name diseasepredstor01 \
  --resource-group diseasepred-rg \
  --location eastus \
  --sku Standard_LRS

# Create a container (bucket) inside it
az storage container create \
  --account-name diseasepredstor01 \
  --name models

# Upload your artifacts
az storage blob upload-batch \
  --account-name diseasepredstor01 \
  --destination models \
  --source saved_models_3c_v3
```

At inference time the app pulls these down once at startup. Use **Managed Identity** (§8) to authenticate without embedding keys.

---

## 4. Deploying the model — three paths, pick one

### Path A — Azure Container Instances (simplest, 5 minutes)

Best for: quick demos, occasional inference, proving the pipeline works in the cloud.
Not great for: production traffic (no autoscaling, no HTTPS by default).

**Step 1 — Package as a container.** Create `Dockerfile` next to `predict_v3.py`:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.9.1 transformers==4.57.3 \
    scikit-learn joblib pandas pillow flask
COPY predict_v3.py serve.py ./
EXPOSE 8000
CMD ["python", "serve.py"]
```

And a minimal Flask wrapper (`serve.py`):

```python
from flask import Flask, request, jsonify
from predict_v3 import DiseasePredictorV3
import os, tempfile

app = Flask(__name__)
pred = DiseasePredictorV3(
    model_path="/models/resmlp_3c_v3_best.pth",
    scaler_path="/models/scaler_3c_v3.pkl",
    encoder_path="/models/label_encoder_3c_v3.pkl",
    n_tta=3,
)

@app.post("/predict")
def predict():
    f = request.files["image"]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        f.save(tmp.name)
        result = pred.run_image(tmp.name)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

**Step 2 — Build and push to Azure Container Registry:**

```bash
az acr create --resource-group diseasepred-rg --name diseasepredacr --sku Basic
az acr login --name diseasepredacr
docker build -t diseasepredacr.azurecr.io/predictor:v1 .
docker push diseasepredacr.azurecr.io/predictor:v1
```

**Step 3 — Run it:**

```bash
az container create \
  --resource-group diseasepred-rg \
  --name predictor \
  --image diseasepredacr.azurecr.io/predictor:v1 \
  --cpu 2 --memory 4 \
  --registry-login-server diseasepredacr.azurecr.io \
  --dns-name-label diseasepred-demo \
  --ports 8000
```

After a minute: `http://diseasepred-demo.eastus.azurecontainer.io:8000/predict` accepts image uploads.

**Cost:** ~$0.05/hour when running. `az container stop --name predictor --resource-group diseasepred-rg` pauses the bill.

---

### Path B — Azure ML Managed Online Endpoint (production-grade)

Best for: real deployment with autoscaling, HTTPS, versioning, A/B tests.
Worth the extra setup if you're actually shipping this.

**Step 1 — Create an Azure ML workspace:**

```bash
az extension add -n ml
az ml workspace create --name diseasepred-ws --resource-group diseasepred-rg
```

**Step 2 — Register your model** (uploads it to the workspace's model registry):

```bash
az ml model create \
  --name resmlp-3c \
  --version 1 \
  --path saved_models_3c_v3 \
  --workspace-name diseasepred-ws \
  --resource-group diseasepred-rg
```

**Step 3 — Create a scoring script** (`score.py`):

```python
import os, json
from predict_v3 import DiseasePredictorV3

def init():
    global pred
    root = os.environ["AZUREML_MODEL_DIR"]
    pred = DiseasePredictorV3(
        model_path=f"{root}/resmlp_3c_v3_best.pth",
        scaler_path=f"{root}/scaler_3c_v3.pkl",
        encoder_path=f"{root}/label_encoder_3c_v3.pkl",
        n_tta=3,
    )

def run(raw_data):
    # raw_data is a dict with "image_path" or base64-encoded image
    data = json.loads(raw_data)
    return pred.run_image(data["image_path"])
```

**Step 4 — Define the endpoint and deployment** (`endpoint.yml` and `deployment.yml` — the Azure ML docs have templates). Then:

```bash
az ml online-endpoint create -f endpoint.yml
az ml online-deployment create -f deployment.yml --all-traffic
```

You get an HTTPS URL with a bearer token, autoscaling, and a built-in dashboard. **Cost:** small CPU instances start ~$0.10/hour; idle endpoints can scale to zero with the Serverless deployment type.

---

### Path C — Azure Functions (serverless, pay-per-request)

Best for: low-traffic use where you don't want a server sitting idle.
Tradeoffs: cold start (~5–10s for first request after idle), memory capped at 1.5 GB on the consumption plan — fine for the MLP head, tight for DINOv2.

A realistic version keeps DINOv2 running on a Container Instance (Path A) and puts the MLP head behind a Function that calls it. Overkill for a demo; useful if you need per-request billing.

---

## 5. Training on a GPU VM (for future model work)

Since the local CPU run takes 3+ hours for feature extraction, a cheap spot GPU VM pays for itself after the first run.

```bash
# T4 GPU VM (Standard_NC4as_T4_v3) — ~$0.50/hr regular, ~$0.10/hr spot
az vm create \
  --resource-group diseasepred-rg \
  --name trainvm \
  --image Ubuntu2204 \
  --size Standard_NC4as_T4_v3 \
  --priority Spot --eviction-policy Deallocate --max-price -1 \
  --admin-username azureuser \
  --generate-ssh-keys
```

SSH in, install CUDA-enabled PyTorch, rsync your code and dataset, run `train_v3.py`, rsync results back. **Deallocate when done:**

```bash
az vm deallocate --resource-group diseasepred-rg --name trainvm
```

Deallocated VMs stop the compute charge (disk still costs a few cents/day). Starting back up takes ~30 seconds.

For heavier work: Azure ML Compute Clusters auto-scale GPU nodes and tear them down when jobs finish — better than hand-managing VMs.

---

## 6. Other Azure services worth knowing

| Service | What it's for | When to reach for it |
|---|---|---|
| **Azure Blob Storage** | Cheap object storage (models, datasets, backups) | Always — your source of truth for model artifacts. |
| **Azure Key Vault** | Secrets, API keys, certificates | The instant you have any secret you don't want in source. |
| **Azure Container Registry** | Private Docker registry | You're deploying containers. |
| **Azure Monitor / Log Analytics** | Logs and metrics for every resource | Debugging why an endpoint is failing in prod. |
| **Azure DevOps / GitHub Actions** | CI/CD | Automating retrain → register → redeploy on new data. |
| **Azure Cognitive Services** | Pre-built AI APIs (vision, speech, language) | Prototyping features you don't need to train yourself. |
| **Azure OpenAI** | GPT models hosted in your tenant | LLM features with enterprise data residency. |
| **Cosmos DB / Azure SQL** | Managed databases | Storing prediction logs, user accounts, metadata. |
| **Static Web Apps** | Host a frontend (React, etc.) | Pair with a Function or Container backend for a full app. |

---

## 7. A concrete "first week in Azure" plan

1. **Day 1** — Create account, install CLI, make resource group `diseasepred-rg`, upload a model file to Blob Storage.
2. **Day 2** — Work through Path A end to end on a sample input.
3. **Day 3** — Add HTTPS via Azure Front Door or switch to App Service (built-in HTTPS).
4. **Day 4** — Add Application Insights logging; see request traces in the portal.
5. **Day 5** — Provision a spot GPU VM, rerun `train_v3.py` there, compare wall time.
6. **Day 6** — Migrate to Azure ML Managed Endpoint (Path B) for a versioned, scalable deploy.
7. **Day 7** — Delete the resource group. Start again next week knowing exactly what you provisioned.

---

## 8. Authentication without keys (Managed Identity)

Don't put storage account keys in your container. Instead:

```bash
az container create ... --assign-identity
az role assignment create \
  --assignee <container-managed-identity-id> \
  --role "Storage Blob Data Reader" \
  --scope /subscriptions/<sub>/resourceGroups/diseasepred-rg/providers/Microsoft.Storage/storageAccounts/diseasepredstor01
```

Inside the container, use `DefaultAzureCredential` from the `azure-identity` Python package — it picks up the identity automatically. No keys in code, no keys in env vars.

---

## 9. Cost control checklist

- **Set a spending limit** on your subscription: Portal → Cost Management → Budgets → create a monthly budget that emails you at 50% / 90% / 100%.
- **Tag every resource** with `project=diseasepred` so you can filter the cost report.
- **Stop/deallocate** VMs when not in use. Stopping is not enough for VMs — you must **deallocate** to halt compute charges.
- **Use spot VMs** for non-urgent training (70–90% cheaper, can be evicted).
- **Delete the resource group** whenever you're done with a project phase. It's the only reliable "off switch".
- **Check the cost report weekly** for the first month. Surprises usually come from: forgotten VMs, Log Analytics ingestion, outbound bandwidth.

---

## 10. Where to go next

- **Microsoft Learn** (https://learn.microsoft.com) — free structured tutorials. The "AI Engineer" and "Azure Fundamentals (AZ-900)" paths are the shortest route to literacy.
- **Azure Architecture Center** (https://learn.microsoft.com/azure/architecture/) — reference designs for common patterns (ML serving, data pipelines, web apps).
- **Pricing Calculator** (https://azure.microsoft.com/pricing/calculator/) — estimate costs before provisioning.
- **Azure Status** (https://status.azure.com) — check before debugging a regional outage.

For this project specifically, the shortest useful next step is **Path A** in §4 — it gets a real HTTPS URL serving your trained model in under an hour and teaches you the mechanics that every other Azure deployment builds on.
