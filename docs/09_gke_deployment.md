# GKE Deployment

This document is the cloud deployment runbook for the current dev setup.

The goal is to go from a fresh GKE environment to the full application stack:

```text
DVC / GCS
   |
   +--> serving parquet files
   +--> summary LoRA adapter
   |
   v
GKE
├── PostgreSQL + PVC
├── Redis
├── FastAPI
├── RQ Worker
│   └── TF-IDF + Logistic Regression sentiment model
├── db-migrate Job
├── data-bootstrap Job
└── NVIDIA L4 GPU node
    └── vLLM
        └── Qwen2.5-3B-Instruct + summary-sft LoRA
```

The validated serving database contains:

```text
products = 94,282
reviews  = 2,103,889
```

This guide matches the current Terraform and Kustomize layout in the repository. It is a dev deployment, so the API is accessed with `kubectl port-forward` and image publishing/deployment is manual.

---

## 1. Prerequisites

Install:

- Google Cloud CLI (`gcloud`)
- Terraform >= 1.6
- Docker
- `kubectl`
- Python environment with DVC + GCS support

Verify:

```bash
gcloud --version
terraform version
docker --version
kubectl version --client
```

Run commands from the repository root.

The following source files are important for the cloud deployment:

```text
infra/terraform/gcp/bootstrap/state/
infra/terraform/gcp/envs/dev/
infra/k8s/base/
infra/k8s/overlays/dev/
infra/k8s/overlays/dev-gpu/
infra/docker/model-puller/Dockerfile
```

Do not commit local Terraform state, plans, or real `terraform.tfvars` files.

---

## 2. Set the environment

Use your own GCP project values here.

```bash
export GCP_PROJECT_ID="YOUR_GCP_PROJECT_ID"
export GCP_REGION="us-central1"
export GCP_ZONE="us-central1-a"

export GKE_CLUSTER="ecommerce-review-dev"
export K8S_NAMESPACE="ecommerce-review"
export AR_REPO="ecommerce-review-analyzer"

export TF_BOOTSTRAP="infra/terraform/gcp/bootstrap/state"
export TF_DEV="infra/terraform/gcp/envs/dev"
export TF_STATE_BUCKET="${GCP_PROJECT_ID}-ecommerce-tfstate"

export DVC_BUCKET="YOUR_DVC_GCS_BUCKET"
export DVC_REMOTE="gcsremote"

export APP_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/ecommerce-review-analyzer"
export DVC_PULLER_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/ecommerce-review-model-puller"
export VLLM_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/vllm-openai"
```

Set the active project/region/zone:

```bash
gcloud config set project "$GCP_PROJECT_ID"
gcloud config set compute/region "$GCP_REGION"
gcloud config set compute/zone "$GCP_ZONE"
```

The current Terraform dev defaults are:

```text
cluster: ecommerce-review-dev
CPU:     e2-standard-2, min 1, max 2
GPU:     g2-standard-4 + NVIDIA L4, min 0, max 2
Spot:    true
```

If your values are different, update your local `terraform.tfvars` before applying Terraform.

---

## 3. Authenticate to GCP

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev"
```

Check the active project:

```bash
gcloud config get-value project
```

---

## 4. Terraform state bucket (first setup only)

The bootstrap Terraform creates the GCS bucket used for Terraform remote state.

If the state bucket already exists and the backend is already configured, skip this section.

Make sure the local bootstrap `terraform.tfvars` contains your project and region:

```hcl
project_id = "YOUR_GCP_PROJECT_ID"
region     = "us-central1"
```

Initialize and apply:

```bash
terraform -chdir="$TF_BOOTSTRAP" init
terraform -chdir="$TF_BOOTSTRAP" validate
terraform -chdir="$TF_BOOTSTRAP" plan -out=tfplan
terraform -chdir="$TF_BOOTSTRAP" apply tfplan
```

Migrate the bootstrap state to GCS:

```bash
terraform \
  -chdir="$TF_BOOTSTRAP" \
  init \
  -migrate-state \
  -backend-config="bucket=${TF_STATE_BUCKET}" \
  -backend-config="prefix=ecommerce-review/bootstrap"
```

Verify:

```bash
terraform -chdir="$TF_BOOTSTRAP" state list
```

Do not destroy the bootstrap state bucket after the dev cluster is deleted. The bucket has Terraform state and versioning enabled.

---

## 5. Check external GCP resources

The current dev Terraform configuration expects two resources to already exist:

1. the Artifact Registry Docker repository,
2. the GCS bucket used by the DVC remote.

Verify Artifact Registry:

```bash
gcloud artifacts repositories describe \
  "$AR_REPO" \
  --location="$GCP_REGION" \
  --project="$GCP_PROJECT_ID"
```

Verify the DVC bucket:

```bash
gcloud storage buckets describe "gs://${DVC_BUCKET}"
```

Terraform reads the Artifact Registry repository as a data source and grants the Kubernetes DVC puller read access to the existing DVC bucket.

---

## 6. Create the dev GKE infrastructure

A local `infra/terraform/gcp/envs/dev/terraform.tfvars` should contain values similar to:

```hcl
project_id = "YOUR_GCP_PROJECT_ID"
region     = "us-central1"
zone       = "us-central1-a"

cluster_name = "ecommerce-review-dev"

cpu_machine_type = "e2-standard-2"
cpu_min_nodes    = 1
cpu_max_nodes    = 2

gpu_machine_type = "g2-standard-4"
gpu_type         = "nvidia-l4"
gpu_max_nodes    = 2
gpu_spot         = true

dvc_bucket_name = "YOUR_DVC_GCS_BUCKET"
```

Initialize the dev backend:

```bash
terraform \
  -chdir="$TF_DEV" \
  init \
  -reconfigure \
  -backend-config="bucket=${TF_STATE_BUCKET}" \
  -backend-config="prefix=ecommerce-review/dev"
```

Format and validate:

```bash
terraform fmt -recursive infra/terraform/gcp
terraform -chdir="$TF_DEV" validate
```

Create the plan:

```bash
rm -f "${TF_DEV}/tfplan"

terraform \
  -chdir="$TF_DEV" \
  plan \
  -out=tfplan
```

Apply:

```bash
terraform \
  -chdir="$TF_DEV" \
  apply tfplan
```

Terraform creates the custom VPC/subnet, Cloud NAT, private GKE cluster, CPU node pool, GPU node pool, node service account, and IAM bindings used by the deployment.

Connect kubectl to the cluster:

```bash
gcloud container clusters get-credentials \
  "$GKE_CLUSTER" \
  --zone="$GCP_ZONE" \
  --project="$GCP_PROJECT_ID"
```

Verify:

```bash
kubectl get nodes -o wide
```

At this point the CPU node should be available. The GPU pool has a minimum of zero, so an L4 node does not need to exist until the vLLM pod needs it.

---

## 7. Verify Workload Identity

Cluster workload pool:

```bash
gcloud container clusters describe \
  "$GKE_CLUSTER" \
  --zone="$GCP_ZONE" \
  --project="$GCP_PROJECT_ID" \
  --format='value(workloadIdentityConfig.workloadPool)'
```

Expected format:

```text
YOUR_GCP_PROJECT_ID.svc.id.goog
```

CPU node pool metadata mode:

```bash
gcloud container node-pools describe \
  cpu-pool \
  --cluster="$GKE_CLUSTER" \
  --zone="$GCP_ZONE" \
  --project="$GCP_PROJECT_ID" \
  --format='value(config.workloadMetadataConfig.mode)'
```

Expected:

```text
GKE_METADATA
```

The same metadata mode is configured on the GPU node pool.

---

## 8. Prepare DVC artifacts

The cloud jobs use the DVC remote named `gcsremote`.

Verify the local DVC setup:

```bash
.venv/bin/python -m dvc remote list
```

If the GCS plugin is missing:

```bash
.venv/bin/python -m pip install "dvc[gs]==3.67.1"
```

### Serving data

Reproduce the serving parquet files if needed:

```bash
make repro-serving
```

Push them to the configured DVC remote:

```bash
.venv/bin/python -m dvc push \
  -r "$DVC_REMOTE" \
  data/serving/appliances_demo_catalog.parquet \
  data/serving/appliances_demo_reviews.parquet
```

Verify remote status:

```bash
.venv/bin/python -m dvc status \
  -c \
  -r "$DVC_REMOTE" \
  data/serving/appliances_demo_catalog.parquet \
  data/serving/appliances_demo_reviews.parquet
```

### Final models

Check the final model pointers:

```bash
.venv/bin/python -m dvc status \
  -c \
  -r "$DVC_REMOTE" \
  artifacts/models/final/sentiment.dvc \
  artifacts/models/final/summary_sft.dvc
```

Push if needed:

```bash
.venv/bin/python -m dvc push \
  -r "$DVC_REMOTE" \
  artifacts/models/final/sentiment.dvc \
  artifacts/models/final/summary_sft.dvc
```

The application image needs the two small sentiment files in its Docker build context:

```text
artifacts/models/final/sentiment/tfidf.joblib
artifacts/models/final/sentiment/sentiment_model.joblib
```

On the current cloud-serving branch these files can be tracked directly in Git because they are small. If they are not present locally, restore the sentiment artifact with DVC before building:

```bash
.venv/bin/python -m dvc pull \
  -r "$DVC_REMOTE" \
  artifacts/models/final/sentiment.dvc
```

Check:

```bash
ls -lh \
  artifacts/models/final/sentiment/tfidf.joblib \
  artifacts/models/final/sentiment/sentiment_model.joblib
```

The larger summary LoRA adapter stays in DVC/GCS and is pulled by the vLLM init container.

---

## 9. Build and push the application image

Create a unique tag:

```bash
export APP_TAG="app-$(git rev-parse --short HEAD)-$(date +%Y%m%d%H%M%S)"
```

Build:

```bash
docker build \
  -t "${APP_IMAGE}:${APP_TAG}" \
  .
```

Verify that the sentiment artifacts are inside the image:

```bash
docker run --rm \
  "${APP_IMAGE}:${APP_TAG}" \
  ls -lh \
  /app/artifacts/models/tfidf.joblib \
  /app/artifacts/models/sentiment_model.joblib
```

Verify they can be loaded:

```bash
docker run --rm \
  "${APP_IMAGE}:${APP_TAG}" \
  python -c '
import joblib
joblib.load("/app/artifacts/models/tfidf.joblib")
joblib.load("/app/artifacts/models/sentiment_model.joblib")
print("sentiment artifacts: OK")
'
```

Push:

```bash
docker push "${APP_IMAGE}:${APP_TAG}"
```

Update the application `newTag` in:

```text
infra/k8s/overlays/dev/kustomization.yaml
```

This push is a manual deployment step. The current GitHub Actions workflow does not publish cloud images.

---

## 10. Build and push the DVC puller image

The DVC puller image is used by Kubernetes init containers to pull serving data and the LoRA adapter from GCS.

Create a tag:

```bash
export DVC_PULLER_TAG="dvc-$(git rev-parse --short HEAD)-$(date +%Y%m%d%H%M%S)"
```

Build:

```bash
docker build \
  -f infra/docker/model-puller/Dockerfile \
  -t "${DVC_PULLER_IMAGE}:${DVC_PULLER_TAG}" \
  .
```

Verify DVC inside the image:

```bash
docker run --rm \
  "${DVC_PULLER_IMAGE}:${DVC_PULLER_TAG}" \
  sh -c '
    cat .dvc/config.local
    dvc remote list
    ls -lh \
      artifacts/models/final/sentiment.dvc \
      artifacts/models/final/summary_sft.dvc
  '
```

Push:

```bash
docker push "${DVC_PULLER_IMAGE}:${DVC_PULLER_TAG}"
```

Update the DVC puller tag in both overlays:

```text
infra/k8s/overlays/dev/kustomization.yaml
infra/k8s/overlays/dev-gpu/kustomization.yaml
```

---

## 11. Mirror the vLLM image to Artifact Registry

The current GPU manifest uses vLLM `v0.7.3`.

Pull:

```bash
docker pull vllm/vllm-openai:v0.7.3
```

Tag it for Artifact Registry:

```bash
docker tag \
  vllm/vllm-openai:v0.7.3 \
  "${VLLM_IMAGE}:v0.7.3"
```

Push:

```bash
docker push "${VLLM_IMAGE}:v0.7.3"
```

The `dev-gpu` overlay should point `vllm-serving` to this image/tag.

---

## 12. Create the namespace and DVC puller service account

The base Kustomization does not create the `dvc-puller` service account, so apply it before the bootstrap and vLLM pods.

```bash
kubectl apply -f infra/k8s/base/namespace.yaml
kubectl apply -f infra/k8s/base/serviceaccounts/dvc-puller.yaml
```

The Terraform IAM binding gives this Kubernetes principal `roles/storage.objectViewer` on the DVC bucket.

Enable uniform bucket-level access if it is not already enabled:

```bash
gcloud storage buckets update \
  "gs://${DVC_BUCKET}" \
  --uniform-bucket-level-access
```

You can verify the principal in the bucket IAM policy:

```bash
export PROJECT_NUMBER="$(
  gcloud projects describe "$GCP_PROJECT_ID" \
    --format='value(projectNumber)'
)"
```

```bash
export DVC_PRINCIPAL="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${GCP_PROJECT_ID}.svc.id.goog/subject/ns/ecommerce-review/sa/dvc-puller"
```

```bash
gcloud storage buckets get-iam-policy \
  "gs://${DVC_BUCKET}" \
  --format=json |
grep -B8 -A5 -F "$DVC_PRINCIPAL"
```

---

## 13. Create the Kubernetes Secret

Do not commit the real secret YAML.

The repository only contains `app-env-secret.example.yaml` as a template.

Create the real secret from local values:

```bash
read -s -p "Postgres password: " POSTGRES_PASSWORD
echo
```

URL-encode the password for `DATABASE_URL`:

```bash
export POSTGRES_PASSWORD_URLENCODED="$(
  POSTGRES_PASSWORD="$POSTGRES_PASSWORD" python3 - <<'PY'
import os
from urllib.parse import quote
print(quote(os.environ["POSTGRES_PASSWORD"], safe=""))
PY
)"
```

Create a temporary env file:

```bash
cat > /tmp/ecommerce-review-secret.env <<EOF2
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
DATABASE_URL=postgresql+psycopg://ecom:${POSTGRES_PASSWORD_URLENCODED}@postgres:5432/ecom_review
VLLM_API_KEY=demo-key
EOF2
```

Create/update the Kubernetes Secret:

```bash
kubectl create secret generic ecommerce-review-env \
  --namespace=ecommerce-review \
  --from-env-file=/tmp/ecommerce-review-secret.env \
  --dry-run=client \
  -o yaml |
kubectl apply -f -
```

Clean the temporary values:

```bash
rm -f /tmp/ecommerce-review-secret.env
unset POSTGRES_PASSWORD
unset POSTGRES_PASSWORD_URLENCODED
```

---

## 14. Deploy the CPU application stack

The CPU overlay contains:

```text
PostgreSQL + PVC
Redis
Alembic migration Job
data-bootstrap Job
FastAPI
RQ Worker
```

It does not include the vLLM deployment.

Render the overlay first:

```bash
kubectl kustomize \
  infra/k8s/overlays/dev \
  > /tmp/ecommerce-review-dev.yaml
```

Check that it does not contain GPU/vLLM resources:

```bash
grep -nE \
  'nvidia.com/gpu|Qwen/Qwen|name: vllm' \
  /tmp/ecommerce-review-dev.yaml || true
```

When rerunning on an existing cluster, delete completed Jobs first:

```bash
kubectl delete job \
  db-migrate \
  data-bootstrap \
  -n ecommerce-review \
  --ignore-not-found
```

Apply:

```bash
kubectl apply -k infra/k8s/overlays/dev
```

Watch pods:

```bash
kubectl get pods -n ecommerce-review -w
```

Wait for migrations:

```bash
kubectl wait \
  --for=condition=complete \
  job/db-migrate \
  -n ecommerce-review \
  --timeout=10m
```

Wait for the full data bootstrap:

```bash
kubectl wait \
  --for=condition=complete \
  job/data-bootstrap \
  -n ecommerce-review \
  --timeout=60m
```

The bootstrap flow is:

```text
wait for Alembic schema
  -> DVC init container pulls both serving parquet files from GCS
  -> application loader inserts products/reviews into PostgreSQL
```

---

## 15. Validate the database

Check row counts:

```bash
kubectl exec \
  -n ecommerce-review \
  deployment/postgres \
  -- psql \
  -U ecom \
  -d ecom_review \
  -c "
SELECT
  (SELECT COUNT(*) FROM products) AS products,
  (SELECT COUNT(*) FROM reviews) AS reviews,
  (SELECT COUNT(*) FROM product_insights) AS insights,
  (SELECT COUNT(*) FROM insight_jobs) AS jobs;
"
```

The validated serving bootstrap produced:

```text
products = 94282
reviews  = 2103889
```

---

## 16. CPU/API smoke test

The current API Service is `ClusterIP`, so use port-forwarding for the dev test.

Terminal A:

```bash
kubectl port-forward \
  -n ecommerce-review \
  service/api \
  8000:8000
```

Terminal B:

```bash
make smoke \
  API_BASE_URL=http://127.0.0.1:8000
```

You can also check:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS "http://127.0.0.1:8000/products?limit=5&offset=0"
```

---

## 17. Deploy vLLM on the L4 GPU pool

The GPU overlay only contains the model-serving resources.

Render:

```bash
kubectl kustomize \
  infra/k8s/overlays/dev-gpu \
  > /tmp/ecommerce-review-dev-gpu.yaml
```

Apply:

```bash
kubectl apply -k infra/k8s/overlays/dev-gpu
```

Watch the vLLM pod:

```bash
kubectl get pods \
  -n ecommerce-review \
  -l app=vllm \
  -w
```

When the vLLM pod is pending, GKE can scale the GPU node pool from zero and create a `g2-standard-4` node with one NVIDIA L4, subject to zonal Spot capacity.

Get the latest vLLM pod name:

```bash
export VLLM_POD="$(
  kubectl get pods \
    -n ecommerce-review \
    -l app=vllm \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{.items[-1:].metadata.name}'
)"
```

Check the DVC init container:

```bash
kubectl logs \
  "$VLLM_POD" \
  -n ecommerce-review \
  -c dvc-pull-lora-adapter
```

The init container pulls:

```text
artifacts/models/final/summary_sft.dvc
```

from the GCS DVC remote into the shared LoRA volume.

Check vLLM startup:

```bash
kubectl logs \
  "$VLLM_POD" \
  -n ecommerce-review \
  -c vllm \
  --tail=200
```

The current deployment runs:

```text
Qwen/Qwen2.5-3B-Instruct
served model name: Qwen/Qwen2.5-3B-Instruct
LoRA adapter: summary-sft
max LoRA rank: 16
GPU memory utilization: 0.85
max model length: 4096
```

Wait until the pod is `1/1 Running` and the readiness probe succeeds.

---

## 18. Test vLLM directly

Terminal A:

```bash
kubectl port-forward \
  -n ecommerce-review \
  service/vllm \
  8001:8000
```

Health:

```bash
curl -i http://127.0.0.1:8001/health
```

List models:

```bash
curl -fsS \
  http://127.0.0.1:8001/v1/models \
  -H 'Authorization: Bearer demo-key' |
python3 -m json.tool
```

The response should include the base model and the `summary-sft` LoRA adapter.

Test the adapter:

```bash
curl -fsS \
  http://127.0.0.1:8001/v1/chat/completions \
  -H 'Authorization: Bearer demo-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "summary-sft",
    "messages": [
      {
        "role": "user",
        "content": "Summarize the main negative customer feedback: The product is noisy, difficult to clean, and several customers report that it stopped working early."
      }
    ],
    "temperature": 0.0,
    "max_tokens": 128
  }' |
python3 -m json.tool
```

A successful request returns HTTP 200 and a generated `choices[0].message.content`.

---

## 19. Verify real worker inference

The current cloud ConfigMap should use:

```yaml
USE_FAKE_SUMMARIZER: "false"
USE_FAKE_SENTIMENT: "false"
VLLM_BASE_URL: "http://vllm:8000/v1"
SUMMARY_MODEL_NAME: "summary-sft"
```

Apply the ConfigMap if you changed it:

```bash
kubectl apply -f infra/k8s/base/configmap.yaml
```

Restart the worker so it receives the latest settings:

```bash
kubectl rollout restart \
  deployment/worker \
  -n ecommerce-review
```

Wait:

```bash
kubectl rollout status \
  deployment/worker \
  -n ecommerce-review \
  --timeout=5m
```

Check the relevant environment variables:

```bash
kubectl exec \
  -n ecommerce-review \
  deployment/worker \
  -- printenv |
grep -E \
  'USE_FAKE_SUMMARIZER|USE_FAKE_SENTIMENT|VLLM_BASE_URL|SUMMARY_MODEL'
```

Check the sentiment files packaged in the application image:

```bash
kubectl exec \
  -n ecommerce-review \
  deployment/worker \
  -- ls -lh \
  /app/artifacts/models/tfidf.joblib \
  /app/artifacts/models/sentiment_model.joblib
```

Test worker-to-vLLM connectivity:

```bash
kubectl exec \
  -n ecommerce-review \
  deployment/worker \
  -- python -c '
import urllib.request
with urllib.request.urlopen("http://vllm:8000/health", timeout=10) as r:
    print("vLLM status:", r.status)
'
```

Expected:

```text
vLLM status: 200
```

---

## 20. End-to-end insight test

Keep the API port-forward running:

```bash
kubectl port-forward \
  -n ecommerce-review \
  service/api \
  8000:8000
```

Create an insight job for a valid product ID:

```bash
curl -fsS -X POST \
  http://127.0.0.1:8000/products/0967805929/insights/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "max_reviews": 50,
    "representative_k": 3,
    "regenerate": true
  }' |
python3 -m json.tool
```

Copy the returned `job_id` and poll:

```bash
curl -fsS \
  http://127.0.0.1:8000/jobs/JOB_ID |
python3 -m json.tool
```

When the job finishes, read the saved insight:

```bash
curl -fsS \
  http://127.0.0.1:8000/products/0967805929/insights |
python3 -m json.tool
```

The runtime path is:

```text
FastAPI
  -> Redis/RQ
  -> RQ Worker
  -> PostgreSQL reviews
  -> TF-IDF sentiment model if needed
  -> representative review selection
  -> vLLM summary-sft
  -> PostgreSQL insight
  -> Redis cache
```

---

## 21. Run the serving benchmark on GKE

With API port-forwarding active:

```bash
make benchmark-local \
  API_BASE_URL=http://127.0.0.1:8000
```

If the run completes successfully, copy the result to a cloud-specific filename so it is not confused with local Docker results:

```bash
cp \
  artifacts/benchmarks/local_serving_benchmark.json \
  artifacts/benchmarks/gke_gpu_vllm_lora_benchmark.json
```

For the current benchmark settings, a healthy async run should finish all 10 submitted jobs:

```text
job_count      = 10
finished_count = 10
success_rate   = 1.0
```

Do not commit a benchmark result if the API or worker died during the run. Check the JSON for connection errors and `submit_failed` statuses first.

---

## 22. Final validation

Check the namespace:

```bash
kubectl get pods -n ecommerce-review
```

Expected steady state:

```text
api       Running
postgres  Running
redis     Running
worker    Running
vllm      Running

db-migrate       Completed
data-bootstrap   Completed
```

Check database state:

```bash
kubectl exec \
  -n ecommerce-review \
  deployment/postgres \
  -- psql \
  -U ecom \
  -d ecom_review \
  -c "
SELECT
  (SELECT COUNT(*) FROM products) AS products,
  (SELECT COUNT(*) FROM reviews) AS reviews,
  (SELECT COUNT(*) FROM product_insights) AS insights,
  (SELECT COUNT(*) FROM insight_jobs) AS jobs;
"
```

After a real insight test, expected values are:

```text
products = 94282
reviews  = 2103889
insights > 0
jobs     > 0
```

---

## 23. CI vs cloud deployment

The GitHub Actions workflow is CI only.

It runs:

```text
Ruff
-> unit tests
-> integration tests with PostgreSQL + Redis + Alembic
-> Docker build validation
```

It does not authenticate to GCP, push images, or run `kubectl apply`.

The current cloud release flow in this document is manual:

```text
build application / DVC puller images
-> push Artifact Registry
-> update Kustomize tags
-> kubectl apply
-> validate GKE
```

This is intentional in the current version of the project and should not be described as automated CD.

---

## 24. Destroy the dev infrastructure

Create a destroy plan:

```bash
terraform \
  -chdir="$TF_DEV" \
  plan \
  -destroy \
  -out=destroy.tfplan
```

Review it:

```bash
terraform \
  -chdir="$TF_DEV" \
  show destroy.tfplan
```

Apply:

```bash
terraform \
  -chdir="$TF_DEV" \
  apply destroy.tfplan
```

Verify that the dev cluster is gone:

```bash
gcloud container clusters list \
  --project="$GCP_PROJECT_ID"
```

Do not destroy the Terraform bootstrap state bucket unless you intentionally want to remove the remote state setup.

The DVC GCS bucket and its artifacts are also separate from the dev cluster lifecycle.

PostgreSQL is currently in-cluster, so its PVC/data is not treated as permanent production storage. A fresh deployment loads the serving dataset again through the bootstrap job.

---

## 25. Fresh redeploy flow

After destroying the dev cluster, the normal next deployment starts at the dev Terraform step:

```text
Terraform apply
-> connect kubectl
-> verify DVC artifacts
-> build/push application image
-> build/push DVC puller image
-> mirror/push vLLM image if needed
-> namespace + dvc-puller service account
-> Kubernetes Secret
-> CPU Kustomize overlay
-> Alembic + full data bootstrap
-> GPU Kustomize overlay
-> vLLM validation
-> end-to-end insight test
-> benchmark
-> Terraform destroy when finished
```

---

### API is not reachable from the internet

The API Service is `ClusterIP` and the dev guide uses `kubectl port-forward`.

There is no ingress/load balancer configuration in the current repository.
