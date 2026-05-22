# Artifact Registry

This project uses Google Artifact Registry to store the Docker image used by both the FastAPI and the RQ Worker deployment.

The API and worker share the same image but run different commands.

## Build and push to Google Artifact Registry

```bash
export GCP_PROJECT_ID=YOUR_PROJECT_ID
export GCP_REGION=us-central1
export IMAGE_URI=$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/ecommerce-review-analyzer/ecommerce-review-analyzer:local

gcloud auth configure-docker $GCP_REGION-docker.pkg.dev

docker build -t ecommerce-review-analyzer:local .
docker tag ecommerce-review-analyzer:local $IMAGE_URI
docker push $IMAGE_URI
