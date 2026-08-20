resource "google_service_account" "gke_nodes" {
  project      = var.project_id
  account_id   = "ecommerce-gke-nodes"
  display_name = "Ecommerce Review GKE nodes"
}

resource "google_project_iam_member" "node_default_role" {
  project = var.project_id
  role    = "roles/container.defaultNodeServiceAccount"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

resource "google_project_iam_member" "node_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

data "google_project" "current" {
  project_id = var.project_id
}

resource "google_storage_bucket_iam_member" "dvc_puller_object_viewer" {
  bucket = var.dvc_bucket_name
  role   = "roles/storage.objectViewer"

  member = "principal://iam.googleapis.com/projects/${data.google_project.current.number}/locations/global/workloadIdentityPools/${var.project_id}.svc.id.goog/subject/ns/ecommerce-review/sa/dvc-puller"
}



