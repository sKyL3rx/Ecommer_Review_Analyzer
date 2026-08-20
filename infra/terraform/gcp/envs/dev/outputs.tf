output "cluster_name" {
  value = google_container_cluster.dev.name
}

output "cluster_location" {
  value = google_container_cluster.dev.location
}

output "network_name" {
  value = google_compute_network.main.name
}

output "subnetwork_name" {
  value = google_compute_subnetwork.gke.name
}

output "artifact_registry_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${data.google_artifact_registry_repository.docker.repository_id}"
}

output "node_service_account" {
  value = google_service_account.gke_nodes.email
}
