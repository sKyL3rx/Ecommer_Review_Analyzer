output "cluster_name" {
  value = google_container_cluster.dev.name
}

output "cluster_location" {
  value = google_container_cluster.dev.location
}