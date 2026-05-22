provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_container_cluster" "dev"{
    name = var.cluster_name
    location = var.zone

    remove_default_node_pool = true
    initial_node_count       = 1

    deletion_protection = false

    network    = "default"
    subnetwork = "default"
}

resource "google_container_node_pool" "cpu_pool" {  
    name       = "cpu-pool"
    location   = var.zone

    cluster    = google_container_cluster.dev.name
    node_count = 1

    node_config {
        machine_type = "e2-standard-2"
        disk_size_gb = 30

        oauth_scopes = [
            "https://www.googleapis.com/auth/cloud-platform"
        ]

        labels = {
            workload = "cpu"
        }
    }

    autoscaling {
    min_node_count = 1
    max_node_count = 2
  }
  
}