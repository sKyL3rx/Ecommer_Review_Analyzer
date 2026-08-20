resource "google_container_node_pool" "gpu_l4_pool" {
  project  = var.project_id
  name     = "gpu-l4-pool"
  cluster  = google_container_cluster.dev.name
  location = var.zone

  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = var.gpu_max_nodes
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = 0
    max_unavailable = 1
  }

  node_config {
    machine_type = var.gpu_machine_type
    image_type   = "COS_CONTAINERD"
    disk_size_gb = 100
    disk_type    = "pd-balanced"
    spot         = var.gpu_spot

    service_account = google_service_account.gke_nodes.email

    labels = {
      workload    = "gpu"
      accelerator = var.gpu_type
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    guest_accelerator {
      type  = var.gpu_type
      count = 1

      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  depends_on = [
    google_project_iam_member.node_default_role,
    google_project_iam_member.node_artifact_reader
  ]
}
