provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_container_cluster" "dev" {
  project  = var.project_id
  name     = var.cluster_name
  location = var.zone

  remove_default_node_pool = true
  initial_node_count       = 1

  deletion_protection   = false
  enable_shielded_nodes = true
  networking_mode       = "VPC_NATIVE"

  network    = google_compute_network.main.id
  subnetwork = google_compute_subnetwork.gke.id

  release_channel {
    channel = "REGULAR"
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = "gke-pods"
    services_secondary_range_name = "gke-services"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  gateway_api_config {
    channel = "CHANNEL_STANDARD"
  }

  logging_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS"
    ]
  }

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS"
    ]

    managed_prometheus {
      enabled = true
    }
  }

  depends_on = [
    google_compute_router_nat.main
  ]
}

resource "google_container_node_pool" "cpu_pool" {
  project  = var.project_id
  name     = "cpu-pool"
  cluster  = google_container_cluster.dev.name
  location = var.zone

  initial_node_count = 1

  autoscaling {
    min_node_count = var.cpu_min_nodes
    max_node_count = var.cpu_max_nodes
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }

  node_config {
    machine_type = var.cpu_machine_type
    image_type   = "COS_CONTAINERD"
    disk_size_gb = 30
    disk_type    = "pd-balanced"

    service_account = google_service_account.gke_nodes.email

    labels = {
      workload = "cpu"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
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
