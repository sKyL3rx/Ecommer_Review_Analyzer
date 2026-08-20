variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Primary GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zonal GKE location"
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "ecommerce-review-dev"
}

variable "network_name" {
  description = "Custom VPC network name"
  type        = string
  default     = "ecommerce-review-dev"
}

variable "subnetwork_name" {
  description = "GKE subnetwork name"
  type        = string
  default     = "ecommerce-review-dev-gke"
}

variable "artifact_repository_name" {
  description = "Artifact Registry Docker repository name"
  type        = string
  default     = "ecommerce-review-analyzer"
}

variable "cpu_machine_type" {
  description = "CPU node machine type"
  type        = string
  default     = "e2-standard-2"
}

variable "cpu_min_nodes" {
  description = "Minimum CPU nodes"
  type        = number
  default     = 1
}

variable "cpu_max_nodes" {
  description = "Maximum CPU nodes"
  type        = number
  default     = 2

  validation {
    condition     = var.cpu_max_nodes >= 1 && var.cpu_max_nodes <= 2
    error_message = "cpu_max_nodes must be between 1 and 2."
  }
}

variable "gpu_machine_type" {
  description = "GPU node machine type"
  type        = string
  default     = "g2-standard-4"
}

variable "gpu_type" {
  description = "GPU accelerator type"
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_max_nodes" {
  description = "Hard maximum number of GPU nodes"
  type        = number
  default     = 2

  validation {
    condition     = var.gpu_max_nodes >= 0 && var.gpu_max_nodes <= 2
    error_message = "gpu_max_nodes must be between 0 and 2."
  }
}

variable "gpu_spot" {
  description = "Use Spot VMs for GPU nodes"
  type        = bool
  default     = true
}

variable "dvc_bucket_name" {
  description = "Existing GCS bucket used by DVC"
  type        = string
  default     = "ecommerce-review-dvc-vaulted-blend-493604-v2"
}