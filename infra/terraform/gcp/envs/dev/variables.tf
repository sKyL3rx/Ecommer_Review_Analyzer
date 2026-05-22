variable "project_id"{
    type        = string
    description = "GCP project ID"
}

variable "region"{
    type        = string
    description = "GCP Region"
    default     = "us-central1"
}

variable "zone"{
    type        = string
    description = "GCP zone"
    default     = "us-central1-a"
}

variable "cluster_name" {
  type        = string
  description = "GKE cluster name"
  default     = "ecommerce-review-dev"
}