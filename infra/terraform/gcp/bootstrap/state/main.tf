terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "Terraform state bucket location"
  type        = string
  default     = "us-central1"
}

resource "google_storage_bucket" "terraform_state" {
  project  = var.project_id
  name     = "${var.project_id}-ecommerce-tfstate"
  location = var.region

  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle {
    prevent_destroy = true
  }

  labels = {
    application = "ecommerce-review-analyzer"
    managed_by  = "terraform"
    purpose     = "terraform-state"
  }
}

output "state_bucket_name" {
  value = google_storage_bucket.terraform_state.name
}
