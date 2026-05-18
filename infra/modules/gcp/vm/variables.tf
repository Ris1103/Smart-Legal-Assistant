variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "asia-south1"   # closest to India
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "asia-south1-a"
}

variable "instance_name" {
  description = "Name for the VM instance"
  type        = string
  default     = "legal-advisor-vm"
}

variable "service_account_email" {
  description = "Service account email for the VM"
  type        = string
}
