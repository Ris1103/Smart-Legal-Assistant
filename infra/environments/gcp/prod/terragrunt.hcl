# Terragrunt config for the GCP prod environment.
# Run: cd infra/environments/gcp/prod && terragrunt apply

terraform {
  source = "../../../modules/gcp/vm"
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
provider "google" {
  project = var.project_id
  region  = var.region
}
EOF
}

# Remote state — store in GCS (free within the project)
# Create once: gsutil mb -l asia-south1 gs://legal-advisor-tf-state
remote_state {
  backend = "gcs"
  config = {
    bucket = "legal-advisor-tf-state"
    prefix = "prod/vm"
  }
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite"
  }
}

inputs = {
  project_id            = "legal-advisor-prod"   # replace with your GCP project ID
  region                = "asia-south1"
  zone                  = "asia-south1-a"
  instance_name         = "legal-advisor-vm"
  service_account_email = "legal-advisor-sa@legal-advisor-prod.iam.gserviceaccount.com"
}
