# Root Terragrunt config.
# Provider generation is intentionally per-environment so GCP and AWS modules coexist.
# See:
#   environments/gcp/prod/terragrunt.hcl  — generates provider "google"
#   environments/aws/prod/terragrunt.hcl  — generates provider "aws"
