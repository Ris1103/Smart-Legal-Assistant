# Terragrunt config for the AWS prod environment.
# Run: cd infra/environments/aws/prod && terragrunt apply
#
# One-time pre-requisites (run from your workstation):
#   1. Create S3 state bucket:
#        aws s3 mb s3://legal-advisor-tf-state-aws --region ap-south-1
#        aws s3api put-bucket-versioning --bucket legal-advisor-tf-state-aws --versioning-configuration Status=Enabled
#   2. Create DynamoDB lock table:
#        aws dynamodb create-table \
#          --table-name legal-advisor-tf-lock \
#          --attribute-definitions AttributeName=LockID,AttributeType=S \
#          --key-schema AttributeName=LockID,KeyType=HASH \
#          --billing-mode PAY_PER_REQUEST \
#          --region ap-south-1
#   3. Fill in vpc_id, subnet_id, ami_id, and ssh_public_key below.

terraform {
  source = "../../../modules/aws/ec2"
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
provider "aws" {
  region = var.region
}
EOF
}

# Remote state — S3 + DynamoDB locking
remote_state {
  backend = "s3"
  config = {
    bucket         = "legal-advisor-tf-state-aws"
    key            = "prod/ec2/terraform.tfstate"
    region         = "ap-south-1"
    encrypt        = true
    dynamodb_table = "legal-advisor-tf-lock"
  }
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite"
  }
}

inputs = {
  instance_name = "legal-advisor-vm"
  region        = "ap-south-1"   # Mumbai

  # Ubuntu 24.04 LTS ap-south-1 — verify the latest before apply:
  #   aws ec2 describe-images --owners 099720109477 \
  #     --filters 'Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*' \
  #     --query 'sort_by(Images,&CreationDate)[-1].ImageId' --region ap-south-1
  ami_id = "ami-REPLACE"

  # Default VPC: aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  #   --query 'Vpcs[0].VpcId' --region ap-south-1
  vpc_id    = "vpc-REPLACE"
  subnet_id = "subnet-REPLACE"   # any public subnet in the default VPC

  ssh_public_key    = "ssh-ed25519 REPLACE"   # contents of ~/.ssh/id_ed25519.pub
  ssh_allowed_cidrs = ["0.0.0.0/0"]           # tighten to your IP before production
}
