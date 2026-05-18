variable "instance_name" {
  description = "Name tag for the EC2 instance and all related resources"
  type        = string
  default     = "legal-advisor-vm"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"   # Mumbai — matches target users (India)
}

variable "ami_id" {
  description = <<-EOT
    Ubuntu 24.04 LTS AMI ID for the chosen region.
    Look up the current AMI:
      aws ec2 describe-images \
        --owners 099720109477 \
        --filters 'Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*' \
        --query 'sort_by(Images,&CreationDate)[-1].ImageId' \
        --region ap-south-1
  EOT
  type        = string
}

variable "vpc_id" {
  description = "VPC to launch the instance in. Use the default VPC if you don't have a custom one: aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --region ap-south-1"
  type        = string
}

variable "subnet_id" {
  description = "Public subnet ID inside the VPC. Must be attached to an Internet Gateway."
  type        = string
}

variable "ssh_public_key" {
  description = "SSH public key content (e.g. contents of ~/.ssh/id_ed25519.pub) — uploaded as an AWS key pair"
  type        = string
  sensitive   = true
}

variable "ssh_allowed_cidrs" {
  description = "CIDRs allowed to reach SSH (port 22). Tighten to your IP or GitHub Actions ranges before production."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
