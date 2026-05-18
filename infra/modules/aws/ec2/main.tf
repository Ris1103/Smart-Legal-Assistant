terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ── Key pair ──────────────────────────────────────────────────────────────────
resource "aws_key_pair" "deploy" {
  key_name   = "${var.instance_name}-key"
  public_key = var.ssh_public_key
}

# ── Security group ────────────────────────────────────────────────────────────
resource "aws_security_group" "vm" {
  name        = "${var.instance_name}-sg"
  description = "Legal Advisor VM — HTTP, HTTPS, SSH"
  vpc_id      = var.vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_allowed_cidrs
  }

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.instance_name}-sg" }
}

# ── IAM role + instance profile (for CloudWatch Agent) ───────────────────────
resource "aws_iam_role" "vm" {
  name = "${var.instance_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "cloudwatch" {
  role       = aws_iam_role.vm.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_instance_profile" "vm" {
  name = "${var.instance_name}-profile"
  role = aws_iam_role.vm.name
}

# ── EC2 instance ──────────────────────────────────────────────────────────────
resource "aws_instance" "vm" {
  ami                    = var.ami_id           # Ubuntu 24.04 LTS — see variables.tf for lookup command
  instance_type          = "t2.micro"           # AWS free tier: 750 h/month for 12 months
  key_name               = aws_key_pair.deploy.key_name
  vpc_security_group_ids = [aws_security_group.vm.id]
  subnet_id              = var.subnet_id
  iam_instance_profile   = aws_iam_instance_profile.vm.name

  user_data = file("${path.module}/cloud-init.yaml")

  root_block_device {
    volume_size = 30      # GB — free tier includes 30 GB gp3
    volume_type = "gp3"
    encrypted   = true
    tags        = { Name = "${var.instance_name}-disk" }
  }

  tags = { Name = var.instance_name }

  lifecycle {
    # Changing AMI or user_data would recreate the instance and wipe all Docker volume data.
    # Update these only intentionally via a targeted apply.
    prevent_destroy = true
    ignore_changes  = [user_data, ami]
  }
}

# ── Elastic IP (stable address for DNS) ───────────────────────────────────────
resource "aws_eip" "vm" {
  instance = aws_instance.vm.id
  domain   = "vpc"
  tags     = { Name = "${var.instance_name}-eip" }
}
