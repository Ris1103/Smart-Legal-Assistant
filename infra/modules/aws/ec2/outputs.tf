output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.vm.id
}

output "elastic_ip" {
  description = "Static public IP — point your DNS A record here"
  value       = aws_eip.vm.public_ip
}

output "public_dns" {
  description = "AWS-assigned public DNS for the Elastic IP"
  value       = aws_eip.vm.public_dns
}
