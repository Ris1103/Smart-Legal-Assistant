output "instance_name" {
  value = google_compute_instance.vm.name
}

output "external_ip" {
  description = "Ephemeral external IP of the VM (note: changes on restart)"
  value       = google_compute_instance.vm.network_interface[0].access_config[0].nat_ip
}
