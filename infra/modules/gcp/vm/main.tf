terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ── VM instance ─────────────────────────────────────────────────────────────
resource "google_compute_instance" "vm" {
  name         = var.instance_name
  machine_type = "e2-micro"   # GCP free-forever tier
  zone         = var.zone

  tags = ["legal-advisor", "http-server", "https-server"]

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = 30   # GB — free tier allows 30 GB
      type  = "pd-standard"
    }
  }

  network_interface {
    network = "default"
    access_config {}   # ephemeral external IP (reserve a static IP if desired)
  }

  metadata = {
    # Cloud-init bootstraps Docker + Docker Compose on first boot
    user-data = file("${path.module}/cloud-init.yaml")
  }

  service_account {
    email  = var.service_account_email
    scopes = ["cloud-platform"]
  }

  lifecycle {
    # Prevent accidental VM recreation — all Docker volume data lives on the boot disk.
    # To replace the VM intentionally: terraform state rm + targeted apply.
    prevent_destroy = true
  }
}

# ── Firewall: allow HTTP + HTTPS inbound ────────────────────────────────────
resource "google_compute_firewall" "allow_web" {
  name    = "${var.instance_name}-allow-web"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server", "https-server"]
}

# ── Static external IP (optional — uncomment when you have a domain) ────────
# resource "google_compute_address" "static_ip" {
#   name   = "${var.instance_name}-ip"
#   region = var.region
# }
