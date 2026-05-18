# GitHub Actions — Required Secrets

Set these in **GitHub → Settings → Secrets and variables → Actions**.

## Build secrets (used by the `build` job)

| Secret | Where to get it |
|--------|----------------|
| `VITE_API_URL` | Your VM's domain or IP, e.g. `https://api.legaladvisor.in` |
| `VITE_CLERK_PUBLISHABLE_KEY` | Clerk Dashboard → Production instance → API Keys → `pk_live_...` |

## Deploy secrets (used by the `deploy` job)

| Secret | Where to get it |
|--------|----------------|
| `GCP_VM_HOST` | External IP of the e2-micro VM (from `terraform output external_ip`) |
| `GCP_VM_USER` | SSH user on the VM (usually `root` or your IAM username) |
| `GCP_SSH_PRIVATE_KEY` | Private key whose public half is in the VM's `~/.ssh/authorized_keys` |

## On the VM (set in `/etc/environment`)

```bash
GHCR_PAT=<GitHub PAT with read:packages scope>
```

Used by the deploy script to `docker login ghcr.io`.

## Production environment (set in `~/legal-advisor/app/.env.prod` on the VM)

Copy `app/.env.prod.example`, fill in all values, place at `~/legal-advisor/app/.env.prod`.
This file is **never committed** — it lives only on the VM.
