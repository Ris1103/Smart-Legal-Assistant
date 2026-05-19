# Deployment Issues Log

Chronological record of deployment problems, fixes applied, and side-effects of those fixes.

---

## 1. GHCR Image Name Rejection (Uppercase)

**Problem:** GitHub Container Registry rejected the image push because the repository name contained uppercase letters (`Ris1103` → rejected by GHCR which requires all-lowercase names).

**Fix:** Hardcoded `ghcr.io/ris1103/legal-advisor-api` (all lowercase) in `.github/workflows/deploy.yml` instead of deriving it from `${{ github.repository }}`.

**Consequence:** Image name is now hardcoded in the workflow — if the repo is ever renamed or transferred, this must be updated manually.

---

## 2. TypeScript `import.meta.env` Errors

**Problem:** Frontend Vite build failed because `import.meta.env` references weren't recognized in the TypeScript config.

**Fix:** Updated `tsconfig.json` / Vite config to include the correct env type definitions.

**Consequence:** None — standard Vite fix.

---

## 3. `memswap_limit` Under `deploy.resources`

**Problem:** `docker-compose.yml` placed `mem_limit` and `memswap_limit` under `deploy.resources`, which is only respected by Docker Swarm, not plain `docker compose`. The e2-micro (1 GB RAM) had no memory cap on containers and was OOMing.

**Fix:** Moved `mem_limit` and `memswap_limit` to top-level service keys (Compose v2 style), which are honoured by `docker compose` directly.

**Consequence:** Memory limits now actually enforced. API capped at 896 MB + 2560 MB swap; frontend capped at 96 MB.

---

## 4. SSH Deploy Timeout

**Problem:** The GitHub Actions `ssh-action` step timed out because `docker compose up --pull always` was pulling a large image (~18.7 GB) over the network, exceeding the default SSH command timeout.

**Fix:** Increased `command_timeout` on the ssh-action step to `30m`.

**Consequence:** Longer CI runs. Root cause (large image) not yet addressed at this point.

---

## 5. BGE Reranker OOM at Startup

**Problem:** `HybridRAGPipeline` loaded the BGE-M3 reranker model (`BAAI/bge-reranker-v2-m3`) at startup, consuming ~2 GB RAM and killing the e2-micro (1 GB RAM + swap).

**Fix:** Set `RERANKER_ENABLED=false` and `EMBEDDING_PROVIDER=google` in `.env.prod` to skip BGE entirely and use the Google Embeddings API instead.

**Consequence:** No local embedding or reranking models are loaded. Embedding quality depends on Google's API availability. Reranking (which improves retrieval precision) is disabled.

---

## 6. `HybridRAGPipeline` Blocking uvicorn Startup

**Problem:** `HybridRAGPipeline()` was instantiated at module level (top of `main.py`). This meant uvicorn couldn't bind its socket or respond to the Docker healthcheck (`/health`) until the RAG pipeline fully initialised — which includes model loading, ChromaDB init, and DB connections. The healthcheck timed out and Docker kept restarting the container.

**Fix:** Moved `HybridRAGPipeline()` inside the FastAPI `lifespan` async context manager. uvicorn now starts and binds immediately; RAG init happens in the background during startup.

**Consequence:** `/health` responds `200 OK` before the RAG pipeline is ready. If a query hits the API in the narrow startup window, it returns `503 RAG Pipeline unavailable` — which is the correct and safe behaviour.

---

## 7. Docker Disk Exhaustion on VM (18.7 GB Images)

**Problem:** Each deploy pushed a new API image tag. With 3 tags accumulated (`latest`, `61b9d1ff`, `0d0236af`) at 18.7 GB each, the 30 GB VM boot disk reached 80% usage (~23 GB used, 5.9 GB free). When the next deploy pulled the new image, extracting `libtorch_cpu.so` failed with:

```
write /var/lib/containerd/.../libtorch_cpu.so: no space left on device
```

**Root cause:** `sentence-transformers`, `langchain-huggingface`, and `FlagEmbedding` were unconditional dependencies in `pyproject.toml`, pulling in PyTorch (~4 GB) even though `RERANKER_ENABLED=false`.

**Fix (immediate):** Prune old image tags from VM to free disk space.

**Fix (root cause):** Moved the three packages to an optional `[reranker]` extras group in `pyproject.toml`. Added `INSTALL_RERANKER` build-arg to `Dockerfile` (default `false`) — only passes `--extra reranker` to `uv sync` when explicitly set to `true`.

**Consequence:** API image shrunk from **18.7 GB → 3.32 GB**. Reranker and HuggingFace embeddings still work locally — install with `uv sync --extra reranker`. Future image tags will not exhaust disk. Old `latest` tag (18.7 GB) remains on VM until manually pruned.

---

## 8. Wrong `docker-compose.yml` Path Assumption

**Problem:** Deploy workflow and manual SSH commands assumed `docker-compose.yml` was at `/home/deploy/docker-compose.yml`. It is actually at `/home/risha/legal-advisor/docker-compose.yml`.

**Fix:** Updated SSH commands to use the correct path.

**Consequence:** None beyond confusion. The deploy workflow uses the repo-checkout path correctly; only ad-hoc manual commands were affected.

---

## 9. Frontend Stuck in `Created` State After Deploy

**Problem:** After the lean-image deploy, the API took ~2 minutes to become healthy (RAG init). The GitHub Actions SSH command timed out before `docker compose up -d` completed (frontend `depends_on` api `condition: service_healthy` blocks the command until API is healthy). The frontend container was left in `Created` state — pulled and configured but never started.

**Fix:** Manually ran `docker compose up -d frontend` on the VM via IAP SSH to start the frontend.

**Consequence:** Every deploy will hit this window unless the deploy command is restructured (e.g., start services independently without waiting on health, or increase the SSH action timeout further). Long-term fix: decouple frontend start from API health in the deploy step.

---

## Summary Table

| # | Problem | Fix | Image / Infra Impact |
|---|---------|-----|----------------------|
| 1 | GHCR uppercase name rejection | Hardcode lowercase image name in workflow | — |
| 2 | TypeScript `import.meta.env` errors | Fix tsconfig/Vite types | — |
| 3 | `mem_limit` ignored (Swarm-only key) | Move to top-level Compose keys | Memory limits now enforced |
| 4 | SSH deploy timeout on large pull | Increase `command_timeout` to 30m | Longer CI |
| 5 | BGE reranker OOM on e2-micro | `RERANKER_ENABLED=false`, `EMBEDDING_PROVIDER=google` | No local models loaded |
| 6 | Module-level RAG init blocks uvicorn | Move init into `lifespan` | `/health` available immediately; 503 during init window |
| 7 | 18.7 GB image exhausts 30 GB disk | Move PyTorch deps to optional `[reranker]` extra | Image 18.7 GB → 3.32 GB |
| 8 | Wrong `docker-compose.yml` path | Use correct `/home/risha/legal-advisor/` path | — |
| 9 | Frontend stuck in `Created` after deploy | Manual `docker compose up -d frontend` via IAP | Deploy step needs restructuring |
