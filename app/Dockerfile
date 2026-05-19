# ── Stage 1: dependency installer ──────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency manifests first (layer cache)
COPY pyproject.toml uv.lock ./

# INSTALL_RERANKER=true adds sentence-transformers + FlagEmbedding (~4 GB PyTorch).
# Leave unset (default) for a lean image when RERANKER_ENABLED=false.
ARG INSTALL_RERANKER=false

# Install all production deps into /app/.venv
# --mount=type=cache keeps the uv download cache between builds without bloating the final image
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
    $([ "$INSTALL_RERANKER" = "true" ] && echo "--extra reranker" || echo "")

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY . .

# Create log directory and fix ownership
RUN mkdir -p logs && chown -R appuser:appuser /app

# Activate venv on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

# Health check — hits the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
