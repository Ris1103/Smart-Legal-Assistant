# AI-Powered Legal Assistant for Indian SMBs

An intelligent legal advisory system for Indian small and medium businesses. Answers legal queries across GST, Income Tax, Company Law, Labour Law, and Criminal Law — and generates legal documents on demand — using a hybrid RAG knowledge base backed by Indian legal PDFs.

---

## What's Been Built

### Phase 0 — Basic RAG Pipeline
The starting point: a single FastAPI backend with a ChromaDB vector store, Google `text-embedding-004` embeddings, and `gemma-3-27b-it` for generation. Supported PDF ingestion and a `/retrieve` endpoint. No relevance checking or fallback.

### Phase 1 — Fix & Stabilize
Hardened the foundation:
- Pydantic `BaseSettings` for config (single `.env` source)
- SHA-256 duplicate detection on ingest; file size validation
- TF-IDF keyword search layered on top of semantic search (hybrid retrieval)
- LLM-powered relevance check; Perplexity web search fallback via async `httpx`
- Faithfulness scoring with a sentinel value (`-1.0`) distinguishing eval errors from unfaithful responses
- API key auth on all endpoints (`X-API-Key` header; disabled when key is empty)
- MLflow tracing on every `/retrieve` call (artifacts + metrics logged to `app/mlruns/`)
- Test suite: 29 tests across `test_agent.py`, `test_ingestion.py`, `test_evaluation.py`, `test_api.py`

### Phase 2 — Multi-Agent Architecture (LangGraph)
Replaced the single-agent path with a stateful multi-agent pipeline on `POST /query`:

```
[orchestrator] → classifies domain + confidence
    ├── confidence < 0.6       → [web_research] → END
    ├── intent == "contract"   → [contract]     → END
    └── otherwise              → [domain_agent] → [qa]
                                                    ├── qa_passed   → END
                                                    └── !qa_passed  → [domain_agent] (max 2 retries)
```

- **Orchestrator** — Gemini classifies the query into domain, confidence score, and intent
- **6 Domain Specialists** — GST, Income Tax, Company Law, Labour Law, Criminal Law, General; each with a dedicated ChromaDB collection and system prompt
- **QA Agent** — faithfulness + disclaimer + completeness gate; feeds feedback back for up to 2 retries
- **Contract Agent** — extracts parameters from natural language and renders Jinja2 templates (NDA, service agreement, employment agreement)
- **Web Research Agent** — async Perplexity fallback for low-confidence or out-of-KB queries
- MLflow tracing extended to `/query` (domain, search type, faithfulness score logged per run)

---

## What's Next

### Phase 3 — MCP Integration
Replace direct service calls with standardised MCP (Model Context Protocol) servers, each a separate FastAPI service:

| Server | Replaces | Exposed Tools |
|--------|----------|---------------|
| `mcp_servers/filesystem_server/` | base64-over-HTTP ingest | `upload_document`, `list_documents`, `delete_document`, `get_metadata` |
| `mcp_servers/database_server/` | direct ChromaDB/PostgreSQL calls | `query_contracts`, `get_template`, `save_contract` |
| `mcp_servers/search_server/` | hardcoded Perplexity REST in `web_research_agent.py` | `web_search(query, num_results)` — provider-agnostic |

Stretch goals: Indian Kanoon MCP (case law), MCA MCP (company registry), GSTN MCP (live GST rates).

### Phase 4 — AWS Deployment
Deploy on AWS Free Tier with a clear migration path to production scale:

- **Compute** — EC2 t2.micro + Docker Compose + Nginx reverse proxy + Let's Encrypt TLS
- **Data** — RDS db.t3.micro PostgreSQL, S3 (documents + MLflow artifacts), ChromaDB on EBS backed up to S3
- **Session cache** — `cachetools.TTLCache` in-process (ElastiCache is not free tier)
- **Observability** — CloudWatch logs + custom metrics, MLflow on RDS+S3 backend, X-Ray tracing
- **CI/CD** — GitHub Actions → ECR → EC2 Docker Compose (staging) → manual approval → prod
- **IaC** — AWS CDK (Python) in `infra/stacks/` covering network, database, storage, API gateway, and monitoring
- **Migration path** — EC2 Docker Compose → ECS Fargate when free tier expires

---

## Setup

### Prerequisites
- Python 3.12

### 1. Clone the repo

```bash
git clone <your-repository-url>
cd <your-project-directory>
```

### 2. Create virtual environment

```bash
# Windows
python -m venv app/.venv
app\.venv\Scripts\activate

# macOS / Linux
python3 -m venv app/.venv
source app/.venv/bin/activate
```

### 3. Install dependencies

```bash
cd app
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp app/.env-example app/.env
```

Fill in `app/.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
PERPLEXITY_MODEL_NAME=sonar
SERVICE_API_KEY=          # leave empty to disable auth in dev
```

### 5. Run

Open two terminals with the venv activated and `cd app`:

**Terminal 1 — backend:**
```bash
uvicorn main:app --reload
```
API at `http://localhost:8000` · Swagger UI at `http://localhost:8000/docs`

**Terminal 2 — frontend:**
```bash
streamlit run streamlit_app.py
```
UI at `http://localhost:8501`

---

## API Reference

### POST /query *(preferred)*
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the penalty for late GST filing?"}'
```

### POST /contracts/generate
```bash
curl -X POST http://localhost:8000/contracts/generate \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Draft an NDA between Acme Corp and John Doe for a software project"}'
```

### POST /ingest
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "base64_text": "<base64-encoded-pdf>",
    "file_type": ".pdf",
    "filename": "gst_act_2017.pdf",
    "metadata": {}
  }'
```

### POST /retrieve *(legacy)*
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"user_query": "What are the penalties under the Income Tax Act?"}'
```

> Add `-H "X-API-Key: your_key"` when `SERVICE_API_KEY` is set.

---

## Tests

```bash
cd app && pytest tests/ -v
```
