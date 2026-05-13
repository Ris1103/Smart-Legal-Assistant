# AI-Powered Legal Assistant for Indian SMBs

An intelligent legal advisory system for Indian small and medium businesses. Answers legal queries across GST, Income Tax, Company Law, Labour Law, and Criminal Law — and generates legal documents on demand — using a hybrid RAG knowledge base backed by Indian legal PDFs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI  :8501                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    FastAPI App  :8000                           │
│                                                                 │
│  POST /query          POST /contracts/generate   GET /contracts │
│  POST /ingest         POST /retrieve (legacy)    POST /refresh  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LangGraph Multi-Agent Pipeline             │   │
│  │                                                         │   │
│  │  [orchestrator] ──► domain + confidence + intent        │   │
│  │       │                                                  │   │
│  │       ├─ confidence < 0.6 ──► [web_research] ──► END    │   │
│  │       ├─ intent=contract  ──► [contract]     ──► END    │   │
│  │       └─ otherwise        ──► [domain_agent] ──► [qa]   │   │
│  │                                    ▲               │    │   │
│  │                                    └── retry ◄─────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  MCPClientManager (when MCP_ENABLED=true)                       │
│    search session ──► :8003   filesystem session ──► :8001      │
│    database session ──► :8002                                   │
└─────────────────────────────────────────────────────────────────┘
                  │                    │                 │
     ┌────────────▼──┐   ┌─────────────▼──┐   ┌─────────▼──────┐
     │  Search MCP   │   │ Filesystem MCP │   │  Database MCP  │
     │    :8003      │   │    :8001       │   │    :8002       │
     │               │   │                │   │                │
     │  web_search   │   │ upload_document │   │ save_contract  │
     │               │   │ list_documents  │   │ query_contracts│
     │  Perplexity / │   │ delete_document │   │ get_contract   │
     │  Tavily / Grok│   │ get_metadata    │   │ get_template   │
     └───────────────┘   └────────────────┘   └────────────────┘
                                │                      │
                    ┌───────────▼──────┐    ┌──────────▼──────┐
                    │  ChromaDB        │    │  SQLite          │
                    │  (chroma_db/)    │    │  contracts.db    │
                    └──────────────────┘    └─────────────────-┘
```

---

## What's Been Built

### Phase 0 — Basic RAG Pipeline
Single FastAPI backend with ChromaDB vector store, Google `text-embedding-004` embeddings, and `gemma-3-27b-it` for generation. Supported PDF ingestion and a `/retrieve` endpoint.

### Phase 1 — Fix & Stabilize
- Pydantic `BaseSettings` for config (single `.env` source)
- SHA-256 duplicate detection on ingest; file size validation
- Hybrid retrieval: semantic (ChromaDB) + keyword (TF-IDF), configurable weight
- LLM relevance check; Perplexity web search fallback via async `httpx`
- Faithfulness scoring with `-1.0` sentinel for eval errors
- API key auth (`X-API-Key` header; disabled when key is empty)
- MLflow tracing on every `/retrieve` call (artifacts + metrics → `app/mlruns/`)
- Test suite: 29 tests across `test_agent.py`, `test_ingestion.py`, `test_evaluation.py`, `test_api.py`

### Phase 2 — Multi-Agent Architecture (LangGraph)
Stateful multi-agent pipeline on `POST /query`:

| Agent | Role |
|-------|------|
| **Orchestrator** | Classifies query into domain, confidence score, and intent (query/contract) |
| **6 Domain Specialists** | GST, Income Tax, Company Law, Labour Law, Criminal Law, General — each with a dedicated ChromaDB collection and system prompt |
| **QA Agent** | Faithfulness + disclaimer + completeness gate; feeds critique back for up to 2 retries |
| **Contract Agent** | Extracts params from natural language; renders Jinja2 templates (NDA, service agreement, employment agreement) |
| **Web Research Agent** | Provider-agnostic web search fallback for low-confidence or out-of-KB queries |

Pre-Phase 3 RAG enhancements: BGE-M3 embedding option, cross-encoder reranking, context compression, RAGAS evaluation, multi-provider search (Perplexity / Tavily / Grok).

### Phase 3 — MCP Integration ✅ COMPLETE
Three standalone MCP servers (SSE/HTTP transport) wrap the main access patterns. The main app connects to them via `MCPClientManager` at startup when `MCP_ENABLED=true`. All existing paths remain fully functional with `MCP_ENABLED=false` (the default).

| Server | Port | Tools |
|--------|------|-------|
| `mcp_servers/search_server/` | 8003 | `web_search(query, num_results, provider)` |
| `mcp_servers/filesystem_server/` | 8001 | `upload_document`, `list_documents`, `delete_document`, `get_metadata` |
| `mcp_servers/database_server/` | 8002 | `save_contract`, `query_contracts`, `get_contract`, `get_template` |

Contract storage uses SQLite (`contracts.db`) — clean migration path to PostgreSQL in Phase 4.

---

## What's Next

### Phase 4 — AWS Deployment
Deploy on AWS Free Tier with a migration path to production scale:

- **Compute** — EC2 t2.micro + Docker Compose + Nginx reverse proxy + Let's Encrypt TLS
- **Data** — RDS db.t3.micro PostgreSQL (replacing SQLite), S3 (documents + MLflow artifacts), ChromaDB on EBS backed up to S3
- **Session cache** — `cachetools.TTLCache` in-process (ElastiCache is not free tier)
- **Observability** — CloudWatch logs + custom metrics, MLflow on RDS+S3 backend, X-Ray tracing
- **CI/CD** — GitHub Actions → ECR → EC2 Docker Compose (staging) → manual approval → prod
- **IaC** — AWS CDK (Python) in `infra/stacks/` covering network, database, storage, API gateway, and monitoring
- **Migration path** — EC2 Docker Compose → ECS Fargate when free tier expires

---

## Project Layout

```
Legal Advisor/
├── app/
│   ├── agents/                  # LangGraph agent nodes
│   │   ├── orchestrator.py
│   │   ├── domain/              # 6 domain specialists
│   │   ├── qa_agent.py
│   │   ├── contract_agent.py
│   │   └── web_research_agent.py
│   ├── api/routes/
│   │   ├── query.py             # POST /query
│   │   └── contracts.py         # POST /contracts/generate, GET /contracts
│   ├── config/settings.py       # Pydantic BaseSettings
│   ├── graph/
│   │   ├── graph_builder.py     # StateGraph assembly
│   │   └── state.py             # AgentState TypedDict
│   ├── mcp_client/              # SSE client wrappers
│   │   ├── client.py            # MCPClientManager
│   │   ├── search_client.py
│   │   ├── filesystem_client.py
│   │   └── database_client.py
│   ├── src/
│   │   ├── ingestion/           # PDF ingestion + chunking
│   │   ├── retriever/           # HybridRAGPipeline
│   │   ├── search/              # search_providers.py (Perplexity/Tavily/Grok)
│   │   └── evaluation/          # faithfulness + RAGAS
│   ├── templates/contracts/     # nda.j2, service_agreement.j2, employment_agreement.j2
│   ├── tests/                   # 29 tests (pytest)
│   ├── main.py                  # FastAPI app + lifespan
│   └── requirements.txt
├── mcp_servers/
│   ├── shared/__init__.py       # sys.path helper
│   ├── search_server/           # FastMCP SSE server :8003
│   ├── filesystem_server/       # FastMCP SSE server :8001
│   └── database_server/         # FastMCP SSE server :8002
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.12

### 1. Clone and create venv

```bash
git clone <your-repository-url>
cd "Legal Advisor"

# Windows
python -m venv app/.venv
app\.venv\Scripts\activate

# macOS / Linux
python3 -m venv app/.venv
source app/.venv/bin/activate
```

### 2. Install dependencies

```bash
cd app
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp app/.env-example app/.env
```

Minimum required in `app/.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
SERVICE_API_KEY=          # leave empty to disable auth in dev
```

To enable MCP servers (Phase 3):

```env
MCP_ENABLED=true
MCP_SEARCH_SERVER_URL=http://localhost:8003
MCP_FILESYSTEM_SERVER_URL=http://localhost:8001
MCP_DATABASE_SERVER_URL=http://localhost:8002
```

### 4. Run

**Main app (two terminals, `cd app` with venv active):**

```bash
# Terminal 1 — backend
uvicorn main:app --reload
# API → http://localhost:8000  |  Swagger → http://localhost:8000/docs

# Terminal 2 — frontend
streamlit run streamlit_app.py
# UI → http://localhost:8501
```

**MCP servers (optional, Phase 3 — each in its own terminal):**

```bash
python mcp_servers/search_server/server.py      # :8003
python mcp_servers/filesystem_server/server.py  # :8001
python mcp_servers/database_server/server.py    # :8002
```

---

## API Reference

### POST /query *(preferred)*
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"user_query": "What is the penalty for late GST filing?"}'
```

### POST /contracts/generate
```bash
curl -X POST http://localhost:8000/contracts/generate \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Draft an NDA between Acme Corp and John Doe for a software project"}'
```

### GET /contracts *(requires MCP_ENABLED=true)*
```bash
curl http://localhost:8000/contracts?limit=10
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

> Add `-H "X-API-Key: your_key"` to any request when `SERVICE_API_KEY` is set.

---

## Tests

```bash
cd app && pytest tests/ -v   # 29 passing
```
