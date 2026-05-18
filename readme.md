# AI-Powered Legal Assistant for Indian SMBs

An intelligent legal advisory system for Indian small and medium businesses. Answers legal queries across GST, Income Tax, Company Law, Labour Law, and Criminal Law — and generates legal documents on demand — using a hybrid RAG knowledge base backed by Indian legal PDFs.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    React UI  :3000  (Vite + Tailwind + Clerk)    │
│   Chat  ·  Document Upload  ·  Contract Generator               │
└────────────────────────┬─────────────────────────────────────────┘
                         │  Bearer JWT (Clerk)
┌────────────────────────▼─────────────────────────────────────────┐
│                    FastAPI App  :8000                            │
│                                                                  │
│  POST /query          POST /contracts/generate   GET /health     │
│  POST /ingest         POST /retrieve (legacy)    GET /users/me   │
│  POST /refresh-index  GET/POST /conversations    POST /webhooks  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │             LangGraph Multi-Agent Pipeline               │   │
│  │                                                          │   │
│  │  [orchestrator] ──► domain + confidence + intent         │   │
│  │       │                                                   │   │
│  │       ├─ confidence < 0.6 ──► [web_research] ──► END     │   │
│  │       ├─ intent=contract  ──► [contract]     ──► END     │   │
│  │       └─ otherwise        ──► [domain_agent] ──► [qa]    │   │
│  │                                    ▲               │     │   │
│  │                                    └── retry ◄─────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  asyncpg Pool ──► Neon PostgreSQL (users, conversations,         │
│                                    api_keys, contracts)          │
│  slowapi rate limiting  ·  CORS middleware  ·  Clerk JWT auth    │
└──────────────────────────────────────────────────────────────────┘
         │                    │                 │
┌────────▼──────┐  ┌──────────▼──────┐  ┌──────▼──────────┐
│  Search MCP   │  │ Filesystem MCP  │  │  Database MCP   │
│    :8003      │  │    :8001        │  │    :8002        │
│  web_search   │  │ upload_document │  │ save_contract   │
│  Tavily/Grok  │  │ list_documents  │  │ query_contracts │
└───────────────┘  └─────────────────┘  └─────────────────┘
         │
┌────────▼──────────────┐
│  Vector Store         │
│  ChromaDB (local dev) │
│  MongoDB Atlas (cloud)│
│  pgvector / Pinecone  │
└───────────────────────┘
```

---

## What's Been Built

### Phase 0 — Basic RAG Pipeline
Single FastAPI backend with ChromaDB vector store, Google embeddings, and Gemma for generation.

### Phase 1 — Fix & Stabilize
- Pydantic `BaseSettings` for config (single `.env` source)
- SHA-256 duplicate detection on ingest; file size validation
- Hybrid retrieval: semantic (ChromaDB) + keyword (TF-IDF), configurable weight
- LLM relevance check; pluggable web search fallback (Perplexity / Tavily / Grok)
- Faithfulness scoring with `-1.0` sentinel for eval errors
- API key auth (`X-API-Key` header; disabled when key is empty)
- MLflow tracing on every `/retrieve` call
- Test suite: 29 tests across agent, ingestion, evaluation, API

### Phase 2 — Multi-Agent Architecture (LangGraph)
Stateful multi-agent pipeline on `POST /query`:

| Agent | Role |
|-------|------|
| **Orchestrator** | Classifies query into domain, confidence score, and intent (query/contract) |
| **6 Domain Specialists** | GST, Income Tax, Company Law, Labour Law, Criminal Law, General |
| **QA Agent** | Faithfulness + disclaimer gate; up to 2 retries |
| **Contract Agent** | Renders Jinja2 templates (NDA, service agreement, employment agreement) |
| **Web Research Agent** | Provider-agnostic fallback (Tavily / Grok / Perplexity) |

### Phase 3 — MCP Integration
Three standalone MCP servers (SSE/HTTP). Active when `MCP_ENABLED=true`.

| Server | Port | Tools |
|--------|------|-------|
| `mcp_servers/search_server/` | 8003 | `web_search` |
| `mcp_servers/filesystem_server/` | 8001 | `upload_document`, `list_documents`, `delete_document`, `get_metadata` |
| `mcp_servers/database_server/` | 8002 | `save_contract`, `query_contracts`, `get_contract`, `get_template` |

### Phase 4 — Production Scale (In Progress)

#### 4.1 Vector Store Abstraction
Provider-agnostic `BaseVectorStore` interface with Strategy + Factory + Singleton pattern.

| Provider | Use Case |
|----------|----------|
| `chromadb` | Local dev default |
| `mongodb_atlas` | Cloud default (M0 free forever) |
| `pgvector` | Neon / Supabase PostgreSQL |
| `pinecone` | Free tier (100k vectors) |

Switch via `VECTOR_STORE_PROVIDER` env var — zero code changes.

#### 4.2 Auth + PostgreSQL (Clerk + Neon)
- Clerk JWT verification on all API endpoints (`require_user` dependency)
- Dev mode: auth bypassed when `CLERK_SECRET_KEY` is empty
- asyncpg connection pool → Neon free-tier PostgreSQL
- Tables: `users`, `conversations`, `api_keys`, `contracts`
- Conversation history endpoints: `GET/POST /conversations`, `GET /conversations/{id}`
- Clerk webhook upserts users on sign-up (`POST /webhooks/clerk`)
- `slowapi` rate limiting (10 req/min per IP, configurable)
- CORS configured for `:3000` (React) and `:8501` (Streamlit)

#### 4.3 React Frontend (Current)
- **React 18 + TypeScript + Vite** at `frontend/`
- **Tailwind CSS** — responsive, mobile-first
- **Clerk React SDK** — sign-in UI, Google SSO, JWT auto-attach
- **TanStack Query** — async data fetching
- **React Router v6** — SPA routing

| Page | Path | Description |
|------|------|-------------|
| Login | `/login` | Clerk `<SignIn />` — email + Google SSO |
| Chat | `/` | Chat UI connected to `POST /query` |
| Documents | `/documents` | Drag-and-drop PDF upload → `POST /ingest` |
| Contracts | `/contracts` | Contract generator → `POST /contracts/generate` |

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
│   │   ├── contracts.py         # POST /contracts/generate
│   │   ├── users.py             # GET /users/me, /conversations/*
│   │   └── webhooks.py          # POST /webhooks/clerk
│   ├── auth/clerk.py            # JWT verification dependency
│   ├── config/settings.py       # Pydantic BaseSettings
│   ├── db/
│   │   ├── database.py          # asyncpg pool
│   │   └── migrations.py        # idempotent DDL
│   ├── graph/
│   │   ├── graph_builder.py
│   │   └── state.py
│   ├── src/
│   │   ├── ingestion/
│   │   ├── retriever/           # HybridRAGPipeline
│   │   ├── search/              # Perplexity / Tavily / Grok
│   │   ├── vectorstore/         # BaseVectorStore + 4 providers
│   │   └── evaluation/
│   ├── templates/contracts/     # nda.j2, service_agreement.j2, employment_agreement.j2
│   ├── tests/                   # 29 tests (pytest)
│   ├── main.py
│   └── requirements.txt
├── frontend/                    # React 18 + Vite + Tailwind
│   ├── src/
│   │   ├── components/          # Layout, Chat, MessageBubble, DocumentUpload, ContractViewer
│   │   ├── hooks/               # useChat, useDocuments
│   │   ├── lib/                 # api.ts (axios), utils.ts
│   │   └── pages/               # LoginPage, DashboardPage, DocumentsPage, ContractsPage
│   ├── .env                     # VITE_CLERK_PUBLISHABLE_KEY, VITE_API_URL
│   └── package.json
├── mcp_servers/
│   ├── search_server/           # :8003
│   ├── filesystem_server/       # :8001
│   └── database_server/         # :8002
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.12
- Node.js 20+

### 1. Clone and create Python venv

```bash
git clone <your-repository-url>
cd "Legal Advisor"

python -m venv app/.venv
# Windows
app\.venv\Scripts\activate
# macOS / Linux
source app/.venv/bin/activate
```

### 2. Install Python dependencies

```bash
cd app && pip install -r requirements.txt
```

### 3. Configure backend environment

```bash
cp app/.env-example app/.env
```

Minimum required in `app/.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
WEB_SEARCH_PROVIDER=tavily

# Auth (Clerk) — leave empty to disable auth in dev
CLERK_SECRET_KEY=
CLERK_PUBLISHABLE_KEY=

# Database (Neon) — leave empty to disable DB features in dev
DATABASE_URL=
```

### 4. Install frontend dependencies

```bash
cd frontend
npm install --cache D:\path\to\.npm-cache   # keeps cache off C drive
```

Configure `frontend/.env`:

```env
VITE_CLERK_PUBLISHABLE_KEY=pk_test_...
VITE_API_URL=http://localhost:8000
```

### 5. Run

**Backend (Terminal 1):**
```bash
cd app && uvicorn main:app --reload
# API → http://localhost:8000  |  Swagger → http://localhost:8000/docs
```

**React frontend (Terminal 2):**
```bash
cd frontend && npm run dev
# UI → http://localhost:3000
```

**Legacy Streamlit UI (optional):**
```bash
cd app && streamlit run streamlit_app.py
# UI → http://localhost:8501
```

**MCP servers (optional, Phase 3):**
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

### POST /ingest
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"base64_text": "<base64-pdf>", "file_type": ".pdf", "filename": "gst_act.pdf", "metadata": {}}'
```

### GET /health
```bash
curl http://localhost:8000/health
```

### GET /users/me *(requires Clerk JWT)*
```bash
curl http://localhost:8000/users/me -H "Authorization: Bearer <token>"
```

> Add `-H "X-API-Key: your_key"` when `SERVICE_API_KEY` is set.

---

## Tests

```bash
cd app && pytest tests/ -v   # 29 passing
```

---

## What's Next

### Phase 4 (Remaining) — Docker + CI/CD + GCP
- **Docker** — multi-stage `Dockerfile` for FastAPI + React; `docker-compose.yml` for all services
- **CI/CD** — GitHub Actions: build → push GHCR → SSH deploy
- **GCP** — e2-micro (free forever) + MongoDB Atlas M0 + Neon PostgreSQL
- **Terraform + Terragrunt** — `infra/` modules for GCP and AWS
