# AI-Powered Legal Assistant for Indian SMBs

An intelligent legal advisory platform built for Indian small and medium businesses. The system answers complex legal queries across GST, Income Tax, Company Law, Labour Law, and Criminal Law, and generates enforceable legal documents on demand. It combines a hybrid RAG pipeline, a LangGraph multi-agent orchestration layer, Clerk-authenticated REST APIs, a Neon PostgreSQL backend, and a React frontend — all designed to run at near-zero cost on GCP's free tier.

---

## What This Is

Most Indian SMBs cannot afford a legal retainer. This platform acts as a first-line legal advisor: it searches a curated knowledge base of Indian legal PDFs, reasons over them with domain-specific AI agents, and falls back to live web search when the local knowledge base is insufficient. When a user needs a document — an NDA, service agreement, or employment contract — the system extracts the relevant parameters from natural language and renders a Jinja2 template into a complete legal draft.

The system is built with production concerns in mind from the start: pluggable vector store backends, Clerk SSO authentication, PostgreSQL for persistent user data, rate limiting, CORS, and a CI/CD pipeline that deploys to GCP on every push to `main`.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                  React Frontend  :3000                               │
│  Vite · TypeScript · Tailwind CSS · Clerk React SDK · TanStack Query │
│                                                                      │
│   /           Chat (POST /query) · inline citations · PDF download   │
│   /documents  Drag-and-drop PDF ingest (POST /ingest)                │
│   /contracts  Contract generator (POST /contracts/generate)          │
│   /login      Clerk SignIn (email + Google SSO)                      │
└───────────────────────────┬──────────────────────────────────────────┘
                            │  HTTPS · Bearer JWT (Clerk RS256)
┌───────────────────────────▼──────────────────────────────────────────┐
│                  FastAPI Backend  :8000                              │
│                                                                      │
│  Middleware: CORS · slowapi rate limiting · Clerk JWT verification   │
│                                                                      │
│  POST /query               ─► LangGraph multi-agent pipeline         │
│  POST /contracts/generate  ─► ContractAgent + Jinja2 templates       │
│  POST /ingest              ─► PDF chunking + vector store            │
│  POST /export/pdf          ─► Legal Advisory Report PDF download     │
│  GET  /health              ─► Liveness probe                         │
│  GET  /users/me            ─► Clerk user upsert                      │
│  GET/POST /conversations   ─► Conversation history                   │
│  POST /webhooks/clerk      ─► Svix-verified user sync                │
│  POST /retrieve            ─► [DEPRECATED] use /query                │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              LangGraph Multi-Agent Pipeline                    │  │
│  │                                                                │  │
│  │  [orchestrator]  classify domain · confidence · intent         │  │
│  │        │                                                        │  │
│  │        ├── confidence < 0.6  ──► [web_research_agent] ──► END  │  │
│  │        ├── intent = contract ──► [contract_agent]     ──► END  │  │
│  │        └── otherwise         ──► [domain_agent]                │  │
│  │                                       │                        │  │
│  │                              multi-query expansion              │  │
│  │                              BM25 + semantic → RRF              │  │
│  │                              BGE reranker (top 8)               │  │
│  │                                       │                        │  │
│  │                                    [qa_agent]                  │  │
│  │                                       ├── pass  ──► END        │  │
│  │                                       └── fail  ──► retry (×2) │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  asyncpg pool ──► Neon PostgreSQL                                    │
│  (users · conversations · api_keys · contracts)                      │
└───────┬──────────────┬──────────────┬───────────────────────────────┘
        │              │              │
┌───────▼────┐  ┌──────▼──────┐  ┌───▼─────────────┐
│ Search MCP │  │  FS MCP     │  │  Database MCP   │
│   :8003    │  │  :8001      │  │  :8002          │
│ web_search │  │ upload_doc  │  │ save_contract   │
│ Tavily/Grok│  │ list_docs   │  │ query_contracts │
└───────┬────┘  └──────┬──────┘  └───┬─────────────┘
        │              │              │
        └──────────────▼──────────────┘
                       │
            ┌──────────▼─────────────┐
            │     Vector Store       │
            │  ChromaDB  (local dev) │
            │  MongoDB Atlas M0      │  ← cloud default
            │  pgvector on Neon      │
            │  Pinecone              │
            └────────────────────────┘
```

---

## Core Concepts

### Retrieval Pipeline — BM25 + Semantic + RRF + Reranking

The retrieval layer is a four-stage pipeline designed to maximise both precision (exact legal terminology) and recall (semantic meaning):

**Stage 1 — Sparse retrieval (BM25):** `BM25Okapi` indexes the full document corpus with document-length normalisation and term-frequency saturation. This outperforms TF-IDF on legal documents, which vary widely in length. Exact section numbers, act names, and tax codes are retrieved with high precision.

**Stage 2 — Dense retrieval (Semantic):** BGE-M3 embeddings stored in the configured vector store. Captures meaning, synonyms, paraphrases, and cross-lingual legal terminology (Hindi terms in English documents).

**Stage 3 — RRF Fusion:** Reciprocal Rank Fusion (`score = Σ 1/(k + rank_i)`, k=60) merges the two ranked lists using only rank positions, making it scale-invariant. Unlike weighted blending, RRF requires no score normalisation between incomparable BM25 and cosine similarity scales.

**Stage 4 — BGE Reranker:** `BAAI/bge-reranker-v2-m3` — the cross-encoder reranker from the same BGE family as the embedding model — re-scores the merged candidate set and returns the top 8. Using a matched reranker ensures the semantic space of retrieval and reranking are aligned. Enable with `RERANKER_ENABLED=true`.

### Multi-Query Expansion

Before retrieval, Gemma generates 3 alternative phrasings of the user's query. All 4 queries (original + 3 variants) are run through the BM25 + semantic pipeline independently. Results are deduplicated by `chunk_id` (keeping the highest-scored copy), then the full unique candidate set is passed to the BGE reranker. This dramatically improves recall for queries where vocabulary mismatch would cause relevant chunks to be missed by any single phrasing.

### Multi-Agent Orchestration (LangGraph)

Every `/query` request builds and executes a `StateGraph`. The `AgentState` TypedDict is the single mutable object passed through all nodes — no shared global state, fully traceable.

The **Orchestrator** (Gemma) classifies the incoming query into one of six legal domains, produces a confidence score, and identifies the intent (informational query vs. contract generation):

- Low confidence → immediate web search fallback (no wasted RAG call)
- Contract intent → directly to `ContractAgent` (no retrieval needed)
- Everything else → domain specialist → QA gate → optional retry

The **QA Agent** checks the domain agent's response for faithfulness, completeness, and the presence of a legal disclaimer. If it fails either check, it writes a critique back into the state and re-invokes the domain agent. This loop runs at most twice to bound latency.

### Layout-Aware Chunking

Activate with `CHUNK_STRATEGY=layout`. Uses `pdfplumber` to extract PDF content with structural awareness:

- **Tables** are serialised to markdown (`| col | col |` format) and stored as discrete chunks with `element_type=table`
- **Section headers** are detected by pattern matching and prepended to all following chunks: `[Section: GST Registration]\n{chunk text}` — giving the LLM section context even when a chunk spans a page break
- **Reading order** is preserved; multi-column layouts are handled correctly
- Falls back to `RecursiveCharacterTextSplitter` if extraction fails

Each chunk carries `element_type`, `section_header`, and `page` metadata in addition to the standard `filename`, `category`, and `file_hash`.

### Source Citations

Every retrieved chunk is annotated with `[Source: filename.pdf, Page N]` before being passed to the LLM. The response prompt instructs the model to reference these tags inline. The API response carries a `citations[]` array — each entry has `filename`, `page`, `chunk_id`, and a 120-character excerpt. The React UI renders citations as a collapsible panel below each assistant message.

### Descriptive Responses

Domain agents generate structured, comprehensive responses with:
- A direct answer to the question
- Relevant statutory provisions with specific section numbers and act names
- Practical implications for an Indian SMB
- Deadlines, penalties, or compliance requirements where applicable
- A legal disclaimer

Pass `response_style=brief` in the request body for a concise 2–3 sentence answer instead.

### Vector Store Abstraction

`BaseVectorStore` is an abstract class defining a provider-agnostic interface (`add_documents`, `similarity_search`, `similarity_search_with_scores`, `get_all`, `get_by_metadata`). `VectorStoreFactory` is a singleton registry that creates and caches provider instances on first use. Switching backends requires only a single env var change — no code changes.

---

## Models & Embeddings

| Component | Model | Provider | Notes |
|-----------|-------|----------|-------|
| Generation | `gemma-4-26b-a4b-it` | Google AI (free tier) | All domain agents, QA agent, orchestrator, query expansion |
| Embeddings | `BAAI/bge-m3` | HuggingFace (local) | Default; multilingual, strong on legal text, ~2 GB |
| Embeddings (alt) | `models/gemini-embedding-001` | Google AI | Switch via `EMBEDDING_PROVIDER=google` |
| Reranker | `BAAI/bge-reranker-v2-m3` | HuggingFace (local) | Matched cross-encoder for BGE-M3; enable with `RERANKER_ENABLED=true` |
| Web Search | Tavily / Grok / Perplexity | External APIs | Provider via `WEB_SEARCH_PROVIDER` |

BGE-M3 is the default embedding model because it is multilingual, produces high-quality dense representations for long legal passages, and runs entirely locally with no API cost. `bge-reranker-v2-m3` is used as the reranker because it is trained in the same semantic space as BGE-M3, giving consistent ranking behaviour across retrieval and reranking stages.

---

## Databases

### Vector Store (ChromaDB / MongoDB Atlas / pgvector / Pinecone)

Stores document chunks as dense vectors alongside metadata. Used for semantic retrieval and BM25 corpus hydration. The provider is selected at startup via `VECTOR_STORE_PROVIDER`.

| Provider | Free Tier | Best For |
|----------|-----------|----------|
| **ChromaDB** | Local, unlimited | Local development — no account needed |
| **MongoDB Atlas M0** | 512 MB forever | Cloud default — Atlas Vector Search included |
| **pgvector on Neon** | 0.5 GB forever | If already using Neon for relational data |
| **Pinecone** | 100k vectors, 1 index | Alternative cloud option |

### Relational Store (Neon PostgreSQL)

Neon's serverless PostgreSQL (free tier, 0.5 GB) stores all user-facing relational data. The asyncpg connection pool is initialised during FastAPI's lifespan startup, and the DDL is applied idempotently on every boot.

```sql
users         — Clerk user IDs, email addresses
conversations — JSONB message arrays per user
api_keys      — bcrypt-hashed developer API keys
contracts     — Generated contract text + parameters per user
```

When `DATABASE_URL` is not set, the pool is skipped and all DB-dependent endpoints return `503` — the RAG and query endpoints are unaffected and keep working in dev.

### MLflow (experiment tracking)

Every `/query` call starts an MLflow run under the `Legal_RAG_Assistant` experiment. Retrieved context, generated summaries, and faithfulness scores are logged as artifacts and metrics. Runs are stored locally in `app/mlruns/`. In production this is disabled (logging to stdout only) to avoid disk usage on the VM.

---

## Security

| Layer | Implementation |
|-------|---------------|
| **Authentication** | Clerk JWT (RS256). `require_user` FastAPI dependency verifies every protected route. Dev mode: bypassed when `CLERK_SECRET_KEY` is empty. |
| **SSO** | Google OAuth via Clerk — zero config on the backend, Clerk handles the OAuth flow. |
| **Webhook verification** | Svix signature check on `POST /webhooks/clerk`. Skipped in dev when `CLERK_WEBHOOK_SECRET` is empty. |
| **Rate limiting** | `slowapi` middleware — 10 req/min per IP by default, configurable via `RATE_LIMIT_PER_MINUTE`. |
| **CORS** | FastAPI `CORSMiddleware` — whitelists `localhost:3000` (React) and `localhost:8501` (Streamlit) in dev. Production origins added via env var. |
| **Service API key** | `X-API-Key` header auth on `/ingest`, `/refresh-index`. Disabled when `SERVICE_API_KEY` is empty. |
| **Input validation** | Pydantic models on all request bodies — type coercion, length limits, regex patterns on file types and search modes. |
| **Secrets** | All credentials in `app/.env` (never committed). `.env-example` documents every variable. `detect-secrets` pre-commit hook blocks accidental key commits. |
| **TLS** | Nginx + Let's Encrypt on the production VM. Terminates SSL before FastAPI. |

---

## UI — React Frontend

The frontend is a single-page application at `frontend/` built with Vite 5, React 18, and TypeScript.

**Stack:**
- **Vite 5** — dev server with HMR, proxies `/api/*` to FastAPI `:8000`
- **Tailwind CSS 3** — utility-first, mobile-first responsive layout
- **Clerk React SDK** — `<SignIn />` component, `useAuth()` hook, JWT auto-attached to every API call
- **TanStack Query** — async mutations with loading/error states, cache invalidation
- **React Router v6** — client-side routing with `<Navigate>` guards
- **Axios** — API client with `Authorization: Bearer` header injection

**Pages:**

| Route | Page | Description |
|-------|------|-------------|
| `/login` | LoginPage | Clerk sign-in: email/password + Google SSO |
| `/` | DashboardPage | Chat interface — suggestion chips, message history, inline citations, PDF download |
| `/documents` | DocumentsPage | Drag-and-drop or click-to-browse PDF upload with live ingest status |
| `/contracts` | ContractsPage | Quick-template buttons + free-text prompt → rendered contract with download |

**Chat features:**
- Each assistant message shows a collapsible **Citations** panel — per-chunk source file, page number, and excerpt
- **Download as PDF** button on every assistant message calls `POST /export/pdf` and triggers a browser download of a formatted Legal Advisory Report

**Layout:** Collapsible sidebar on desktop, slide-in drawer on mobile. `UserButton` from Clerk in the sidebar footer handles avatar, profile, and sign-out with no custom code.

---

## PDF Export

`POST /export/pdf` accepts any query + summary + citations and returns a formatted PDF:

- Title: **Legal Advisory Report** with timestamp and domain
- Query block
- Full answer body (paragraph-by-paragraph, preserving formatting)
- Sources section — deduplicated by filename + page, with excerpt
- Disclaimer footer

The React Chat UI calls this endpoint automatically when the user clicks **Download as PDF** on any assistant response.

---

## Cloud Architecture

### Primary: GCP (Free Forever)

The target deployment is a single GCP e2-micro VM (free forever under the Always Free tier) running Docker Compose behind Nginx.

```
Internet
    │
    ▼
Nginx (port 80/443) — Let's Encrypt TLS
    ├── /          ──► React  (container :3000)
    └── /api/      ──► FastAPI (container :8000)

GCP e2-micro VM (us-central1 or asia-south1)
  └── Docker Compose
        ├── api       (ghcr.io/ris1103/legal-advisor-api:latest)
        ├── frontend  (ghcr.io/ris1103/legal-advisor-ui:latest)
        ├── mcp-search     (:8003)
        ├── mcp-filesystem (:8001)
        └── mcp-database   (:8002)

Managed Services (all free tier, external to VM):
  ├── MongoDB Atlas M0  — vector store (512 MB)
  ├── Neon PostgreSQL   — relational data (0.5 GB)
  └── GCS bucket        — PDF uploads, MLflow artifacts (5 GB)
```

No persistent disk is needed on the VM because every stateful component lives in a managed cloud service. The VM is stateless and can be replaced without data loss.

**Cost: $0/month.** The e2-micro is Always Free, Atlas M0 is free forever, Neon's free tier covers this workload (~100 req/month), and GCS 5 GB is free.

### Alternative: AWS

The same Docker Compose setup can run on an EC2 t2.micro (12 months free). The Terraform modules mirror GCP structure: VPC + security group, EC2 instance, S3 bucket (replaces GCS). Switch by running `cd infra/environments/aws && terragrunt apply` instead of the GCP equivalent.

```
AWS equivalent:
  EC2 t2.micro          ← replaces e2-micro
  S3 bucket             ← replaces GCS
  (MongoDB Atlas + Neon stay the same — they're cloud-agnostic)
```

---

## Deployment Pipeline

### Docker

Each service has a multi-stage `Dockerfile`:

**FastAPI** (`app/Dockerfile`):
1. `python:3.12-slim` — install `requirements.txt`
2. Copy source, run `uvicorn main:app --host 0.0.0.0 --port 8000`

**React** (`frontend/Dockerfile`):
1. `node:20-slim` — `npm ci && npm run build`
2. `nginx:alpine` — serve `/dist` from Nginx

Images are pushed to GitHub Container Registry (GHCR) on every merge to `main`.

### CI/CD (GitHub Actions)

```
push to main
    │
    ├── build api image  ──► ghcr.io/ris1103/legal-advisor-api:$SHA
    ├── build ui image   ──► ghcr.io/ris1103/legal-advisor-ui:$SHA
    │
    └── SSH into GCP VM
            ├── docker compose pull
            └── docker compose up -d
```

Secrets stored in GitHub: `VM_SSH_KEY`, `VM_HOST`, `GHCR_TOKEN`.

### Infrastructure as Code (Terraform + Terragrunt)

```
infra/
├── terragrunt.hcl              ← remote state config (GCS bucket)
├── modules/
│   ├── gcp/
│   │   ├── vm/                 ← e2-micro, firewall (80, 443, 22)
│   │   ├── storage/            ← GCS bucket
│   │   └── networking/         ← VPC, static IP
│   └── aws/
│       ├── vm/                 ← EC2 t2.micro, security group
│       ├── storage/            ← S3 bucket
│       └── networking/         ← VPC, elastic IP
└── environments/
    ├── gcp/terragrunt.hcl      ← GCP-specific inputs
    └── aws/terragrunt.hcl      ← AWS-specific inputs
```

Deploy to GCP: `cd infra/environments/gcp && terragrunt apply`
Switch to AWS: `cd infra/environments/aws && terragrunt apply`

---

## Configuration Reference

All settings are loaded by `app/config/settings.py` (Pydantic `BaseSettings`) from `app/.env`.

```env
# --- API Keys ---
GOOGLE_API_KEY=                     # Google AI Studio key (required)
TAVILY_API_KEY=                     # Web search — tavily.com
GROK_API_KEY=                       # Web search alternative — x.ai
PERPLEXITY_API_KEY=                 # Web search alternative

# --- Service ---
FASTAPI_URL=http://localhost:8000
SERVICE_API_KEY=                    # X-API-Key header auth; empty = disabled

# --- Models ---
GENERATIVE_MODEL_NAME=gemma-4-26b-a4b-it
EMBEDDING_PROVIDER=bge              # "bge" | "google"
BGE_MODEL_NAME=BAAI/bge-m3
GOOGLE_EMBEDDING_MODEL_NAME=models/gemini-embedding-001
WEB_SEARCH_PROVIDER=tavily          # "tavily" | "grok" | "perplexity"

# --- Chunking ---
CHUNK_STRATEGY=recursive            # "recursive" | "semantic" | "layout"
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# --- Retrieval ---
RERANKER_ENABLED=true               # BGE reranker; false to skip for speed
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
TOP_K_RETRIEVAL=8

# --- Context Compression (optional) ---
CONTEXT_COMPRESSION_ENABLED=false
COMPRESSION_SIMILARITY_THRESHOLD=0.5

# --- Vector Store ---
VECTOR_STORE_PROVIDER=chromadb      # "chromadb" | "mongodb_atlas" | "pgvector" | "pinecone"
CHROMA_DB_PATH=                     # defaults to app/chroma_db/
MONGODB_ATLAS_URI=                  # mongodb+srv://...
PGVECTOR_DSN=                       # postgresql://...
PINECONE_API_KEY=
PINECONE_INDEX_NAME=legal-advisor

# --- Auth (Clerk) ---
CLERK_SECRET_KEY=                   # sk_test_... — empty = auth disabled in dev
CLERK_PUBLISHABLE_KEY=              # pk_test_...
CLERK_WEBHOOK_SECRET=               # whsec_... — empty = webhook verification skipped

# --- Database (Neon) ---
DATABASE_URL=                       # postgresql+asyncpg://... — empty = DB disabled in dev
RATE_LIMIT_PER_MINUTE=10

# --- MCP (optional) ---
MCP_ENABLED=false
MCP_SEARCH_SERVER_URL=http://localhost:8003
MCP_FILESYSTEM_SERVER_URL=http://localhost:8001
MCP_DATABASE_SERVER_URL=http://localhost:8002
```

**Frontend** (`frontend/.env`):
```env
VITE_CLERK_PUBLISHABLE_KEY=pk_test_...
VITE_API_URL=http://localhost:8000
```

---

## Local Development

### Prerequisites
- Python 3.12
- Node.js 20.x

### Backend

```bash
git clone <repo-url>
cd "Legal Advisor"

python -m venv app/.venv
app\.venv\Scripts\activate          # Windows
# source app/.venv/bin/activate     # macOS / Linux

cd app
pip install -r requirements.txt
cp .env-example .env                # fill in GOOGLE_API_KEY and TAVILY_API_KEY at minimum

uvicorn main:app --reload
# API  → http://localhost:8000
# Docs → http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# UI → http://localhost:3000
```

### MCP Servers (optional)

```bash
python mcp_servers/search_server/server.py      # :8003
python mcp_servers/filesystem_server/server.py  # :8001
python mcp_servers/database_server/server.py    # :8002
```

Set `MCP_ENABLED=true` in `app/.env` to activate the MCP code paths.

---

## API Reference

### POST /query
Primary endpoint. Runs multi-query expansion → BM25 + semantic → RRF → BGE reranker → domain agent → QA gate.
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"user_query": "What is the penalty for late GST filing?", "response_style": "detailed"}'
```
Response includes `summary`, `citations[]`, `domain`, `confidence`, `results[]`.

### POST /export/pdf
Generate a downloadable Legal Advisory Report PDF.
```bash
curl -X POST http://localhost:8000/export/pdf \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "summary": "...", "citations": [...], "domain": "GST"}' \
  --output report.pdf
```

### POST /contracts/generate
Generate a legal document from a natural language description.
```bash
curl -X POST http://localhost:8000/contracts/generate \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Draft an NDA between Acme Corp and John Doe for a 6-month AI project"}'
```

### POST /ingest
Ingest a PDF into the vector store. Deduplicates by SHA-256 hash.
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"base64_text": "<base64>", "file_type": ".pdf", "filename": "gst_act.pdf", "metadata": {}}'
```

### GET /health
```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "2.0.0"}
```

### GET /users/me *(Clerk JWT required)*
```bash
curl http://localhost:8000/users/me -H "Authorization: Bearer <clerk_jwt>"
```

### GET /conversations *(Clerk JWT required)*
```bash
curl http://localhost:8000/conversations -H "Authorization: Bearer <clerk_jwt>"
```

> Add `-H "X-API-Key: <key>"` when `SERVICE_API_KEY` is configured.

---

## Project Layout

```
Legal Advisor/
├── app/
│   ├── agents/
│   │   ├── orchestrator.py          # Gemma domain classifier
│   │   ├── domain/                  # 6 specialists: GST, Income Tax, Company Law,
│   │   │                            #   Labour Law, Criminal Law, General
│   │   ├── qa_agent.py              # Faithfulness + disclaimer gate, retry loop
│   │   ├── contract_agent.py        # Jinja2 contract renderer
│   │   └── web_research_agent.py    # Provider-agnostic web search node
│   ├── api/routes/
│   │   ├── query.py                 # POST /query (citations, response_style)
│   │   ├── contracts.py             # POST /contracts/generate
│   │   ├── export.py                # POST /export/pdf
│   │   ├── users.py                 # GET /users/me, /conversations/*
│   │   └── webhooks.py              # POST /webhooks/clerk
│   ├── auth/
│   │   └── clerk.py                 # JWT verification dependency
│   ├── config/
│   │   └── settings.py              # Pydantic BaseSettings (single .env source)
│   ├── db/
│   │   ├── database.py              # asyncpg pool + get_db dependency
│   │   └── migrations.py            # Idempotent DDL (runs on every boot)
│   ├── graph/
│   │   ├── graph_builder.py         # StateGraph assembly + routing logic
│   │   └── state.py                 # AgentState TypedDict (citations, response_style)
│   ├── mcp_client/                  # SSE client wrappers for MCP servers
│   ├── src/
│   │   ├── ingestion/
│   │   │   ├── ingestion_src.py     # PDF ingest, SHA-256 dedup, metadata
│   │   │   ├── chunker_factory.py   # recursive | semantic | layout dispatcher
│   │   │   └── layout_chunker.py    # pdfplumber layout-aware chunker
│   │   ├── retriever/
│   │   │   ├── retriever_rag.py     # BM25 + semantic + RRF + BGE reranker
│   │   │   ├── query_expander.py    # Multi-query expansion (3 Gemma variants)
│   │   │   └── embedder_factory.py  # BGE-M3 / Google embedding selector
│   │   ├── search/                  # Pluggable web search (Tavily/Grok/Perplexity)
│   │   ├── vectorstore/             # BaseVectorStore ABC + 4 provider impls + factory
│   │   └── evaluation/              # Faithfulness scoring + RAGAS integration
│   ├── templates/contracts/         # nda.j2 · service_agreement.j2 · employment_agreement.j2
│   ├── tests/                       # 29 tests (pytest) — all passing
│   ├── main.py                      # FastAPI app, lifespan, middleware, routes
│   ├── requirements.txt
│   └── .env-example
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat.tsx             # Query input, message list, PDF download button
│   │   │   ├── MessageBubble.tsx    # Message renderer + collapsible citations panel
│   │   │   ├── Layout.tsx           # Responsive sidebar + mobile drawer
│   │   │   ├── DocumentUpload.tsx   # Drag-and-drop PDF ingest
│   │   │   └── ContractViewer.tsx   # Contract display + download
│   │   ├── hooks/
│   │   │   ├── useChat.ts           # Chat state, citations, PDF export call
│   │   │   └── useDocuments.ts      # Ingest + contract generation
│   │   ├── lib/
│   │   │   ├── api.ts               # Axios instance + Bearer token injection
│   │   │   └── utils.ts             # cn() Tailwind class merger
│   │   └── pages/                   # LoginPage · DashboardPage · DocumentsPage · ContractsPage
│   ├── .env                         # VITE_CLERK_PUBLISHABLE_KEY · VITE_API_URL
│   ├── tailwind.config.ts
│   ├── vite.config.ts
│   └── package.json
├── mcp_servers/
│   ├── search_server/               # FastMCP SSE server :8003
│   ├── filesystem_server/           # FastMCP SSE server :8001
│   └── database_server/             # FastMCP SSE server :8002
├── infra/                           # Terraform + Terragrunt (GCP + AWS)
└── README.md
```

---

## Tests

```bash
cd app && pytest tests/ -v
# 29 passed
```

Test coverage: agent relevance checking, web search provider switching, API endpoint validation (200s, 400s, 413s), faithfulness scoring edge cases, document ingestion (deduplication, file size, category classification).
