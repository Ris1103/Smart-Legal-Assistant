import pathlib
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM / API Keys ---
    google_api_key: str
    perplexity_api_key: str = ""
    perplexity_model_name: str = "llama-3-sonar-large-32k-online"
    tavily_api_key: str = ""
    grok_api_key: str = ""

    # --- Service Config ---
    fastapi_url: str = "http://localhost:8000"
    service_api_key: str = ""  # X-API-Key for endpoint auth; empty = auth disabled

    # --- Storage ---
    chroma_db_path: str = str(
        pathlib.Path(__file__).resolve().parent.parent / "chroma_db"
    )
    chroma_collection_name: str = "legal_documents"

    # --- Ingestion limits ---
    max_file_size_mb: int = 50

    # --- Models ---
    generative_model_name: str = "gemma-4-26b-a4b-it"
    embedding_provider: str = "google"   # "google" | "bge"
    google_embedding_model_name: str = "models/gemini-embedding-001"
    bge_model_name: str = "BAAI/bge-m3"
    grok_model_name: str = "grok-3"

    # --- Chunking ---
    chunk_strategy: str = "recursive"   # "recursive" | "semantic"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retrieval ---
    semantic_weight: float = 0.7
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    top_k_retrieval: int = 8

    # --- Context Compression ---
    context_compression_enabled: bool = False
    compression_similarity_threshold: float = 0.5

    # --- Evaluation ---
    evaluation_framework: str = "custom"   # "custom" | "ragas"

    # --- Web Search ---
    web_search_provider: str = "perplexity"   # "perplexity" | "tavily" | "grok"

    # --- Vector Store ---
    vector_store_provider: str = "chromadb"  # chromadb | mongodb_atlas | pgvector | pinecone | elasticsearch

    # MongoDB Atlas
    mongodb_atlas_uri: str = ""
    mongodb_atlas_db: str = "legal_advisor"
    mongodb_atlas_collection: str = "legal_documents"

    # pgvector (Neon or any Postgres with pgvector extension)
    pgvector_dsn: str = ""  # postgres://user:pass@host/db
    pgvector_collection_name: str = "legal_documents"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "legal-advisor"

    # Elasticsearch (local Docker or Elastic Cloud)
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_api_key: str = ""           # empty = no auth (local Docker)
    elasticsearch_index_name: str = "legal-advisor"

    # --- Auth (Clerk) ---
    clerk_secret_key: str = ""          # sk_test_... from Clerk dashboard
    clerk_publishable_key: str = ""     # pk_test_... (used by frontend)
    clerk_webhook_secret: str = ""      # whsec_... for webhook verification

    # --- Database (Neon PostgreSQL) ---
    database_url: str = ""              # postgresql+asyncpg://user:pass@host/db
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10

    # --- Rate Limiting ---
    rate_limit_per_minute: int = 10

    # --- MCP Integration ---
    mcp_enabled: bool = False
    mcp_search_server_url: str = "http://localhost:8003"
    mcp_filesystem_server_url: str = "http://localhost:8001"
    mcp_database_server_url: str = "http://localhost:8002"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
