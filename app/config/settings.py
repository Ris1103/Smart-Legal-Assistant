import pathlib
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LLM / API Keys ---
    google_api_key: str
    perplexity_api_key: str = ""
    perplexity_model_name: str = "llama-3-sonar-large-32k-online"

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
