from langchain_core.embeddings import Embeddings

from config.settings import settings as _default_settings


def get_embedder(cfg=None) -> Embeddings:
    """Return an Embeddings instance based on settings.embedding_provider."""
    cfg = cfg or _default_settings
    if cfg.embedding_provider == "bge":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=cfg.bge_model_name)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
