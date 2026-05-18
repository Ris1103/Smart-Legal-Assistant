import logging
from typing import Optional

from langchain_core.embeddings import Embeddings

from config.settings import settings
from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

_PROVIDERS = ("chromadb", "mongodb_atlas", "pgvector", "pinecone", "elasticsearch")


class VectorStoreFactory:
    _instances: dict[str, BaseVectorStore] = {}

    @classmethod
    def get_instance(
        cls,
        provider: Optional[str] = None,
        embedding_model: Optional[Embeddings] = None,
    ) -> BaseVectorStore:
        provider = provider or settings.vector_store_provider
        if provider not in cls._instances:
            cls._instances[provider] = cls._create(provider, embedding_model)
        return cls._instances[provider]

    @classmethod
    def reset(cls, provider: Optional[str] = None) -> None:
        """Force a fresh instance on next call. Used in tests."""
        if provider:
            cls._instances.pop(provider, None)
        else:
            cls._instances.clear()

    @classmethod
    def _create(
        cls, provider: str, embedding_model: Optional[Embeddings]
    ) -> BaseVectorStore:
        if embedding_model is None:
            from src.retriever.embedder_factory import get_embedder
            embedding_model = get_embedder(settings)

        logger.info(f"VectorStoreFactory: creating provider='{provider}'")

        if provider == "chromadb":
            from src.vectorstore.chroma_store import ChromaVectorStore
            return ChromaVectorStore(embedding_model)

        if provider == "mongodb_atlas":
            from src.vectorstore.mongo_atlas_store import MongoAtlasVectorStore
            return MongoAtlasVectorStore(embedding_model)

        if provider == "pgvector":
            from src.vectorstore.pgvector_store import PgVectorStore
            return PgVectorStore(embedding_model)

        if provider == "pinecone":
            from src.vectorstore.pinecone_store import PineconeVectorStore
            return PineconeVectorStore(embedding_model)

        if provider == "elasticsearch":
            from src.vectorstore.elastic_store import ElasticsearchVectorStore
            return ElasticsearchVectorStore(embedding_model)

        raise ValueError(
            f"Unknown vector_store_provider '{provider}'. "
            f"Valid options: {_PROVIDERS}"
        )
