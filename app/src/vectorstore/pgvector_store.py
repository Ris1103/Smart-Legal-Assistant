import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

_COLLECTION = "legal_documents"


class PgVectorStore(BaseVectorStore):
    def __init__(self, embedding_model: Embeddings):
        from langchain_postgres import PGVector

        self._store = PGVector(
            embeddings=embedding_model,
            collection_name=_COLLECTION,
            connection=settings.pgvector_dsn,
        )
        logger.info(f"PgVectorStore: collection='{_COLLECTION}'")

    def add_documents(self, docs: list[Document]) -> list[str]:
        return self._store.add_documents(docs)

    def similarity_search(
        self, query: str, k: int = 5, filter: Optional[dict] = None
    ) -> list[Document]:
        return self._store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_scores(
        self, query: str, k: int = 5, filter: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_score(query, k=k, filter=filter)

    def get_all(self) -> dict:
        # PGVector doesn't expose a bulk-get; query with a broad filter
        results = self._store.similarity_search("", k=10_000)
        return {
            "documents": [d.page_content for d in results],
            "metadatas": [d.metadata for d in results],
        }

    def get_by_metadata(self, where: dict, limit: int = 1) -> dict:
        results = self._store.similarity_search("", k=limit, filter=where)
        # Use page_content as a stand-in ID — good enough for duplicate detection
        return {"ids": [d.page_content[:64] for d in results]}

    @property
    def name(self) -> str:
        return _COLLECTION
