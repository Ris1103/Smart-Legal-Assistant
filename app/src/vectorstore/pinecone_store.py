import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class PineconeVectorStore(BaseVectorStore):
    def __init__(self, embedding_model: Embeddings):
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore as _LCPinecone

        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        self._store = _LCPinecone(index=index, embedding=embedding_model, text_key="text")
        self._index = index
        logger.info(f"PineconeVectorStore: index='{settings.pinecone_index_name}'")

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
        # Pinecone free tier: fetch all via a stats-then-fetch approach
        stats = self._index.describe_index_stats()
        total = stats.total_vector_count
        if total == 0:
            return {"documents": [], "metadatas": []}
        # Retrieve up to 10k vectors using a broad query
        results = self._store.similarity_search("", k=min(total, 10_000))
        return {
            "documents": [d.page_content for d in results],
            "metadatas": [d.metadata for d in results],
        }

    def get_by_metadata(self, where: dict, limit: int = 1) -> dict:
        results = self._store.similarity_search("", k=limit, filter=where)
        return {"ids": [d.metadata.get("id", d.page_content[:64]) for d in results]}

    @property
    def name(self) -> str:
        return settings.pinecone_index_name
