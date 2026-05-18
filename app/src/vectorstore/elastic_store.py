import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ElasticsearchVectorStore(BaseVectorStore):
    def __init__(self, embedding_model: Embeddings):
        from langchain_elasticsearch import ElasticsearchStore

        kwargs: dict = {
            "es_url": settings.elasticsearch_url,
            "index_name": settings.elasticsearch_index_name,
            "embedding": embedding_model,
        }
        if settings.elasticsearch_api_key:
            kwargs["es_api_key"] = settings.elasticsearch_api_key

        self._store = ElasticsearchStore(**kwargs)
        self._index = settings.elasticsearch_index_name
        logger.info(f"ElasticsearchVectorStore: index='{self._index}', url='{settings.elasticsearch_url}'")

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
        resp = self._store.client.search(
            index=self._index,
            body={"query": {"match_all": {}}, "size": 10000},
        )
        hits = resp["hits"]["hits"]
        return {
            "documents": [h["_source"].get("text", "") for h in hits],
            "metadatas": [h["_source"].get("metadata", {}) for h in hits],
        }

    def get_by_metadata(self, where: dict, limit: int = 1) -> dict:
        filters = [{"term": {f"metadata.{k}.keyword": v}} for k, v in where.items()]
        resp = self._store.client.search(
            index=self._index,
            body={"query": {"bool": {"filter": filters}}, "size": limit},
        )
        return {"ids": [h["_id"] for h in resp["hits"]["hits"]]}

    @property
    def name(self) -> str:
        return self._index
