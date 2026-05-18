import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class MongoAtlasVectorStore(BaseVectorStore):
    def __init__(self, embedding_model: Embeddings):
        from langchain_mongodb import MongoDBAtlasVectorSearch
        from pymongo import MongoClient

        client = MongoClient(settings.mongodb_atlas_uri)
        collection = client[settings.mongodb_atlas_db][settings.mongodb_atlas_collection]
        self._store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedding_model,
            index_name="vector_index",
        )
        self._collection = collection
        self._name = settings.mongodb_atlas_collection
        logger.info(
            f"MongoAtlasVectorStore: db='{settings.mongodb_atlas_db}', "
            f"collection='{self._name}'"
        )

    def add_documents(self, docs: list[Document]) -> list[str]:
        return self._store.add_documents(docs)

    def similarity_search(
        self, query: str, k: int = 5, filter: Optional[dict] = None
    ) -> list[Document]:
        pre_filter = {"$and": [{k: {"$eq": v}} for k, v in filter.items()]} if filter else None
        return self._store.similarity_search(query, k=k, pre_filter=pre_filter)

    def similarity_search_with_scores(
        self, query: str, k: int = 5, filter: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        pre_filter = {"$and": [{k: {"$eq": v}} for k, v in filter.items()]} if filter else None
        return self._store.similarity_search_with_score(query, k=k, pre_filter=pre_filter)

    def get_all(self) -> dict:
        cursor = self._collection.find({}, {"text": 1, "metadata": 1, "_id": 0})
        documents, metadatas = [], []
        for doc in cursor:
            documents.append(doc.get("text", ""))
            metadatas.append(doc.get("metadata", {}))
        return {"documents": documents, "metadatas": metadatas}

    def get_by_metadata(self, where: dict, limit: int = 1) -> dict:
        query = {f"metadata.{k}": v for k, v in where.items()}
        results = list(self._collection.find(query, {"_id": 1}).limit(limit))
        return {"ids": [str(r["_id"]) for r in results]}

    @property
    def name(self) -> str:
        return self._name
