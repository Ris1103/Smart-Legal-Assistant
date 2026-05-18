import pathlib
import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        embedding_model: Embeddings,
        chroma_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        raw_dir = chroma_dir or settings.chroma_db_path
        resolved_dir = str(pathlib.Path(raw_dir).resolve())
        coll = collection_name or settings.chroma_collection_name
        logger.info(f"ChromaVectorStore: dir='{resolved_dir}', collection='{coll}'")
        self._store = Chroma(
            persist_directory=resolved_dir,
            embedding_function=embedding_model,
            collection_name=coll,
        )
        self._name = coll

    def add_documents(self, docs: list[Document]) -> list[str]:
        return self._store.add_documents(docs)

    def similarity_search(
        self, query: str, k: int = 5, filter: Optional[dict] = None
    ) -> list[Document]:
        return self._store.similarity_search(query, k=k, where=filter)

    def similarity_search_with_scores(
        self, query: str, k: int = 5, filter: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_relevance_scores(
            query, k=k, where=filter
        )

    def get_all(self) -> dict:
        result = self._store.get(include=["metadatas", "documents"])
        return {
            "documents": result.get("documents", []),
            "metadatas": result.get("metadatas", []),
        }

    def get_by_metadata(self, where: dict, limit: int = 1) -> dict:
        result = self._store.get(where=where, limit=limit)
        return {"ids": result.get("ids", [])}

    @property
    def name(self) -> str:
        return self._name
