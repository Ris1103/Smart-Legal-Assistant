from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """Abstract interface for all vector store providers."""

    @abstractmethod
    def add_documents(self, docs: list[Document]) -> list[str]:
        """Add documents and return their IDs."""
        ...

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 5, filter: Optional[dict] = None
    ) -> list[Document]:
        ...

    @abstractmethod
    def similarity_search_with_scores(
        self, query: str, k: int = 5, filter: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        ...

    @abstractmethod
    def get_all(self) -> dict:
        """Return {"documents": [...], "metadatas": [...]} for all stored chunks."""
        ...

    @abstractmethod
    def get_by_metadata(self, where: dict, limit: int = 1) -> dict:
        """Return {"ids": [...]} for chunks matching the metadata filter."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Collection / index name."""
        ...
