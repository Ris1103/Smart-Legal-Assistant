from typing import List

import httpx
from langchain_core.embeddings import Embeddings

from config.settings import settings as _default_settings


class HFApiEmbeddings(Embeddings):
    """Calls HuggingFace Inference API for BAAI/bge-m3 embeddings — no local model loading."""

    _URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-m3"

    def __init__(self, api_key: str):
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def _post(self, texts: List[str]) -> List[List[float]]:
        resp = httpx.post(
            self._URL,
            headers=self._headers,
            json={"inputs": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._post(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._post([text])[0]


def get_embedder(cfg=None) -> Embeddings:
    """Return an Embeddings instance based on settings.embedding_provider."""
    cfg = cfg or _default_settings
    if cfg.embedding_provider == "hf_api":
        return HFApiEmbeddings(api_key=cfg.hf_api_key)
    if cfg.embedding_provider == "bge":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=cfg.bge_model_name)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=cfg.google_embedding_model_name)
