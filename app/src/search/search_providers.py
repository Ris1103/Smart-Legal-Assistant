"""
Pluggable web search providers (Strategy pattern).
Add a new provider by subclassing WebSearchProvider and registering it
in get_search_provider().
"""
import logging
from abc import ABC, abstractmethod

import httpx

from config.settings import settings as _default_settings

logger = logging.getLogger(__name__)

_LEGAL_SYSTEM_MESSAGE = (
    "You are an expert legal assistant specialising in Indian law. "
    "Provide a concise, accurate, and well-structured answer. "
    "Always include a disclaimer that this is for informational purposes "
    "only and not formal legal advice."
)


class WebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str) -> str:
        """Return a text answer for the given query."""


class PerplexityProvider(WebSearchProvider):
    def __init__(self, api_key: str, model_name: str):
        self._api_key = api_key
        self._model = model_name

    async def search(self, query: str) -> str:
        if not self._api_key:
            raise ValueError("PERPLEXITY_API_KEY is not set.")
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _LEGAL_SYSTEM_MESSAGE},
                {"role": "user", "content": query},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class TavilyProvider(WebSearchProvider):
    def __init__(self, api_key: str):
        self._api_key = api_key

    async def search(self, query: str) -> str:
        if not self._api_key:
            raise ValueError("TAVILY_API_KEY is not set.")
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "max_results": 5,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json=payload,
            )
            resp.raise_for_status()
        data = resp.json()
        # Prefer the synthesised answer; fall back to concatenated snippets.
        if data.get("answer"):
            return data["answer"]
        snippets = [r.get("content", "") for r in data.get("results", [])]
        return "\n\n".join(snippets) or "No results found."


class GrokProvider(WebSearchProvider):
    def __init__(self, api_key: str, model_name: str):
        self._api_key = api_key
        self._model = model_name

    async def search(self, query: str) -> str:
        if not self._api_key:
            raise ValueError("GROK_API_KEY is not set.")
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _LEGAL_SYSTEM_MESSAGE},
                {"role": "user", "content": query},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def get_search_provider(cfg=None) -> WebSearchProvider:
    """Factory: return the configured web search provider."""
    cfg = cfg or _default_settings
    match cfg.web_search_provider:
        case "tavily":
            return TavilyProvider(cfg.tavily_api_key)
        case "grok":
            return GrokProvider(cfg.grok_api_key, cfg.grok_model_name)
        case _:
            return PerplexityProvider(cfg.perplexity_api_key, cfg.perplexity_model_name)
