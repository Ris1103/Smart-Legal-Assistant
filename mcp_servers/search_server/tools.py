"""Tool handlers for the Search MCP server."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "app"))

from config.settings import settings
from src.search.search_providers import get_search_provider, TavilyProvider


async def web_search(query: str, num_results: int = 5, provider: str | None = None) -> dict:
    """
    Search the web for legal information.

    Args:
        query: The search query
        num_results: Number of results to return (supported by Tavily)
        provider: Optional provider override ("perplexity" | "tavily" | "grok")

    Returns:
        {"answer": str, "sources": list[str], "provider": str}
    """
    cfg = settings
    if provider:
        # Temporarily override provider without mutating global settings
        class _Cfg:
            pass
        _cfg = _Cfg()
        for attr in vars(cfg.__class__).keys():
            if not attr.startswith("_"):
                try:
                    setattr(_cfg, attr, getattr(cfg, attr))
                except Exception:
                    pass
        _cfg.web_search_provider = provider
        search_provider = get_search_provider(_cfg)
    else:
        search_provider = get_search_provider(cfg)

    # Pass num_results to Tavily if supported
    if isinstance(search_provider, TavilyProvider):
        search_provider._max_results = num_results

    answer = await search_provider.search(query)
    return {
        "answer": answer,
        "sources": [],
        "provider": provider or cfg.web_search_provider,
    }
