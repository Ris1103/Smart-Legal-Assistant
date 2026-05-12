"""
Web Research Agent — falls back to a configured web search provider when
local documents are insufficient (orchestrator confidence < 0.6 or no
local results).
"""
import logging
from datetime import datetime

from config.settings import settings
from graph.state import AgentState
from src.search.search_providers import get_search_provider

logger = logging.getLogger(__name__)


async def web_research_node(state: AgentState) -> AgentState:
    """LangGraph async node — calls the configured web search provider."""
    query = state["query"]
    provider_name = settings.web_search_provider
    logger.info(f"WebResearchAgent: provider={provider_name}, query='{query[:80]}'")

    provider = get_search_provider(settings)

    try:
        summary = await provider.search(query)
        return {
            **state,
            "summary": summary,
            "retrieved_docs": [
                {
                    "content": summary,
                    "metadata": {"source": f"{provider_name.capitalize()} Web Search"},
                }
            ],
            "source_files": ["Live Web Search"],
            "search_type": "web",
            "metadata": {
                **(state.get("metadata") or {}),
                "search_type": "web_search",
                "web_search_provider": provider_name,
                "timestamp": datetime.now().isoformat(),
            },
        }

    except ValueError as e:
        # API key not configured
        logger.warning(f"WebResearchAgent: {e}")
        return {
            **state,
            "summary": (
                f"Web search is not configured ({e}). "
                "Please set the relevant API key in your environment."
            ),
            "retrieved_docs": [],
            "source_files": ["Web Search (not configured)"],
            "search_type": "web",
        }
    except Exception as e:
        logger.error(f"WebResearchAgent error: {e}")
        return {
            **state,
            "summary": f"An error occurred during the web search: {e}",
            "retrieved_docs": [],
            "source_files": [],
            "search_type": "web",
            "error": str(e),
        }
