"""
Web Research Agent — falls back to Perplexity AI when local documents
are insufficient (orchestrator confidence < 0.6 or no local results).
Replaces the old search_perplexity() function in src/agent.py.
"""
import logging
from datetime import datetime
from typing import Optional

import httpx

from config.settings import settings
from graph.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM_MESSAGE = (
    "You are an expert legal assistant specialising in Indian law. "
    "Provide a concise, accurate, and well-structured answer. "
    "Always include a disclaimer that this is for informational purposes "
    "only and not formal legal advice."
)


async def web_research_node(state: AgentState) -> AgentState:
    """
    LangGraph async node — calls Perplexity AI and populates state.
    """
    query = state["query"]
    logger.info(f"WebResearchAgent: searching for '{query[:80]}'")

    if not settings.perplexity_api_key:
        logger.warning("PERPLEXITY_API_KEY not set — web search unavailable.")
        return {
            **state,
            "summary": (
                "Web search is not configured. "
                "Please set PERPLEXITY_API_KEY in your environment."
            ),
            "retrieved_docs": [],
            "source_files": ["Web Search (not configured)"],
            "search_type": "web",
        }

    payload = {
        "model": settings.perplexity_model_name,
        "messages": [
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": query},
        ],
    }
    headers = {
        "Authorization": f"Bearer {settings.perplexity_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()

        data = resp.json()
        summary = data["choices"][0]["message"]["content"]

        return {
            **state,
            "summary": summary,
            "retrieved_docs": [
                {
                    "content": summary,
                    "metadata": {"source": "Perplexity Web Search"},
                }
            ],
            "source_files": ["Live Web Search"],
            "search_type": "web",
            "metadata": {
                **(state.get("metadata") or {}),
                "search_type": "web_search",
                "timestamp": datetime.now().isoformat(),
            },
        }

    except httpx.HTTPStatusError as e:
        logger.error(
            f"Perplexity HTTP {e.response.status_code}: {e.response.text}"
        )
        return {
            **state,
            "summary": (
                f"Web search failed (HTTP {e.response.status_code}). "
                "Please try again."
            ),
            "retrieved_docs": [],
            "source_files": [],
            "search_type": "web",
            "error": str(e),
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
