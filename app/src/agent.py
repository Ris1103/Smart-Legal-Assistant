import logging
from typing import Dict, Any, List

import google.generativeai as genai
import mlflow

from config.settings import settings
from src.search.search_providers import get_search_provider

logger = logging.getLogger(__name__)


def is_context_relevant(
    query: str, documents: List, model: genai.GenerativeModel
) -> bool:
    """
    Uses an LLM to determine if the retrieved documents are relevant to the query.
    Returns True if context is relevant, False otherwise.
    """
    if not documents:
        logger.info("No documents found, context is not relevant.")
        return False

    context = "\n\n".join([doc.page_content for doc in documents])

    # Truncate outside the f-string to avoid literal comment text in the prompt
    context_snippet = context[:32000]

    prompt = f"""You are a relevance-checking assistant. Your task is to determine if the provided CONTEXT contains information that can directly answer the USER QUERY.
Respond with only the single word 'yes' or 'no'.

---CONTEXT---
{context_snippet}

---USER QUERY---
{query}
"""

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip().lower()
        logger.info(f"Relevance check for query '{query[:60]}' returned: '{answer}'")
        return answer == "yes"
    except Exception as e:
        logger.error(f"Error during relevance check: {e}. Defaulting to 'not relevant'.")
        return False


@mlflow.trace(name="web_search")
async def search_perplexity(query: str) -> Dict[str, Any]:
    """
    Performs a web search using the configured provider (WEB_SEARCH_PROVIDER).
    """
    provider_name = settings.web_search_provider
    logger.info(f"Performing web search via {provider_name} for: '{query[:80]}'")
    try:
        provider = get_search_provider(settings)
        summary = await provider.search(query)
        return {
            "query": query,
            "summary": summary,
            "results": [{"content": summary, "metadata": {"source": f"{provider_name.capitalize()} Web Search"}}],
            "metadata": {
                "search_type": "web_search_fallback",
                "source_files": ["Live Web Search"],
            },
        }
    except ValueError as e:
        logger.warning(f"Web search not configured: {e}")
        return {
            "query": query,
            "summary": f"Web search is not configured: {e}",
            "results": [],
            "metadata": {"source": "Web Search (not configured)", "search_type": "web_search_fallback", "source_files": []},
        }
    except Exception as e:
        logger.error(f"Web search error ({provider_name}): {e}")
        return {
            "query": query,
            "summary": f"An error occurred during the web search: {e}",
            "results": [],
            "metadata": {"source": f"{provider_name} (Error)", "search_type": "web_search_fallback", "source_files": []},
        }
