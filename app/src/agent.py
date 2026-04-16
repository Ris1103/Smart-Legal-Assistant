import logging
from typing import Dict, Any, List

import httpx
import google.generativeai as genai
import mlflow

from config.settings import settings

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


@mlflow.trace(name="perplexity_web_search")
async def search_perplexity(query: str) -> Dict[str, Any]:
    """
    Performs a search using the Perplexity AI REST API via httpx.
    """
    if not settings.perplexity_api_key:
        logger.warning("PERPLEXITY_API_KEY not set. Web search unavailable.")
        return {
            "query": query,
            "summary": "Web search is not configured. Please set PERPLEXITY_API_KEY.",
            "results": [],
            "metadata": {"source": "Perplexity AI (not configured)", "search_type": "web_search_fallback", "source_files": []},
        }

    system_message = (
        "You are an expert legal assistant specialising in Indian law. "
        "Provide a concise, accurate, and well-structured answer. "
        "Always include a disclaimer that this is for informational purposes only and not formal legal advice."
    )

    payload = {
        "model": settings.perplexity_model_name,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ],
    }
    headers = {
        "Authorization": f"Bearer {settings.perplexity_api_key}",
        "Content-Type": "application/json",
    }

    logger.info(f"Performing web search via Perplexity API for: '{query[:80]}'")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        data = response.json()
        summary = data["choices"][0]["message"]["content"]

        return {
            "query": query,
            "summary": summary,
            "results": [{"content": summary, "metadata": {"source": "Perplexity Web Search"}}],
            "metadata": {
                "search_type": "web_search_fallback",
                "source_files": ["Live Web Search"],
            },
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"Perplexity API returned HTTP {e.response.status_code}: {e.response.text}")
        return {
            "query": query,
            "summary": f"Web search failed (HTTP {e.response.status_code}). Please try again.",
            "results": [],
            "metadata": {"source": "Perplexity AI (Error)", "search_type": "web_search_fallback", "source_files": []},
        }
    except Exception as e:
        logger.error(f"Error calling Perplexity API: {e}")
        return {
            "query": query,
            "summary": f"An error occurred during the web search: {e}",
            "results": [],
            "metadata": {"source": "Perplexity AI (Error)", "search_type": "web_search_fallback", "source_files": []},
        }
