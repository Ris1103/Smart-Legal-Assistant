import os
import logging
from typing import Dict, Any, List
from langchain.schema import Document
import google.generativeai as genai
from fastapi.concurrency import run_in_threadpool
from perplexipy import PerplexityClient
import mlflow

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
PPLX_MODEL_NAME = os.getenv("PERPLEXITY_MODEL_NAME", "llama-3-sonar-large-32k-online")
PPLX_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Initialize the PerplexiPy client
perplexity_client = None
if PPLX_API_KEY:
    try:
        # --- FIX: Pass the API key directly to the constructor ---
        perplexity_client = PerplexityClient(key=PPLX_API_KEY)
        perplexity_client.model = PPLX_MODEL_NAME  # Set the model
        logger.info(f"PerplexiPy client initialized. Using model: {PPLX_MODEL_NAME}")
    except Exception as e:
        logger.warning(f"Could not initialize PerplexiPy client: {e}")
else:
    logger.warning("PERPLEXITY_API_KEY not found. Perplexity web search will be disabled.")


def is_context_relevant(
    query: str, documents: List[Document], model: genai.GenerativeModel
) -> bool:
    """
    Uses an LLM to determine if the retrieved documents are relevant to the query.
    """
    if not documents:
        logger.info("No documents found, context is not relevant.")
        return False

    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = f"""
    You are a relevance-checking assistant. Your task is to determine if the provided CONTEXT contains information that can directly answer the USER QUERY.
    Respond with only the single word 'yes' or 'no'.

    ---CONTEXT---
    {context[:4000]}  # Limit context to avoid overly long prompts

    ---USER QUERY---
    {query}
    """

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip().lower()
        logger.info(f"Relevance check for query '{query}' returned: '{answer}'")
        return answer == "yes"
    except Exception as e:
        logger.error(f"Error during relevance check: {e}. Defaulting to 'not relevant'.")
        return False


@mlflow.trace(name="perplexity_web_search")
async def search_perplexity(query: str) -> Dict[str, Any]:
    """
    Performs a search using the PerplexiPy library in a non-blocking way.
    """
    if not perplexity_client:
        return {
            "query": query,
            "summary": "Perplexity client is not configured. Cannot perform web search.",
            "results": [],
            "metadata": {"source": "Perplexity AI (Error)"},
        }

    logger.info(f"Performing web search with PerplexiPy for: '{query}'")
    try:
        full_prompt = (
            "You are an expert legal assistant. Provide a concise and accurate answer. "
            f"Here is the user's query: {query}"
        )
        summary = await run_in_threadpool(perplexity_client.query, full_prompt)

        return {
            "query": query,
            "summary": summary,
            "results": [{"content": summary, "metadata": {"source": "Perplexity Web Search"}}],
            "metadata": {
                "search_type": "web_search_fallback",
                "source_files": ["Live Web Search"],
            },
        }

    except Exception as e:
        logger.error(f"Error calling PerplexiPy API: {e}")
        return {
            "query": query,
            "summary": f"An error occurred during the web search: {e}",
            "results": [],
            "metadata": {"source": "Perplexity AI (Error)"},
        }
