"""
Multi-query expansion — generates alternative phrasings of a legal query
to improve retrieval recall. Fails gracefully to the original query.
"""
import json
import logging
from typing import List

import google.generativeai as genai

logger = logging.getLogger(__name__)

_PROMPT = (
    "You are a legal query reformulation assistant. "
    "Generate exactly 3 alternative phrasings of the following legal query "
    "that capture the same intent but use different vocabulary and structure. "
    "Return ONLY a JSON array of 3 strings, no explanation.\n\n"
    "Query: {query}\n\n"
    "Response (JSON array only):"
)


def expand_query(query: str, model: genai.GenerativeModel) -> List[str]:
    """
    Returns [original_query, variant1, variant2, variant3].
    Falls back to [original_query] on any failure.
    """
    try:
        response = model.generate_content(_PROMPT.format(query=query))
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        variants = json.loads(text)
        if isinstance(variants, list) and len(variants) >= 1:
            all_queries = [query] + [str(v) for v in variants[:3]]
            logger.info(f"Query expanded to {len(all_queries)} variants.")
            return all_queries
    except Exception as e:
        logger.warning(f"Query expansion failed ({e}), using original query only.")
    return [query]
