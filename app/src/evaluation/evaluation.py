import logging
from typing import List, Dict, Any

import mlflow
import google.generativeai as genai

logger = logging.getLogger(__name__)
mlflow.set_experiment("Legal_RAG_Assistant")

# Sentinel returned when evaluation itself errors out, so callers can
# distinguish "genuinely unfaithful (0.0)" from "could not evaluate (-1.0)".
FAITHFULNESS_ERROR_SENTINEL = -1.0


def calculate_faithfulness(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    summary: str,
    model: genai.GenerativeModel,
) -> float:
    """
    Uses an LLM to evaluate if the summary is factually supported by the
    retrieved context.

    Returns:
        float: Score between 0.0 and 1.0 if successful.
               FAITHFULNESS_ERROR_SENTINEL (-1.0) if the LLM response
               cannot be parsed, so callers can log the failure separately.
    """
    logger.info("Calculating faithfulness score...")
    if not retrieved_docs:
        return 0.0

    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    prompt = (
        "You are a meticulous fact-checker. Evaluate a SUMMARY against a "
        "CONTEXT.\n"
        "Count statements in the SUMMARY that are directly supported by the "
        "CONTEXT.\n"
        "Output a single float between 0.0 and 1.0 "
        "(supported / total statements).\n\n"
        f"---CONTEXT---\n{context[:8000]}\n\n"
        f"---SUMMARY---\n{summary}\n\n"
        "---SCORE---"
    )

    try:
        response = model.generate_content(prompt)
        score = float(response.text.strip())
        logger.info(f"Faithfulness score calculated: {score}")
        return score
    except (ValueError, TypeError) as e:
        logger.error(
            f"Could not parse faithfulness score from LLM response: {e}"
        )
        return FAITHFULNESS_ERROR_SENTINEL
    except Exception as e:
        logger.error(f"Error during faithfulness calculation: {e}")
        return FAITHFULNESS_ERROR_SENTINEL
