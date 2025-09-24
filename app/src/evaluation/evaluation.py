import logging
import tempfile
import os
from typing import List, Dict, Any

import mlflow
import google.generativeai as genai
from langchain.schema import Document

logger = logging.getLogger(__name__)
mlflow.set_experiment("Legal_RAG_Assistant")


def calculate_faithfulness(
    query: str, retrieved_docs: List[Dict[str, Any]], summary: str, model: genai.GenerativeModel
) -> float:
    """
    Uses an LLM to evaluate if the summary is factually supported by the retrieved context.
    Returns a score between 0.0 and 1.0.
    """
    logger.info("Calculating faithfulness score...")
    if not retrieved_docs:
        return 0.0

    context = "\n\n".join([doc["content"] for doc in retrieved_docs])

    prompt = f"""
    You are a meticulous fact-checker. Your task is to evaluate a SUMMARY based on a provided CONTEXT.
    Analyze each statement in the SUMMARY and determine if it is factually supported by the CONTEXT.
    Count the number of statements in the SUMMARY that are directly supported by the CONTEXT.
    Count the total number of statements in the SUMMARY.

    Finally, provide a single floating-point number representing the faithfulness score (supported statements / total statements).
    The output should be a single number between 0.0 and 1.0.

    ---CONTEXT---
    {context[:8000]}

    ---SUMMARY---
    {summary}

    ---SCORE---
    """

    try:
        response = model.generate_content(prompt)
        score = float(response.text.strip())
        logger.info(f"Faithfulness score calculated: {score}")
        return score
    except (ValueError, TypeError) as e:
        logger.error(f"Could not parse faithfulness score from LLM response: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error during faithfulness calculation: {e}")
        return 0.0
