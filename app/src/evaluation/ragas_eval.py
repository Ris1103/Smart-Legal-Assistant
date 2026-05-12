"""
RAGAS-based evaluation wrapper.
Activated when settings.evaluation_framework == "ragas".
Returns a dict with faithfulness, answer_relevancy, context_precision,
context_recall — all floats in [0, 1].
"""
import logging
from typing import Any, Dict, List

from src.evaluation.evaluation import FAITHFULNESS_ERROR_SENTINEL

logger = logging.getLogger(__name__)


def evaluate_with_ragas(
    query: str,
    answer: str,
    contexts: List[str],
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on a single query/answer/context triple.

    Returns a dict with keys:
        faithfulness, answer_relevancy, context_precision, context_recall
    On any failure returns sentinel values so callers can detect errors.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
            }
        )
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        row = result.to_pandas().iloc[0].to_dict()
        return {
            "faithfulness": float(row.get("faithfulness", FAITHFULNESS_ERROR_SENTINEL)),
            "answer_relevancy": float(row.get("answer_relevancy", FAITHFULNESS_ERROR_SENTINEL)),
            "context_precision": float(row.get("context_precision", FAITHFULNESS_ERROR_SENTINEL)),
            "context_recall": float(row.get("context_recall", FAITHFULNESS_ERROR_SENTINEL)),
        }
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {
            "faithfulness": FAITHFULNESS_ERROR_SENTINEL,
            "answer_relevancy": FAITHFULNESS_ERROR_SENTINEL,
            "context_precision": FAITHFULNESS_ERROR_SENTINEL,
            "context_recall": FAITHFULNESS_ERROR_SENTINEL,
        }
