"""
QA Agent — evaluates the domain agent's response for:
  1. Faithfulness (score >= 0.7)
  2. Presence of a legal disclaimer
  3. Completeness (LLM judge)

On failure (up to MAX_RETRIES), it routes back to the domain agent
with structured qa_feedback. After MAX_RETRIES it passes through.
"""
import logging
import re
from datetime import datetime

import google.generativeai as genai

from config.settings import settings
from graph.state import AgentState
from src.evaluation.evaluation import (
    FAITHFULNESS_ERROR_SENTINEL,
    calculate_faithfulness,
)
from src.evaluation.ragas_eval import evaluate_with_ragas

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
FAITHFULNESS_THRESHOLD = 0.7

_DISCLAIMER_PATTERN = re.compile(
    r"informational purposes only|not (?:formal |constitute )?legal advice|"
    r"consult (?:a )?(?:qualified )?(?:lawyer|advocate|legal professional)",
    re.IGNORECASE,
)

_COMPLETENESS_PROMPT = """\
You are a legal QA reviewer. Does the RESPONSE adequately answer the QUERY?
Reply with only JSON: {{"complete": true/false, "feedback": "<one sentence>"}}

QUERY: {query}
RESPONSE: {response}
"""


class QAAgent:
    def __init__(self, model: genai.GenerativeModel):
        self.model = model

    def __call__(self, state: AgentState) -> AgentState:
        retries = state.get("qa_retries", 0)
        query = state.get("query", "")
        summary = state.get("summary", "")
        docs = state.get("retrieved_docs", [])
        search_type = state.get("search_type", "local")

        logger.info(f"QAAgent evaluating (retry #{retries}).")

        # Skip deep QA for web search results (no local context to compare)
        if search_type == "web":
            return {**state, "qa_passed": True}

        issues = []

        # 1. Faithfulness (+ optional RAGAS metrics)
        ragas_scores: dict = {}
        if settings.evaluation_framework == "ragas":
            contexts = [d["content"] for d in docs if d.get("content")]
            ragas_scores = evaluate_with_ragas(query, summary, contexts)
            faithfulness = ragas_scores.get("faithfulness", FAITHFULNESS_ERROR_SENTINEL)
        else:
            faithfulness = calculate_faithfulness(
                query=query,
                retrieved_docs=docs,
                summary=summary,
                model=self.model,
            )

        if faithfulness == FAITHFULNESS_ERROR_SENTINEL:
            logger.warning("QA: faithfulness eval errored — skipping check.")
        elif faithfulness < FAITHFULNESS_THRESHOLD:
            issues.append(
                f"Low faithfulness ({faithfulness:.2f} < "
                f"{FAITHFULNESS_THRESHOLD}): response may contain "
                "information not in the source documents."
            )

        # 2. Disclaimer presence
        if not _DISCLAIMER_PATTERN.search(summary):
            issues.append(
                "Missing legal disclaimer. The response must include a "
                "statement that it is for informational purposes only and "
                "not formal legal advice."
            )

        # 3. Completeness
        completeness_feedback = self._check_completeness(query, summary)
        if completeness_feedback:
            issues.append(completeness_feedback)

        qa_passed = len(issues) == 0 or retries >= MAX_RETRIES

        if qa_passed:
            if retries >= MAX_RETRIES and issues:
                logger.warning(
                    f"QA: max retries reached with issues: {issues}"
                )
            eval_meta = {"faithfulness_score": faithfulness}
            if ragas_scores:
                eval_meta.update(ragas_scores)
            return {
                **state,
                "qa_passed": True,
                "faithfulness_score": faithfulness,
                "ragas_scores": ragas_scores or None,
                "metadata": {
                    **(state.get("metadata") or {}),
                    "qa_retries": retries,
                    "qa_issues": issues,
                    **eval_meta,
                    "timestamp": datetime.now().isoformat(),
                },
            }

        qa_feedback = " | ".join(issues)
        logger.info(f"QA failed (retry {retries + 1}): {qa_feedback}")

        return {
            **state,
            "qa_passed": False,
            "qa_feedback": qa_feedback,
            "qa_retries": retries + 1,
            "faithfulness_score": faithfulness,
        }

    def _check_completeness(
        self, query: str, response: str
    ) -> str:
        """Returns a feedback string if incomplete, empty string if OK."""
        prompt = _COMPLETENESS_PROMPT.format(
            query=query, response=response[:4000]
        )
        try:
            import json
            import re as _re
            raw = self.model.generate_content(prompt).text.strip()
            raw = _re.sub(r"^```(?:json)?\s*", "", raw)
            raw = _re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
            if not result.get("complete", True):
                return result.get("feedback", "Response appears incomplete.")
        except Exception as e:
            logger.warning(f"QA completeness check failed: {e}")
        return ""


def route_after_qa(state: AgentState) -> str:
    """
    LangGraph conditional edge after QA node.
    Returns "domain_agent" to retry or "end" to finish.
    """
    if state.get("qa_passed"):
        return "end"
    return "domain_agent"
