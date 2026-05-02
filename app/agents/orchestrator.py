"""
Orchestrator node — classifies the user query into a legal domain and
detects contract generation intent.

Routing rules
-------------
- confidence >= 0.6 AND intent == "query"    → domain agent node
- confidence >= 0.6 AND intent == "contract" → contract agent node
- confidence < 0.6                            → web_research agent node
"""
import json
import logging
import re
from datetime import datetime

import google.generativeai as genai

from graph.state import AgentState

logger = logging.getLogger(__name__)

VALID_DOMAINS = {
    "GST",
    "Income Tax",
    "Company Law",
    "Labour Law",
    "Criminal Law",
    "General",
}

_CLASSIFICATION_PROMPT = """\
You are a legal domain classifier for Indian law. Analyse the USER QUERY
and respond with a single JSON object — nothing else.

Domains: GST, Income Tax, Company Law, Labour Law, Criminal Law, General

JSON schema:
{{
  "domain": "<one of the domains above>",
  "confidence": <float 0.0–1.0>,
  "intent": "<query | contract>"
}}

Rules:
- Use "contract" intent only when the user explicitly asks to draft,
  create, or generate a legal document/agreement/contract.
- Use "General" when the query does not clearly fit any specific domain.
- confidence < 0.6 means you are unsure; the system will fall back to
  a live web search.

USER QUERY: {query}
"""


def orchestrator_node(
    state: AgentState, model: genai.GenerativeModel
) -> AgentState:
    """
    Classify the query and populate domain, confidence, intent.
    """
    query = state["query"]
    logger.info(f"Orchestrator classifying: '{query[:80]}'")

    prompt = _CLASSIFICATION_PROMPT.format(query=query)

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        domain = parsed.get("domain", "General")
        confidence = float(parsed.get("confidence", 0.5))
        intent = parsed.get("intent", "query")

        # Normalise domain
        if domain not in VALID_DOMAINS:
            domain = "General"
            confidence = min(confidence, 0.5)

        logger.info(
            f"Classified as domain='{domain}', confidence={confidence:.2f}, "
            f"intent='{intent}'"
        )
    except Exception as e:
        logger.error(
            f"Orchestrator classification failed: {e}. Defaulting to General."
        )
        domain, confidence, intent = "General", 0.4, "query"

    return {
        **state,
        "domain": domain,
        "confidence": confidence,
        "intent": intent,
        "metadata": {
            **(state.get("metadata") or {}),
            "orchestrator_domain": domain,
            "orchestrator_confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        },
    }


def route_after_orchestrator(state: AgentState) -> str:
    """
    LangGraph conditional edge function.
    Returns the name of the next node.
    """
    if state.get("confidence", 0.0) < 0.6:
        return "web_research"
    if state.get("intent") == "contract":
        return "contract"
    return "domain_agent"
