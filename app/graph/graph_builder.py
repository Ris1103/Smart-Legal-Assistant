"""
Builds and compiles the LangGraph agent graph.

Graph topology
--------------
  [orchestrator]
       │
       ├─ confidence < 0.6  ──────────────→ [web_research]
       ├─ intent == contract ─────────────→ [contract]
       └─ otherwise ─────────────────────→ [domain_agent]
                                                  │
                                            [qa_agent]
                                                  │
                                    ┌─────────────┴──────────────┐
                                    │ qa_passed                  │ !qa_passed
                                    ▼                            ▼
                                  [END]                   [domain_agent]
                                                          (max 2 retries)
  [web_research] ─────────────────────────────────────→ [END]
  [contract]      ─────────────────────────────────────→ [END]
"""
import functools
import logging
from typing import Optional

import google.generativeai as genai
from langgraph.graph import END, StateGraph

from agents.contract_agent import ContractAgent
from agents.domain.company_law_agent import CompanyLawAgent
from agents.domain.criminal_law_agent import CriminalLawAgent
from agents.domain.general_agent import GeneralLegalAgent
from agents.domain.gst_agent import GSTAgent
from agents.domain.income_tax_agent import IncomeTaxAgent
from agents.domain.labour_law_agent import LabourLawAgent
from agents.orchestrator import orchestrator_node, route_after_orchestrator
from agents.qa_agent import QAAgent, route_after_qa
from agents.web_research_agent import web_research_node
from graph.state import AgentState
from src.retriever.retriever_rag import HybridRAGPipeline

logger = logging.getLogger(__name__)

_DOMAIN_TO_AGENT_CLASS = {
    "GST": GSTAgent,
    "Income Tax": IncomeTaxAgent,
    "Company Law": CompanyLawAgent,
    "Labour Law": LabourLawAgent,
    "Criminal Law": CriminalLawAgent,
    "General": GeneralLegalAgent,
}


def _domain_agent_node(
    state: AgentState,
    pipeline: HybridRAGPipeline,
    generative_model: genai.GenerativeModel,
) -> AgentState:
    """
    Dispatch to the correct domain specialist based on state['domain'].
    Falls back to GeneralLegalAgent for unknown domains.
    """
    domain = state.get("domain", "General")
    agent_cls = _DOMAIN_TO_AGENT_CLASS.get(domain, GeneralLegalAgent)
    agent = agent_cls(pipeline=pipeline, generative_model=generative_model)
    logger.info(f"Dispatching to {agent_cls.__name__} for domain '{domain}'")
    return agent(state)


def build_graph(
    pipeline: HybridRAGPipeline,
    generative_model: genai.GenerativeModel,
    orchestrator_model: Optional[genai.GenerativeModel] = None,
):
    """
    Construct and compile the LangGraph StateGraph.

    Args:
        pipeline: The shared HybridRAGPipeline instance.
        generative_model: Gemma model for domain agents and QA.
        orchestrator_model: Optionally a lighter/faster model for the
                            orchestrator. Defaults to generative_model.
    """
    orch_model = orchestrator_model or generative_model

    graph = StateGraph(AgentState)

    # ------------------------------------------------------------------ #
    # Nodes                                                                #
    # ------------------------------------------------------------------ #

    graph.add_node(
        "orchestrator",
        functools.partial(orchestrator_node, model=orch_model),
    )
    graph.add_node(
        "domain_agent",
        functools.partial(
            _domain_agent_node,
            pipeline=pipeline,
            generative_model=generative_model,
        ),
    )
    graph.add_node("web_research", web_research_node)
    graph.add_node(
        "contract",
        ContractAgent(generative_model=generative_model),
    )
    graph.add_node(
        "qa",
        QAAgent(model=generative_model),
    )

    # ------------------------------------------------------------------ #
    # Edges                                                                #
    # ------------------------------------------------------------------ #

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "domain_agent": "domain_agent",
            "web_research": "web_research",
            "contract": "contract",
        },
    )

    # Domain agent always goes to QA
    graph.add_edge("domain_agent", "qa")

    # QA: pass → END, fail → retry domain_agent
    graph.add_conditional_edges(
        "qa",
        route_after_qa,
        {
            "end": END,
            "domain_agent": "domain_agent",
        },
    )

    # Web research and contract skip QA → straight to END
    graph.add_edge("web_research", END)
    graph.add_edge("contract", END)

    return graph.compile()
