"""
AgentState — the shared typed state passed through every node in the
LangGraph agent graph. All fields are Optional so nodes can selectively
populate what they need.
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ------------------------------------------------------------------ #
    # Input                                                                #
    # ------------------------------------------------------------------ #
    query: str                        # Original user query
    session_id: Optional[str]         # For future session history support

    # ------------------------------------------------------------------ #
    # Orchestrator output                                                  #
    # ------------------------------------------------------------------ #
    domain: str                       # Classified domain (e.g. "GST")
    confidence: float                 # Classifier confidence 0.0–1.0
    intent: str                       # "query" | "contract"

    # ------------------------------------------------------------------ #
    # Retrieval / generation                                               #
    # ------------------------------------------------------------------ #
    retrieved_docs: List[Dict[str, Any]]  # Docs from domain agent
    summary: str                          # Generated answer / contract
    source_files: List[str]               # Originating filenames
    search_type: str                      # "local" | "web"

    # ------------------------------------------------------------------ #
    # QA                                                                   #
    # ------------------------------------------------------------------ #
    faithfulness_score: float         # 0.0–1.0; -1.0 = eval error
    ragas_scores: Optional[Dict[str, float]]  # Full RAGAS metrics when enabled
    qa_passed: bool                   # True when QA gate cleared
    qa_feedback: str                  # Critique fed back to domain agent
    qa_retries: int                   # Number of QA retries so far

    # ------------------------------------------------------------------ #
    # Contract                                                             #
    # ------------------------------------------------------------------ #
    contract_type: Optional[str]      # e.g. "nda", "service_agreement"
    contract_params: Dict[str, Any]   # Extracted parameters for template
    contract_text: str                # Rendered contract

    # ------------------------------------------------------------------ #
    # MCP                                                                  #
    # ------------------------------------------------------------------ #
    mcp_clients: Optional[Dict[str, Any]]   # injected at invocation time

    # ------------------------------------------------------------------ #
    # Final response                                                       #
    # ------------------------------------------------------------------ #
    response: str                     # Assembled final response
    metadata: Dict[str, Any]          # Timestamps, sources, tool used
    error: Optional[str]              # Set if a node fails hard
