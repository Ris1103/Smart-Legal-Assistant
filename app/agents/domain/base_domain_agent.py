"""
Base domain agent — all domain specialists inherit from this class.
Subclasses only need to override get_collection_name() and
get_system_prompt().
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import google.generativeai as genai

from graph.state import AgentState
from src.retriever.retriever_rag import HybridRAGPipeline

logger = logging.getLogger(__name__)

_QA_FEEDBACK_PREFIX = "[QA Feedback] "


class BaseDomainAgent(ABC):
    """Handles retrieval and response generation for one legal domain."""

    def __init__(
        self,
        pipeline: HybridRAGPipeline,
        generative_model: genai.GenerativeModel,
    ):
        self.pipeline = pipeline
        self.model = generative_model

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_collection_name(self) -> str:
        """ChromaDB collection for this domain."""

    @abstractmethod
    def get_system_prompt(self) -> str:
        """System-level instruction injected before the context."""

    # ------------------------------------------------------------------
    # Node function
    # ------------------------------------------------------------------

    def __call__(self, state: AgentState) -> AgentState:
        query = state["query"]
        qa_feedback: Optional[str] = state.get("qa_feedback")

        # Augment query with QA feedback on retries
        effective_query = query
        if qa_feedback:
            effective_query = (
                f"{query}\n\n{_QA_FEEDBACK_PREFIX}{qa_feedback}"
            )
            logger.info(
                f"{self.__class__.__name__} retrying with QA feedback."
            )

        logger.info(
            f"{self.__class__.__name__} processing: '{query[:60]}'"
        )

        try:
            result = self.pipeline.process_query(
                query=effective_query,
                k=8,
                search_type="hybrid",
            )
        except Exception as e:
            logger.error(
                f"{self.__class__.__name__} retrieval failed: {e}"
            )
            return {
                **state,
                "summary": (
                    "I could not retrieve information for your query. "
                    "Please try again or rephrase your question."
                ),
                "retrieved_docs": [],
                "source_files": [],
                "search_type": "local",
                "error": str(e),
            }

        docs = result.get("results", [])
        raw_summary = result.get("summary", "")

        # Re-generate with domain-specific system prompt
        if docs:
            summary = self._generate_response(query, docs, qa_feedback)
        else:
            summary = raw_summary or (
                "No relevant information found in the local knowledge base."
            )

        source_files = list(
            {
                d.get("metadata", {}).get("filename", "")
                for d in docs
                if d.get("metadata", {}).get("filename")
            }
        )

        return {
            **state,
            "retrieved_docs": docs,
            "summary": summary,
            "source_files": source_files,
            "search_type": "local",
            "metadata": {
                **(state.get("metadata") or {}),
                "domain_agent": self.__class__.__name__,
                "num_results": len(docs),
                "timestamp": datetime.now().isoformat(),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_response(
        self,
        query: str,
        docs: list,
        qa_feedback: Optional[str],
    ) -> str:
        context = "\n\n".join(
            d["content"] for d in docs if d.get("content")
        )
        feedback_block = ""
        if qa_feedback:
            feedback_block = (
                f"\n\nIMPORTANT — A quality reviewer flagged the previous "
                f"response:\n{qa_feedback}\nPlease address these issues."
            )

        prompt = (
            f"{self.get_system_prompt()}"
            f"{feedback_block}"
            "\n\n---CONTEXT---\n"
            f"{context[:12000]}"
            "\n\n---QUESTION---\n"
            f"{query}"
            "\n\n---ANSWER---"
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Generation failed in {self.__class__.__name__}: {e}")
            return (
                "I encountered an error generating a response. "
                "Please try again."
            )
