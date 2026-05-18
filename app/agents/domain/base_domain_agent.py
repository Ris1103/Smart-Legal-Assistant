"""
Base domain agent — all domain specialists inherit from this class.
Subclasses only need to override get_collection_name() and get_system_prompt().
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import google.generativeai as genai

from config.settings import settings
from graph.state import AgentState
from src.retriever.retriever_rag import HybridRAGPipeline
from src.retriever.query_expander import expand_query

logger = logging.getLogger(__name__)

_QA_FEEDBACK_PREFIX = "[QA Feedback] "


class BaseDomainAgent(ABC):

    def __init__(
        self,
        pipeline: HybridRAGPipeline,
        generative_model: genai.GenerativeModel,
    ):
        self.pipeline = pipeline
        self.model = generative_model

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
        response_style: str = state.get("response_style", "detailed")

        effective_query = query
        if qa_feedback:
            effective_query = f"{query}\n\n{_QA_FEEDBACK_PREFIX}{qa_feedback}"
            logger.info(f"{self.__class__.__name__} retrying with QA feedback.")

        logger.info(f"{self.__class__.__name__} processing: '{query[:60]}'")

        # Multi-query expansion
        queries = expand_query(effective_query, self.model)

        # Retrieve for each query variant, deduplicate by chunk_id
        k_per_query = max(4, 8 // len(queries))
        seen_keys = set()
        all_docs = []

        for q in queries:
            try:
                result = self.pipeline.hybrid_search(query=q, k=k_per_query)
                for doc in result:
                    cid = doc.metadata.get("chunk_id")
                    fname = doc.metadata.get("filename", "")
                    key = f"{fname}::{cid}" if cid is not None else doc.page_content[:100]
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_docs.append(doc)
            except Exception as e:
                logger.warning(f"Retrieval failed for query variant '{q[:40]}': {e}")

        if not all_docs:
            return {
                **state,
                "summary": "No relevant information found in the local knowledge base.",
                "retrieved_docs": [],
                "citations": [],
                "source_files": [],
                "search_type": "local",
            }

        # Rerank all deduplicated candidates
        if settings.reranker_enabled:
            all_docs = self.pipeline.rerank(effective_query, all_docs)
        all_docs = all_docs[:8]

        # Generate response with inline citations
        summary = self._generate_response(query, all_docs, qa_feedback, response_style)

        source_files = list({
            doc.metadata.get("filename", "")
            for doc in all_docs
            if doc.metadata.get("filename")
        })

        citations = [
            {
                "filename": doc.metadata.get("filename", "unknown"),
                "page": doc.metadata.get("page", doc.metadata.get("chunk_id", "?")),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "excerpt": doc.page_content[:120],
            }
            for doc in all_docs
        ]

        docs_serialized = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in all_docs
        ]

        return {
            **state,
            "retrieved_docs": docs_serialized,
            "summary": summary,
            "source_files": source_files,
            "citations": citations,
            "search_type": "local",
            "metadata": {
                **(state.get("metadata") or {}),
                "domain_agent": self.__class__.__name__,
                "num_results": len(all_docs),
                "query_variants": len(queries),
                "timestamp": datetime.now().isoformat(),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress_docs(self, query: str, docs: list) -> list:
        try:
            from langchain.retrievers.document_compressors import EmbeddingsFilter
            from langchain_core.documents import Document as LC_Document
            lc_docs = [
                LC_Document(page_content=d["content"], metadata=d.get("metadata", {}))
                for d in docs if d.get("content")
            ]
            compressor = EmbeddingsFilter(
                embeddings=self.pipeline.embedding_model,
                similarity_threshold=settings.compression_similarity_threshold,
            )
            compressed = compressor.compress_documents(lc_docs, query)
            return [{"content": d.page_content, "metadata": d.metadata} for d in compressed]
        except Exception as e:
            logger.warning(f"Context compression failed, using original docs: {e}")
            return docs

    def _generate_response(
        self,
        query: str,
        docs: List,
        qa_feedback: Optional[str],
        response_style: str = "detailed",
    ) -> str:
        # docs can be Document objects or dicts
        def get_content(d):
            return d.page_content if hasattr(d, "page_content") else d.get("content", "")
        def get_meta(d):
            return d.metadata if hasattr(d, "metadata") else d.get("metadata", {})

        if settings.context_compression_enabled:
            dict_docs = [{"content": get_content(d), "metadata": get_meta(d)} for d in docs]
            dict_docs = self._compress_docs(query, dict_docs)
            annotated = [
                f"[Source: {d['metadata'].get('filename','?')}, Page {d['metadata'].get('page','?')}]\n{d['content']}"
                for d in dict_docs
            ]
        else:
            annotated = [
                f"[Source: {get_meta(d).get('filename','?')}, Page {get_meta(d).get('page','?')}]\n{get_content(d)}"
                for d in docs
            ]

        context = "\n\n".join(annotated)

        feedback_block = ""
        if qa_feedback:
            feedback_block = (
                f"\n\nIMPORTANT — A quality reviewer flagged the previous response:\n"
                f"{qa_feedback}\nPlease address these issues in your answer."
            )

        if response_style == "brief":
            style_instruction = "Provide a concise 2-3 sentence answer."
        else:
            style_instruction = (
                "Provide a comprehensive, well-structured answer that includes:\n"
                "- A direct answer to the question\n"
                "- Relevant statutory provisions (cite specific section numbers and act names)\n"
                "- Practical implications for an Indian SMB\n"
                "- Important deadlines, penalties, or compliance requirements if applicable\n"
                "When citing information, reference the [Source: filename] tags from the context."
            )

        prompt = (
            f"{self.get_system_prompt()}\n\n"
            f"{style_instruction}"
            f"{feedback_block}"
            "\n\n---CONTEXT---\n"
            f"{context}"
            "\n\n---QUESTION---\n"
            f"{query}"
            "\n\n---ANSWER---"
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Generation failed in {self.__class__.__name__}: {e}")
            return "I encountered an error generating a response. Please try again."
