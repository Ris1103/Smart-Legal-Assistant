import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from langchain_core.documents import Document
import google.generativeai as genai

from src.retriever.embedder_factory import get_embedder
from src.vectorstore.base import BaseVectorStore

from rank_bm25 import BM25Okapi
import numpy as np
import logging
import mlflow

from config.settings import settings

load_dotenv()

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class HybridRAGPipeline:
    def __init__(self, vectorstore: Optional[BaseVectorStore] = None):
        try:
            genai.configure(api_key=settings.google_api_key)
        except Exception as e:
            logger.error(f"Failed to configure Google GenAI: {e}")
            raise

        self.embedding_model = get_embedder(settings)
        self.generative_model = genai.GenerativeModel(settings.generative_model_name)
        logger.info(
            f"Initialized embedding ({settings.embedding_provider}) and "
            f"generative model ({settings.generative_model_name})."
        )

        self._reranker = None

        if vectorstore is None:
            from src.vectorstore.factory import VectorStoreFactory
            vectorstore = VectorStoreFactory.get_instance(
                embedding_model=self.embedding_model
            )
        self.vectorstore = vectorstore
        logger.info(f"Using vector store: {type(vectorstore).__name__} ('{vectorstore.name}')")

        self.bm25: Optional[BM25Okapi] = None
        self.bm25_docs: List[Document] = []
        self.refresh_bm25_corpus()

    # ------------------------------------------------------------------
    # Corpus refresh
    # ------------------------------------------------------------------

    def refresh_bm25_corpus(self):
        logger.info("Refreshing BM25 corpus from vector store...")
        try:
            all_docs = self.vectorstore.get_all()
            texts = all_docs.get("documents", [])
            metadatas = all_docs.get("metadatas", [])
            if texts:
                tokenized = [_tokenize(t) for t in texts]
                self.bm25 = BM25Okapi(tokenized)
                self.bm25_docs = [
                    Document(page_content=t, metadata=m)
                    for t, m in zip(texts, metadatas)
                ]
                logger.info(f"BM25 corpus refreshed with {len(texts)} documents.")
            else:
                logger.warning("No documents found in vector store during refresh.")
                self.bm25, self.bm25_docs = None, []
        except Exception as e:
            logger.error(f"Error refreshing BM25 corpus: {e}")
            self.bm25, self.bm25_docs = None, []

    # Keep old name as alias so background tasks using the old name still work
    def refresh_tfidf_corpus(self):
        self.refresh_bm25_corpus()

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def semantic_search_with_scores(
        self, query: str, k: int = 5, filename_filter: Optional[str] = None
    ) -> List[tuple]:
        try:
            where_filter = {"filename": filename_filter} if filename_filter else None
            results = self.vectorstore.similarity_search_with_scores(
                query, k=k, filter=where_filter
            )
            logger.info(f"Semantic search with scores returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error in semantic search with scores: {e}")
            return []

    def semantic_search(
        self, query: str, k: int = 5, filename_filter: Optional[str] = None
    ) -> List[Document]:
        try:
            where_filter = {"filename": filename_filter} if filename_filter else None
            results = self.vectorstore.similarity_search(query, k=k, filter=where_filter)
            logger.info(f"Semantic search returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def keyword_search(
        self, query: str, k: int = 5, filename_filter: Optional[str] = None
    ) -> List[Document]:
        """BM25 keyword search. Returns ranked Document list."""
        if not self.bm25 or not self.bm25_docs:
            return []
        try:
            scores = self.bm25.get_scores(_tokenize(query))
            top_indices = np.argsort(scores)[::-1][:k * 3]  # over-fetch before filter
            results = []
            for idx in top_indices:
                if scores[idx] <= 0:
                    break
                doc = self.bm25_docs[idx]
                if filename_filter and doc.metadata.get("filename") != filename_filter:
                    continue
                results.append(doc)
                if len(results) == k:
                    break
            return results
        except Exception as e:
            logger.error(f"Error in BM25 keyword search: {e}")
            return []

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _rrf_merge(
        self,
        semantic_results: List[Document],
        keyword_results: List[Document],
        k: int = 60,
    ) -> List[Document]:
        """Reciprocal Rank Fusion over two ranked lists."""
        scores: Dict[str, float] = {}
        docs_map: Dict[str, Document] = {}

        def doc_key(doc: Document) -> str:
            cid = doc.metadata.get("chunk_id")
            fname = doc.metadata.get("filename", "")
            if cid is not None:
                return f"{fname}::{cid}"
            return doc.page_content[:120]

        for rank, doc in enumerate(semantic_results, start=1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs_map[key] = doc

        for rank, doc in enumerate(keyword_results, start=1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs_map[key] = doc

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [docs_map[key] for key in sorted_keys]

    # ------------------------------------------------------------------
    # Reranker (BGE)
    # ------------------------------------------------------------------

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs
        if self._reranker is None:
            try:
                from FlagEmbedding import FlagReranker
                self._reranker = FlagReranker(
                    settings.reranker_model, use_fp16=True
                )
                logger.info(f"Loaded BGE reranker: {settings.reranker_model}")
            except Exception as e:
                logger.warning(f"Could not load BGE reranker ({e}), falling back to cross-encoder.")
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(settings.reranker_model)

        try:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self._reranker.compute_score(pairs)
            if not isinstance(scores, list):
                scores = list(scores)
            return [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return docs

    # ------------------------------------------------------------------
    # Hybrid search (semantic + BM25 → RRF)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        filename_filter: Optional[str] = None,
        **_kwargs,  # absorb legacy semantic_weight kwarg
    ) -> List[Document]:
        semantic_results = self.semantic_search(query, k=k, filename_filter=filename_filter)
        keyword_results = self.keyword_search(query, k=k, filename_filter=filename_filter)
        merged = self._rrf_merge(semantic_results, keyword_results)
        return merged[:k]

    # ------------------------------------------------------------------
    # Main process_query
    # ------------------------------------------------------------------

    @mlflow.trace(name="local_rag_pipeline")
    def process_query(
        self,
        query: str,
        k: int = 5,
        search_type: str = "hybrid",
        filename_filter: Optional[str] = None,
        page: int = 1,
        page_size: int = 5,
        response_style: str = "detailed",
    ) -> Dict[str, Any]:
        logger.info(
            f"Processing query: '{query}' with {search_type} search. "
            f"Filter: {filename_filter}, page={page}, page_size={page_size}"
        )

        results: List[Document] = []
        if search_type == "semantic":
            results = self.semantic_search(query, k=k, filename_filter=filename_filter)
        elif search_type == "keyword":
            results = self.keyword_search(query, k=k, filename_filter=filename_filter)
        else:
            results = self.hybrid_search(query, k=k, filename_filter=filename_filter)

        if results and settings.reranker_enabled:
            results = self.rerank(query, results)

        # Build annotated context with source tags for citation
        citations = []
        annotated_chunks = []
        for doc in results:
            fname = doc.metadata.get("filename", "unknown")
            page_num = doc.metadata.get("page", doc.metadata.get("chunk_id", "?"))
            chunk_id = doc.metadata.get("chunk_id", "")
            tag = f"[Source: {fname}, Page {page_num}]"
            annotated_chunks.append(f"{tag}\n{doc.page_content}")
            citations.append({
                "filename": fname,
                "page": page_num,
                "chunk_id": chunk_id,
                "excerpt": doc.page_content[:120],
            })

        if not results:
            summary = "No relevant information found in the local documents."
        else:
            combined_content = "\n\n".join(annotated_chunks)
            summary = self.summarize_with_gemma(combined_content, response_style)

        filenames = {doc.metadata.get("filename") for doc in results if doc.metadata.get("filename")}

        total_results = len(results)
        start = (page - 1) * page_size
        paginated_results = results[start: start + page_size]

        return {
            "query": query,
            "summary": summary,
            "results": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in paginated_results
            ],
            "citations": citations,
            "metadata": {
                "num_results": total_results,
                "page": page,
                "page_size": page_size,
                "source_files": list(filenames),
                "search_type": search_type,
                "timestamp": datetime.now().isoformat(),
            },
        }

    # ------------------------------------------------------------------
    # Summarizer
    # ------------------------------------------------------------------

    @mlflow.trace(name="gemma_summarizer")
    def summarize_with_gemma(self, context: str, response_style: str = "detailed") -> str:
        logger.info(f"Generating {response_style} response with Gemma...")
        if response_style == "brief":
            style_instruction = (
                "Provide a concise 2-3 sentence answer to the question based on the context."
            )
        else:
            style_instruction = (
                "Provide a comprehensive, well-structured answer that includes:\n"
                "1. A direct answer to the question\n"
                "2. Relevant statutory provisions (cite section numbers and act names)\n"
                "3. Practical implications for an Indian SMB\n"
                "4. Any important deadlines, penalties, or compliance requirements\n"
                "When citing information, reference the source tag (e.g. [Source: filename.pdf]) "
                "provided in the context. Structure your response with clear paragraphs."
            )

        system_prompt = (
            "You are an expert legal assistant specialising in Indian law for SMBs. "
            f"{style_instruction}\n"
            "Always end with: 'Disclaimer: This information is for educational purposes only "
            "and does not constitute formal legal advice. Please consult a qualified legal "
            "professional for advice specific to your situation.'"
        )
        prompt = f"{system_prompt}\n\n---CONTEXT---\n{context}\n\n---QUESTION---\n---ANSWER---"
        try:
            response = self.generative_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error during Gemma summarization: {e}")
            return "Could not generate a response due to an error."
