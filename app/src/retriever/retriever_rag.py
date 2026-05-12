import pathlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# --- LangChain and Google Imports ---
from langchain_chroma import Chroma
from langchain_core.documents import Document
import google.generativeai as genai

from src.retriever.embedder_factory import get_embedder

# --- Standard Library Imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import mlflow

from config.settings import settings

# Load environment variables from .env file (fallback for local dev)
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRAGPipeline:
    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the RAG pipeline with Google AI and hybrid search capabilities.

        Args:
            chroma_dir: Absolute path to ChromaDB persistence directory.
                        Defaults to settings.chroma_db_path.
            collection_name: ChromaDB collection to use.
                             Defaults to settings.chroma_collection_name.
        """
        # Resolve to an absolute path so the DB is always found regardless of
        # the working directory from which the app is started.
        raw_dir = chroma_dir or settings.chroma_db_path
        self.chroma_dir = str(pathlib.Path(raw_dir).resolve())
        self.collection_name = collection_name or settings.chroma_collection_name

        try:
            genai.configure(api_key=settings.google_api_key)
            logger.info("Google GenAI client configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Google GenAI: {e}")
            raise

        self.embedding_model = get_embedder(settings)
        self.generative_model = genai.GenerativeModel(settings.generative_model_name)
        logger.info(
            f"Initialized embedding ({settings.embedding_provider}) and "
            f"generative model ({settings.generative_model_name})."
        )

        # Cross-encoder re-ranker (lazy-loaded on first use)
        self._reranker = None

        logger.info(f"Using ChromaDB at '{self.chroma_dir}', collection '{self.collection_name}'.")
        self.vectorstore = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
        )

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000, stop_words="english", ngram_range=(1, 2)
        )
        # Prepare the corpus once on startup
        self.refresh_tfidf_corpus()

    def refresh_tfidf_corpus(self):
        """
        Refreshes the in-memory TF-IDF corpus from all documents in ChromaDB.
        Call this after new documents are ingested.
        """
        logger.info("Refreshing TF-IDF corpus from ChromaDB...")
        try:
            all_docs = self.vectorstore.get(include=["metadatas", "documents"])
            self.documents = all_docs.get("documents", [])
            self.metadatas = all_docs.get("metadatas", [])
            if self.documents:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
                logger.info(f"TF-IDF corpus refreshed with {len(self.documents)} documents.")
            else:
                logger.warning("No documents found in ChromaDB during refresh.")
                self.documents, self.metadatas, self.tfidf_matrix = [], [], None
        except Exception as e:
            logger.error(f"Error refreshing TF-IDF corpus: {e}")
            self.documents, self.metadatas, self.tfidf_matrix = [], [], None

    def semantic_search_with_scores(
        self, query: str, k: int = 5, filename_filter: Optional[str] = None
    ) -> list[tuple[Document, float]]:
        """
        Perform semantic search with relevance scores, optionally filtering by filename.
        """
        try:
            where_filter = {"filename": filename_filter} if filename_filter else None
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query, k=k, where=where_filter
            )
            logger.info(f"Semantic search with scores returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error in semantic search with scores: {e}")
            return []

    def semantic_search(
        self, query: str, k: int = 5, filename_filter: Optional[str] = None
    ) -> List[Document]:
        """Perform semantic search using ChromaDB, optionally filtering by filename."""
        try:
            where_filter = {"filename": filename_filter} if filename_filter else None
            results = self.vectorstore.similarity_search(query, k=k, where=where_filter)
            logger.info(f"Semantic search returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def keyword_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform keyword search using TF-IDF."""
        if not self.documents or self.tfidf_matrix is None:
            return []
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[::-1][:k]
            return [
                {
                    "content": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "score": similarities[idx],
                }
                for idx in top_indices
                if similarities[idx] > 0
            ]
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Re-rank docs with a cross-encoder model. Lazy-loads on first call."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(settings.reranker_model)
            logger.info(f"Loaded cross-encoder reranker: {settings.reranker_model}")
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._reranker.predict(pairs)
        return [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        semantic_weight: Optional[float] = None,
        filename_filter: Optional[str] = None,
    ) -> List[Document]:
        """Perform hybrid search, combining semantic and keyword results."""
        if semantic_weight is None:
            semantic_weight = settings.semantic_weight
        semantic_results = self.semantic_search(query, k=k, filename_filter=filename_filter)
        keyword_results = self.keyword_search(query, k=k)

        # Post-filter keyword results if a filter is applied
        if filename_filter:
            keyword_results = [
                res
                for res in keyword_results
                if res["metadata"].get("filename") == filename_filter
            ]

        combined_results = {}

        def doc_key(doc: Document) -> str:
            return doc.metadata.get("source", doc.page_content[:100])

        for i, doc in enumerate(semantic_results):
            score = (k - i) / k * semantic_weight
            combined_results[doc_key(doc)] = {"document": doc, "score": score}

        keyword_weight = 1 - semantic_weight
        for result in keyword_results:
            doc = Document(page_content=result["content"], metadata=result["metadata"])
            key = doc_key(doc)
            score = result["score"] * keyword_weight
            if key in combined_results:
                combined_results[key]["score"] += score
            else:
                combined_results[key] = {"document": doc, "score": score}

        sorted_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)
        return [result["document"] for result in sorted_results[:k]]

    @mlflow.trace(name="local_rag_pipeline")
    def process_query(
        self,
        query: str,
        k: int = 5,
        search_type: str = "hybrid",
        filename_filter: Optional[str] = None,
        page: int = 1,
        page_size: int = 5,
    ) -> Dict[str, Any]:
        """
        Process a query and return summarized results with pagination support.
        """
        logger.info(
            f"Processing query: '{query}' with {search_type} search. "
            f"Filter: {filename_filter}, page={page}, page_size={page_size}"
        )

        results = []
        if search_type == "semantic":
            results = self.semantic_search(query, k=k, filename_filter=filename_filter)
        elif search_type == "keyword":
            keyword_results = self.keyword_search(query, k=k)
            if filename_filter:
                keyword_results = [
                    res
                    for res in keyword_results
                    if res["metadata"].get("filename") == filename_filter
                ]
            results = [
                Document(page_content=r["content"], metadata=r["metadata"])
                for r in keyword_results
            ]
        else:  # 'hybrid'
            results = self.hybrid_search(query, k=k, filename_filter=filename_filter)

        if results and settings.reranker_enabled:
            results = self.rerank(query, results)

        if not results:
            summary = "No relevant information found in the local documents."
        else:
            combined_content = "\n\n".join([doc.page_content for doc in results])
            summary = self.summarize_with_gemma(combined_content)

        filenames = {
            doc.metadata.get("filename") for doc in results if doc.metadata.get("filename")
        }

        # Apply pagination to the results list
        total_results = len(results)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_results = results[start:end]

        return {
            "query": query,
            "summary": summary,
            "results": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in paginated_results
            ],
            "metadata": {
                "num_results": total_results,
                "page": page,
                "page_size": page_size,
                "source_files": list(filenames),
                "search_type": search_type,
                "timestamp": datetime.now().isoformat(),
            },
        }

    @mlflow.trace(name="gemma_summarizer")
    def summarize_with_gemma(self, context: str) -> str:
        """
        Summarize the retrieved context using the Gemma model.
        """
        logger.info("Generating summary with Gemma...")
        system_prompt = (
            "You are an expert legal assistant. Your task is to provide a clear, "
            "concise, and accurate summary of the provided legal text. Focus on the key "
            "points, regulations, and conclusions. Do not add any information that is "
            "not present in the text. Present the summary in a professional format."
        )
        prompt = f"{system_prompt}\n\n---CONTEXT---\n{context}\n\n---SUMMARY---"
        try:
            response = self.generative_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error during Gemma summarization: {e}")
            return "Could not generate a summary due to an error."
