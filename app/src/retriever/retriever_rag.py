import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# --- LangChain and Google Imports ---
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import google.generativeai as genai

# --- Standard Library Imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRAGPipeline:
    def __init__(self, chroma_dir: str = "chroma_db"):
        """
        Initialize the RAG pipeline with Google AI and hybrid search capabilities.
        """
        self.chroma_dir = chroma_dir

        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            logger.info("Google GenAI client configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Google GenAI: {e}")
            raise

        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.generative_model = genai.GenerativeModel("gemma-3-27b-it")
        logger.info("Initialized Google embedding and generative models.")

        self.vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=self.embedding_model,
            collection_name="legal_documents",
        )

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000, stop_words="english", ngram_range=(1, 2)
        )
        # Prepare the corpus once on startup
        self.refresh_tfidf_corpus()

    def refresh_tfidf_corpus(self):
        """
        Refreshes the in-memory TF-IDF corpus from all documents in ChromaDB.
        This should be called after new documents are ingested.
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
            # Use a 'where' clause for efficient, database-level filtering
            where_filter = {"filename": filename_filter} if filename_filter else None
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query, k=k, where=where_filter
            )
            logger.info(f"Semantic search with scores returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error in semantic search with scores: {e}")
            return []

    # --- RESTORED: This method is required by hybrid_search ---
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

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.7,
        filename_filter: Optional[str] = None,
    ) -> List[Document]:
        """Perform hybrid search, combining semantic and keyword results."""
        # Note: Filtering is applied at the semantic level and post-retrieval for keyword
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
            """Create a unique key for a document."""
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

    def process_query(
        self,
        query: str,
        k: int = 5,
        search_type: str = "hybrid",
        filename_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a query and return summarized results, optionally filtering by filename.
        """
        logger.info(
            f"Processing query: '{query}' with {search_type} search. Filter: {filename_filter}"
        )

        results = []
        if search_type == "semantic":
            results = self.semantic_search(query, k=k, filename_filter=filename_filter)
        else:  # 'hybrid' or 'keyword'
            results = self.hybrid_search(query, k=k, filename_filter=filename_filter)

        # If hybrid search didn't yield enough results, we can just use semantic as a fallback
        if not results and filename_filter:
            results = self.semantic_search(query, k=k, filename_filter=filename_filter)

        if not results:
            summary_message = "No relevant information found."
            if filename_filter:
                summary_message = (
                    f"No relevant information found in the document '{filename_filter}'."
                )
            logger.warning("No results found for query after filtering.")
            return {
                "query": query,
                "summary": summary_message,
                "results": [],
                "metadata": {"source_files": [filename_filter or "All Documents"]},
            }

        combined_content = "\n\n".join([doc.page_content for doc in results])
        summary = self.summarize_with_gemma(combined_content)

        categories = {
            doc.metadata.get("category") for doc in results if doc.metadata.get("category")
        }
        filenames = {
            doc.metadata.get("filename") for doc in results if doc.metadata.get("filename")
        }

        metadata = {
            "num_results": len(results),
            "categories": list(categories),
            "source_files": list(filenames),
            "search_type": search_type,
            "timestamp": datetime.now().isoformat(),
        }

        return {
            "query": query,
            "summary": summary,
            "results": [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in results
            ],
            "metadata": metadata,
        }

    def summarize_with_gemma(self, context: str) -> str:
        """
        Summarize the retrieved context using the Gemma model with a system prompt.
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
