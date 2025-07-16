import os
import json
from datetime import datetime
from typing import List, Dict, Any
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

        Args:
            chroma_dir: Directory containing ChromaDB.
        """
        self.chroma_dir = chroma_dir

        # --- 1. Configure Google AI ---
        # The API key is loaded from the .env file
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            logger.info("Google GenAI client configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Google GenAI: {e}")
            raise

        # --- 2. Initialize Google Models ---
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.generative_model = genai.GenerativeModel("gemma-3-27b-it")
        logger.info("Initialized Google embedding and generative models.")

        # Initialize ChromaDB vectorstore with the Google embedding model
        self.vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=self.embedding_model,
            collection_name="legal_documents",
        )

        # Initialize TF-IDF vectorizer for keyword search
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000, stop_words="english", ngram_range=(1, 2)
        )

        # Prepare TF-IDF corpus from all documents in the vectorstore
        self._prepare_tfidf_corpus()

    def _prepare_tfidf_corpus(self):
        """Prepare TF-IDF corpus from all documents in ChromaDB."""
        try:
            all_docs = self.vectorstore.get(include=["metadatas", "documents"])
            self.documents = all_docs.get("documents", [])
            self.metadatas = all_docs.get("metadatas", [])

            if self.documents:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
                logger.info(f"TF-IDF corpus prepared with {len(self.documents)} documents.")
            else:
                logger.warning("No documents found in ChromaDB.")
                self.documents, self.metadatas, self.tfidf_matrix = [], [], None
        except Exception as e:
            logger.error(f"Error preparing TF-IDF corpus: {e}")
            self.documents, self.metadatas, self.tfidf_matrix = [], [], None

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform semantic search using ChromaDB."""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
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

            results = [
                {
                    "content": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "score": similarities[idx],
                }
                for idx in top_indices
                if similarities[idx] > 0
            ]
            logger.info(f"Keyword search returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    def hybrid_search(
        self, query: str, k: int = 10, semantic_weight: float = 0.7
    ) -> List[Document]:
        """Perform hybrid search combining semantic and keyword results."""
        semantic_results = self.semantic_search(query, k=k)
        keyword_results = self.keyword_search(query, k=k)

        combined_results = {}

        # Use a more robust key based on metadata if available
        def doc_key(doc):
            return doc.metadata.get("source", doc.page_content[:100])

        # Add semantic results with rank-based scoring
        for i, doc in enumerate(semantic_results):
            score = (k - i) / k * semantic_weight
            combined_results[doc_key(doc)] = {"document": doc, "score": score}

        # Add or update with keyword results
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
        final_results = [result["document"] for result in sorted_results[:k]]
        logger.info(f"Hybrid search returned {len(final_results)} results.")
        return final_results

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

    def process_query(self, query: str, k: int = 5, search_type: str = "hybrid") -> Dict[str, Any]:
        """Process a query and return summarized results."""
        logger.info(f"Processing query: '{query}' with {search_type} search.")
        if search_type == "semantic":
            results = self.semantic_search(query, k)
        elif search_type == "keyword":
            keyword_results = self.keyword_search(query, k)
            results = [
                Document(page_content=r["content"], metadata=r["metadata"])
                for r in keyword_results
            ]
        else:  # 'hybrid'
            results = self.hybrid_search(query, k)

        if not results:
            logger.warning("No results found for query.")
            return {
                "query": query,
                "summary": "No relevant information found in the documents.",
                "results": [],
                "metadata": {},
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

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results to a text file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_results_{timestamp}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Query: {results['query']}\n")
                f.write(f"Timestamp: {results['metadata'].get('timestamp')}\n\n")
                f.write("=" * 20 + " SUMMARY " + "=" * 20 + "\n")
                f.write(results["summary"] + "\n\n")
                f.write("=" * 20 + " SOURCES " + "=" * 20 + "\n")
                f.write(
                    f"Source Files: {', '.join(results['metadata'].get('source_files', []))}\n\n"
                )
                f.write("=" * 20 + " DETAILS " + "=" * 20 + "\n")
                for i, result in enumerate(results["results"], 1):
                    f.write(f"\n--- Result {i} ---\n")
                    f.write(f"Source: {result['metadata'].get('filename', 'N/A')}\n")
                    f.write(f"Content: {result['content'][:500]}...\n")
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Example usage of the Google RAG pipeline."""
    rag = HybridRAGPipeline()

    queries = [
        "What are the GST compliance requirements for small businesses?",
        "How to register a company under Companies Act?",
        "Penalties under Income Tax Act for non-compliance",
    ]

    for query in queries:
        print(f"\n{'=' * 20}\nProcessing query: {query}\n{'=' * 20}")
        results = rag.process_query(query, k=5, search_type="hybrid")

        sanitized_query = "".join(c for c in query if c.isalnum() or c in " _-").rstrip()
        filename = f"rag_results_{sanitized_query[:30]}.txt"
        rag.save_results(results, filename)

        print(f"Summary: {results['summary'][:200]}...")


if __name__ == "__main__":
    main()
