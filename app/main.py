import logging
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Local Imports (Updated to include the agent) ---
from src.retriever.retriever_rag import HybridRAGPipeline
from src.ingestion.ingestion_src import ingest_document_from_base64
from src.agent import is_context_relevant, search_perplexity

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Legal RAG API",
    description="An API for retrieving information from legal documents and ingesting new ones.",
    version="1.0.0",
)

# --- Initialize the RAG Pipeline ---
# This is created once when the app starts up.
try:
    rag_pipeline = HybridRAGPipeline()
except Exception as e:
    logger.error(f"Fatal error during RAG Pipeline initialization: {e}")
    rag_pipeline = None


# --- Pydantic Models (No changes here) ---


class RetrieveRequest(BaseModel):
    """Request model for the /retrieve endpoint."""

    user_query: str = Field(..., min_length=3, description="The user's question.")
    search_type: Optional[str] = Field(
        "hybrid",
        pattern="^(semantic|keyword|hybrid)$",
        description="Search type for local search: 'semantic', 'keyword', or 'hybrid'.",
    )
    k: Optional[int] = Field(
        5, gt=0, le=20, description="Number of documents to retrieve for local search."
    )
    # --- ADDED: Optional filter for scoped search ---
    filename_filter: Optional[str] = Field(
        None, min_length=1, description="Optional: Filename to scope the search to."
    )


class RetrieveResponse(BaseModel):
    """Response model for the /retrieve endpoint."""

    query: str
    summary: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class IngestRequest(BaseModel):
    """Request model for the /ingest endpoint."""

    base64_text: str = Field(..., description="Base64 encoded content of the file.")
    file_type: str = Field(..., pattern=r"^\.pdf$", description="File extension, e.g., '.pdf'.")
    filename: str = Field(..., min_length=1, description="Original name of the file.")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Optional additional metadata.")


class IngestResponse(BaseModel):
    """Response model for the /ingest endpoint."""

    status: str
    message: str
    filename: str
    chunks_added: int


class RefreshResponse(BaseModel):
    status: str
    documents_indexed: int


# --- API Endpoints ---


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """
    Receives a query, first attempts to find a relevant local answer,
    and falls back to a web search if the local context is insufficient.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not available.")

    try:
        # Step 1: Always perform a local search first to get documents and scores.
        logger.info(f"Step 1: Performing local search for query: '{request.user_query}'")
        local_results = rag_pipeline.semantic_search_with_scores(request.user_query, k=request.k)

        # Extract just the documents to pass to the relevance checker.
        documents_for_check = [doc for doc, score in local_results]

        # Step 2: Use the agent to check if the retrieved context is relevant.
        logger.info("Step 2: Checking relevance of local results.")
        if is_context_relevant(
            request.user_query, documents_for_check, rag_pipeline.generative_model
        ):
            # Step 3a: If relevant, process the query using the full RAG pipeline and return.
            logger.info("Local context is RELEVANT. Generating summary from local documents.")
            return rag_pipeline.process_query(
                query=request.user_query, k=request.k, search_type=request.search_type
            )
        else:
            # Step 3b: If not relevant, fall back to a web search.
            logger.info("Local context is NOT RELEVANT or empty. Falling back to web search.")
            return await search_perplexity(request.user_query)

    except Exception as e:
        logger.error(f"Error processing retrieve request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process the query.")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Receives a file and ingests its content into the vector store
    by calling the dedicated ingestion function.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not available.")

    try:
        chunks_added = ingest_document_from_base64(
            vectorstore=rag_pipeline.vectorstore,
            base64_text=request.base64_text,
            filename=request.filename,
            file_type=request.file_type,
            metadata=request.metadata,
        )

        return IngestResponse(
            status="success",
            message="File ingested successfully.",
            filename=request.filename,
            chunks_added=chunks_added,
        )
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid request for ingestion: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail="Failed to ingest the file.")


@app.post("/refresh-index", response_model=RefreshResponse)
async def refresh_index():
    """
    Manually triggers a refresh of the in-memory TF-IDF keyword index.
    Should be called after a successful document ingestion.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not available.")
    try:
        rag_pipeline.refresh_tfidf_corpus()
        return RefreshResponse(status="success", documents_indexed=len(rag_pipeline.documents))
    except Exception as e:
        logger.error(f"Failed to refresh TF-IDF index: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh index.")


if __name__ == "__main__":
    # To run this app, navigate to your terminal in this directory and execute:
    # uvicorn main:app --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
