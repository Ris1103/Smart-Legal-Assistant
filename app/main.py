import logging
from typing import List, Dict, Any, Optional
import tempfile
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow

# --- Local Imports (Updated to include the agent) ---
from src.retriever.retriever_rag import HybridRAGPipeline
from src.ingestion.ingestion_src import ingest_document_from_base64
from src.agent import is_context_relevant, search_perplexity
from src.evaluation.evaluation import calculate_faithfulness

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
mlflow.set_experiment("Legal_RAG_Assistant")

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
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is not available.")

    run_name = f"query_{request.user_query[:50].replace(' ', '_')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("user_query", request.user_query)
        run_id = run.info.run_id

        local_results_with_scores = rag_pipeline.semantic_search_with_scores(
            request.user_query, k=request.k, filename_filter=request.filename_filter
        )
        documents_for_check = [doc for doc, score in local_results_with_scores]

        if is_context_relevant(
            request.user_query, documents_for_check, rag_pipeline.generative_model
        ):
            mlflow.log_param("tool_used", "local_search")
            # This call will be traced because it's inside an active run
            results = rag_pipeline.process_query(
                query=request.user_query,
                k=request.k,
                search_type=request.search_type,
                filename_filter=request.filename_filter,
            )
            tool_used = "local_search"
        else:
            mlflow.log_param("tool_used", "web_search")
            # This call will be traced
            results = await search_perplexity(request.user_query)
            tool_used = "web_search"

        summary = results["summary"]
        retrieved_docs_list = results["results"]

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "retrieved_context.txt")
            summary_path = os.path.join(tmpdir, "generated_summary.txt")

            context_str = "\n\n---\n\n".join([doc["content"] for doc in retrieved_docs_list])
            with open(context_path, "w", encoding="utf-8") as f:
                f.write(context_str)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)

            mlflow.log_artifacts(tmpdir)

        if tool_used == "local_search" and retrieved_docs_list:
            faithfulness = calculate_faithfulness(
                query=request.user_query,
                retrieved_docs=retrieved_docs_list,
                summary=summary,
                model=rag_pipeline.generative_model,
            )
            mlflow.log_metric("faithfulness_score", faithfulness)

        results["mlflow_run_id"] = run_id
        return results


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
