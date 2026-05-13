import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import uvicorn
import mlflow
from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    Depends,
    Security,
)
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from config.settings import settings
from src.retriever.retriever_rag import HybridRAGPipeline
from src.ingestion.ingestion_src import ingest_document_from_base64
from src.agent import is_context_relevant, search_perplexity
from src.evaluation.evaluation import (
    calculate_faithfulness,
    FAITHFULNESS_ERROR_SENTINEL,
)
from api.routes.query import get_query_router
from api.routes.contracts import get_contracts_router

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
mlflow.set_experiment("Legal_RAG_Assistant")

# --- MCP Client Manager (optional) ---
_mcp_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_manager
    if settings.mcp_enabled:
        from mcp_client.client import MCPClientManager
        _mcp_manager = MCPClientManager({
            "search": settings.mcp_search_server_url,
            "filesystem": settings.mcp_filesystem_server_url,
            "database": settings.mcp_database_server_url,
        })
        await _mcp_manager.start()
        logger.info("MCP client manager started.")
    yield
    if _mcp_manager:
        await _mcp_manager.stop()
        logger.info("MCP client manager stopped.")


# --- FastAPI App ---
app = FastAPI(
    title="Legal RAG API",
    description=(
        "API for retrieving information from legal documents "
        "and ingesting new ones."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# --- API Key Auth ---
# When settings.service_api_key is empty, auth is disabled (dev mode).
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    key: Optional[str] = Security(_api_key_header),
):
    if not settings.service_api_key:
        return  # auth disabled in dev
    if key != settings.service_api_key:
        raise HTTPException(
            status_code=403, detail="Invalid or missing API key."
        )


# --- RAG Pipeline (singleton) ---
try:
    rag_pipeline = HybridRAGPipeline()
except Exception as e:
    logger.error(f"Fatal error during RAG Pipeline initialization: {e}")
    rag_pipeline = None

# --- Phase 2 Multi-Agent Routes ---
# _mcp_manager is None at import time; routers read it via a closure so they
# always see the live value set during the lifespan startup.
def _get_mcp():
    return _mcp_manager

app.include_router(get_query_router(rag_pipeline, verify_api_key, _get_mcp))
app.include_router(get_contracts_router(rag_pipeline, verify_api_key, _get_mcp))


# --- Pydantic Models ---

class RetrieveRequest(BaseModel):
    user_query: str = Field(..., min_length=3)
    search_type: Optional[str] = Field(
        "hybrid", pattern="^(semantic|keyword|hybrid)$"
    )
    k: Optional[int] = Field(5, gt=0, le=20)
    filename_filter: Optional[str] = Field(None, min_length=1)
    page: Optional[int] = Field(1, ge=1)
    page_size: Optional[int] = Field(5, ge=1, le=20)
    collection: Optional[str] = Field(None)


class RetrieveResponse(BaseModel):
    query: str
    summary: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    mlflow_run_id: Optional[str] = None


class IngestRequest(BaseModel):
    base64_text: str = Field(...)
    file_type: str = Field(..., pattern=r"^\.pdf$")
    filename: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = Field({})
    collection: Optional[str] = Field(None)


class IngestResponse(BaseModel):
    status: str
    message: str
    filename: str
    chunks_added: int


class RefreshResponse(BaseModel):
    status: str
    documents_indexed: int


# --- Endpoints ---

@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    dependencies=[Depends(verify_api_key)],
)
async def retrieve(request: RetrieveRequest):
    if not rag_pipeline:
        raise HTTPException(
            status_code=503, detail="RAG Pipeline unavailable."
        )

    run_name = f"query_{request.user_query[:50].replace(' ', '_')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("user_query", request.user_query)
        run_id = run.info.run_id

        local_results_with_scores = (
            rag_pipeline.semantic_search_with_scores(
                request.user_query,
                k=request.k,
                filename_filter=request.filename_filter,
            )
        )
        documents_for_check = [
            doc for doc, _ in local_results_with_scores
        ]

        if is_context_relevant(
            request.user_query,
            documents_for_check,
            rag_pipeline.generative_model,
        ):
            mlflow.log_param("tool_used", "local_search")
            results = rag_pipeline.process_query(
                query=request.user_query,
                k=request.k,
                search_type=request.search_type,
                filename_filter=request.filename_filter,
                page=request.page,
                page_size=request.page_size,
            )
            tool_used = "local_search"
        else:
            mlflow.log_param("tool_used", "web_search")
            results = await search_perplexity(request.user_query)
            tool_used = "web_search"

        summary = results["summary"]
        retrieved_docs_list = results["results"]

        # Log context and summary as MLflow artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx_path = os.path.join(tmpdir, "retrieved_context.txt")
            sum_path = os.path.join(tmpdir, "generated_summary.txt")
            context_str = "\n\n---\n\n".join(
                [doc["content"] for doc in retrieved_docs_list]
            )
            with open(ctx_path, "w", encoding="utf-8") as f:
                f.write(context_str)
            with open(sum_path, "w", encoding="utf-8") as f:
                f.write(summary)
            mlflow.log_artifacts(tmpdir)

        # Faithfulness — only for local search
        if tool_used == "local_search" and retrieved_docs_list:
            faithfulness = calculate_faithfulness(
                query=request.user_query,
                retrieved_docs=retrieved_docs_list,
                summary=summary,
                model=rag_pipeline.generative_model,
            )
            if faithfulness == FAITHFULNESS_ERROR_SENTINEL:
                mlflow.log_metric("faithfulness_eval_error", 1)
            else:
                mlflow.log_metric("faithfulness_score", faithfulness)

        results["mlflow_run_id"] = run_id
        return results


@app.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(verify_api_key)],
)
async def ingest(
    request: IngestRequest, background_tasks: BackgroundTasks
):
    """
    Ingest a PDF into the vector store.
    TF-IDF index is refreshed automatically in the background so
    subsequent keyword searches reflect the new document immediately.
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=503, detail="RAG Pipeline unavailable."
        )

    try:
        chunks_added = ingest_document_from_base64(
            vectorstore=rag_pipeline.vectorstore,
            base64_text=request.base64_text,
            filename=request.filename,
            file_type=request.file_type,
            metadata=request.metadata,
        )

        if chunks_added == 0:
            return IngestResponse(
                status="duplicate",
                message=(
                    "Document already exists in the knowledge base."
                ),
                filename=request.filename,
                chunks_added=0,
            )

        # Refresh TF-IDF index in the background
        background_tasks.add_task(rag_pipeline.refresh_tfidf_corpus)

        return IngestResponse(
            status="success",
            message="File ingested successfully.",
            filename=request.filename,
            chunks_added=chunks_added,
        )
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to ingest the file."
        )


@app.post(
    "/refresh-index",
    response_model=RefreshResponse,
    dependencies=[Depends(verify_api_key)],
)
async def refresh_index():
    """Manually trigger a TF-IDF refresh (e.g. after bulk ingestion)."""
    if not rag_pipeline:
        raise HTTPException(
            status_code=503, detail="RAG Pipeline unavailable."
        )
    try:
        rag_pipeline.refresh_tfidf_corpus()
        return RefreshResponse(
            status="success",
            documents_indexed=len(rag_pipeline.documents),
        )
    except Exception as e:
        logger.error(f"Failed to refresh TF-IDF index: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to refresh index."
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
