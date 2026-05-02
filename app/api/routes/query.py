"""
POST /query — multi-agent query endpoint (replaces /retrieve).
Runs the full LangGraph agent graph.
"""
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import mlflow
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from graph.graph_builder import build_graph
from src.evaluation.evaluation import FAITHFULNESS_ERROR_SENTINEL

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------ #
# Pydantic models                                                      #
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    user_query: str = Field(..., min_length=3)
    session_id: Optional[str] = Field(None)
    collection: Optional[str] = Field(None)


class QueryResponse(BaseModel):
    query: str
    domain: Optional[str] = None
    confidence: Optional[float] = None
    summary: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    mlflow_run_id: Optional[str] = None


# ------------------------------------------------------------------ #
# Endpoint                                                             #
# ------------------------------------------------------------------ #

def get_query_router(rag_pipeline, verify_api_key):
    """
    Factory that creates the router with injected dependencies.
    Called once from main.py after the pipeline is initialised.
    """

    @router.post(
        "/query",
        response_model=QueryResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def query(request: QueryRequest):
        if not rag_pipeline:
            raise HTTPException(
                status_code=503,
                detail="RAG Pipeline unavailable.",
            )

        run_name = (
            f"agent_query_{request.user_query[:40].replace(' ', '_')}"
        )

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("user_query", request.user_query)
            run_id = run.info.run_id

            graph = build_graph(
                pipeline=rag_pipeline,
                generative_model=rag_pipeline.generative_model,
            )

            initial_state = {
                "query": request.user_query,
                "session_id": request.session_id,
                "qa_retries": 0,
                "metadata": {},
            }

            try:
                final_state = await graph.ainvoke(initial_state)
            except Exception as e:
                logger.error(f"Agent graph failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Agent pipeline error: {e}",
                )

            summary = final_state.get("summary", "")
            docs = final_state.get("retrieved_docs", [])
            domain = final_state.get("domain")
            confidence = final_state.get("confidence")
            faithfulness = final_state.get(
                "faithfulness_score", FAITHFULNESS_ERROR_SENTINEL
            )

            # MLflow logging
            mlflow.log_param("domain", domain)
            mlflow.log_param(
                "search_type", final_state.get("search_type", "unknown")
            )
            if faithfulness == FAITHFULNESS_ERROR_SENTINEL:
                mlflow.log_metric("faithfulness_eval_error", 1)
            elif faithfulness >= 0:
                mlflow.log_metric("faithfulness_score", faithfulness)

            # Persist artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                ctx_path = os.path.join(tmpdir, "context.txt")
                sum_path = os.path.join(tmpdir, "summary.txt")
                with open(ctx_path, "w", encoding="utf-8") as f:
                    f.write(
                        "\n\n---\n\n".join(
                            d.get("content", "") for d in docs
                        )
                    )
                with open(sum_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                mlflow.log_artifacts(tmpdir)

        metadata = {
            **(final_state.get("metadata") or {}),
            "source_files": final_state.get("source_files", []),
            "num_results": len(docs),
            "search_type": final_state.get("search_type", "unknown"),
            "qa_retries": final_state.get("qa_retries", 0),
        }

        return QueryResponse(
            query=request.user_query,
            domain=domain,
            confidence=confidence,
            summary=summary,
            results=docs,
            metadata=metadata,
            mlflow_run_id=run_id,
        )

    return router
