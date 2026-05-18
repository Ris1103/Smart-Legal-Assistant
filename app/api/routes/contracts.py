"""
POST /contracts/generate — dedicated contract generation endpoint.
GET  /contracts          — list persisted contracts (MCP path only).
"""
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agents.contract_agent import ContractAgent
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class ContractRequest(BaseModel):
    user_query: str = Field(..., min_length=10)
    contract_type: Optional[str] = Field(
        None,
        description=(
            "Override auto-detection. "
            "One of: nda, service_agreement, employment_agreement"
        ),
    )
    params: Optional[Dict[str, Any]] = Field(
        {},
        description="Pre-filled parameters (skip LLM extraction).",
    )


class ContractResponse(BaseModel):
    contract_type: str
    contract_text: str
    params_used: Dict[str, Any]
    message: str


def get_contracts_router(rag_pipeline, verify_api_key, mcp_manager_fn=None):
    """
    Factory that creates the router with injected dependencies.
    """

    @router.get(
        "/contracts",
        response_model=List[Dict[str, Any]],
        dependencies=[Depends(verify_api_key)],
    )
    async def list_contracts(
        contract_type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ):
        mgr = mcp_manager_fn() if mcp_manager_fn else None
        if not settings.mcp_enabled or mgr is None:
            raise HTTPException(
                status_code=501,
                detail="Contract listing requires MCP_ENABLED=true.",
            )
        session = mgr.get("database")
        if not session:
            raise HTTPException(status_code=503, detail="Database MCP server unavailable.")
        from mcp_client.database_client import MCPDatabaseClient
        client = MCPDatabaseClient(session)
        result = await client.query_contracts(contract_type, limit, offset)
        if isinstance(result, str):
            result = json.loads(result)
        return result

    @router.post(
        "/contracts/generate",
        response_model=ContractResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def generate_contract(request: ContractRequest):
        if not rag_pipeline:
            logger.error("POST /contracts/generate — RAG Pipeline unavailable")
            raise HTTPException(
                status_code=503, detail="RAG Pipeline unavailable."
            )

        logger.info(
            "POST /contracts/generate contract_type=%r query_len=%d",
            request.contract_type,
            len(request.user_query),
        )

        try:
            agent = ContractAgent(
                generative_model=rag_pipeline.generative_model
            )

            initial_state: Dict[str, Any] = {
                "query": request.user_query,
                "metadata": {},
            }
            if request.contract_type:
                initial_state["contract_type"] = request.contract_type

            result = agent(initial_state)
        except Exception as exc:
            logger.error(
                "ContractAgent failed contract_type=%r: %s",
                request.contract_type, exc, exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail="Contract generation failed."
            )

        if result.get("error"):
            logger.warning(
                "ContractAgent returned error contract_type=%r: %s",
                request.contract_type, result["error"],
            )
            raise HTTPException(status_code=400, detail=result["error"])

        logger.info(
            "Contract generated contract_type=%s",
            result.get("contract_type", "unknown"),
        )
        return ContractResponse(
            contract_type=result.get("contract_type", "unknown"),
            contract_text=result.get("contract_text", ""),
            params_used=result.get("contract_params", {}),
            message=(
                "Contract generated successfully. Review and customise "
                "before use. This is not formal legal advice."
            ),
        )

    return router
