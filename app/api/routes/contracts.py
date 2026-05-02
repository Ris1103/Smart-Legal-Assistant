"""
POST /contracts/generate — dedicated contract generation endpoint.
"""
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agents.contract_agent import ContractAgent

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


def get_contracts_router(rag_pipeline, verify_api_key):
    """
    Factory that creates the router with injected dependencies.
    """

    @router.post(
        "/contracts/generate",
        response_model=ContractResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def generate_contract(request: ContractRequest):
        if not rag_pipeline:
            raise HTTPException(
                status_code=503, detail="RAG Pipeline unavailable."
            )

        agent = ContractAgent(
            generative_model=rag_pipeline.generative_model
        )

        # Build an initial AgentState-like dict
        initial_state: Dict[str, Any] = {
            "query": request.user_query,
            "metadata": {},
        }

        # If contract_type is explicitly provided, inject it so the
        # agent's detection skips or is overridden.
        if request.contract_type:
            initial_state["contract_type"] = request.contract_type

        result = agent(initial_state)

        if result.get("error"):
            raise HTTPException(
                status_code=400, detail=result["error"]
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
