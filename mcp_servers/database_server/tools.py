"""Tool handlers for the Database MCP server."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "app"))

from store import insert_contract, fetch_contracts, fetch_contract

_TEMPLATES_DIR = pathlib.Path(__file__).resolve().parents[2] / "app" / "templates" / "contracts"


async def save_contract(contract_type: str, params: dict, rendered_text: str) -> dict:
    """
    Persist a rendered contract.

    Returns:
        {"id": int, "created_at": str}
    """
    return await insert_contract(contract_type, params, rendered_text)


async def query_contracts(
    contract_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """
    List saved contracts (no rendered_text in list view).

    Returns:
        [{"id", "contract_type", "params", "created_at"}]
    """
    return await fetch_contracts(contract_type, limit, offset)


async def get_contract(contract_id: int) -> dict:
    """
    Retrieve a single contract by ID, including rendered_text.

    Returns:
        Full contract record or {"error": str}
    """
    return await fetch_contract(contract_id)


async def get_template(contract_type: str) -> dict:
    """
    Return the raw Jinja2 template text for a given contract type.

    Returns:
        {"contract_type": str, "template_text": str} or {"error": str}
    """
    path = _TEMPLATES_DIR / f"{contract_type}.j2"
    if not path.exists():
        available = [p.stem for p in _TEMPLATES_DIR.glob("*.j2")]
        return {"error": f"Template '{contract_type}' not found. Available: {available}"}
    return {"contract_type": contract_type, "template_text": path.read_text(encoding="utf-8")}
