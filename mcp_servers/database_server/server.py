"""Database MCP server — port 8002."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "app"))

from mcp.server.fastmcp import FastMCP
from tools import save_contract as _save, query_contracts as _query
from tools import get_contract as _get, get_template as _template

mcp = FastMCP("legal-database")


@mcp.tool()
async def save_contract(contract_type: str, params: dict, rendered_text: str) -> dict:
    """Persist a rendered contract to SQLite."""
    return await _save(contract_type, params, rendered_text)


@mcp.tool()
async def query_contracts(
    contract_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list:
    """List saved contracts (excludes rendered_text for brevity)."""
    return await _query(contract_type, limit, offset)


@mcp.tool()
async def get_contract(contract_id: int) -> dict:
    """Fetch a full contract record including rendered_text."""
    return await _get(contract_id)


@mcp.tool()
async def get_template(contract_type: str) -> dict:
    """Return the Jinja2 template source for a given contract type."""
    return await _template(contract_type)


if __name__ == "__main__":
    mcp.run(transport="sse", port=8002)
