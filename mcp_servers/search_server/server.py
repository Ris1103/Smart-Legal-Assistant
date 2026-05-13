"""Search MCP server — port 8003."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "app"))

from mcp.server.fastmcp import FastMCP
from tools import web_search as _web_search

mcp = FastMCP("legal-search")


@mcp.tool()
async def web_search(
    query: str,
    num_results: int = 5,
    provider: str | None = None,
) -> dict:
    """Search the web for legal information using the configured provider."""
    return await _web_search(query, num_results, provider)


if __name__ == "__main__":
    mcp.run(transport="sse", port=8003)
