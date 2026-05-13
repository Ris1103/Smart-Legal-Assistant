"""Typed async wrappers for the search MCP server tools."""
from mcp import ClientSession


class MCPSearchClient:
    def __init__(self, session: ClientSession):
        self._session = session

    async def web_search(
        self,
        query: str,
        num_results: int = 5,
        provider: str | None = None,
    ) -> dict:
        args: dict = {"query": query, "num_results": num_results}
        if provider:
            args["provider"] = provider
        result = await self._session.call_tool("web_search", args)
        return result.content[0].text if result.content else {}
