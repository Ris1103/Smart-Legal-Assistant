"""Typed async wrappers for the database MCP server tools."""
from mcp import ClientSession


class MCPDatabaseClient:
    def __init__(self, session: ClientSession):
        self._session = session

    async def save_contract(self, contract_type: str, params: dict, rendered_text: str) -> dict:
        result = await self._session.call_tool(
            "save_contract",
            {"contract_type": contract_type, "params": params, "rendered_text": rendered_text},
        )
        return result.content[0].text if result.content else {}

    async def query_contracts(
        self, contract_type: str | None = None, limit: int = 20, offset: int = 0
    ) -> list:
        args: dict = {"limit": limit, "offset": offset}
        if contract_type:
            args["contract_type"] = contract_type
        result = await self._session.call_tool("query_contracts", args)
        return result.content[0].text if result.content else []

    async def get_contract(self, contract_id: int) -> dict:
        result = await self._session.call_tool("get_contract", {"contract_id": contract_id})
        return result.content[0].text if result.content else {}

    async def get_template(self, contract_type: str) -> dict:
        result = await self._session.call_tool("get_template", {"contract_type": contract_type})
        return result.content[0].text if result.content else {}
