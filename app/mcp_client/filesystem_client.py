"""Typed async wrappers for the filesystem MCP server tools."""
from mcp import ClientSession


class MCPFilesystemClient:
    def __init__(self, session: ClientSession):
        self._session = session

    async def upload_document(self, file_path: str, filename: str, metadata: dict | None = None) -> dict:
        args: dict = {"file_path": file_path, "filename": filename}
        if metadata:
            args["metadata"] = metadata
        result = await self._session.call_tool("upload_document", args)
        return result.content[0].text if result.content else {}

    async def list_documents(self, collection: str | None = None, category_filter: str | None = None) -> list:
        args: dict = {}
        if collection:
            args["collection"] = collection
        if category_filter:
            args["category_filter"] = category_filter
        result = await self._session.call_tool("list_documents", args)
        return result.content[0].text if result.content else []

    async def delete_document(self, filename: str | None = None, file_hash: str | None = None) -> dict:
        args: dict = {}
        if filename:
            args["filename"] = filename
        if file_hash:
            args["file_hash"] = file_hash
        result = await self._session.call_tool("delete_document", args)
        return result.content[0].text if result.content else {}

    async def get_metadata(self, filename: str) -> dict:
        result = await self._session.call_tool("get_metadata", {"filename": filename})
        return result.content[0].text if result.content else {}
