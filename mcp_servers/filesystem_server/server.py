"""Filesystem MCP server — port 8001."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "app"))

from mcp.server.fastmcp import FastMCP
from tools import upload_document as _upload, list_documents as _list
from tools import delete_document as _delete, get_metadata as _get_meta

mcp = FastMCP("legal-filesystem")


@mcp.tool()
async def upload_document(file_path: str, filename: str, metadata: dict | None = None) -> dict:
    """Ingest a document from disk into the vector store."""
    return await _upload(file_path, filename, metadata)


@mcp.tool()
async def list_documents(collection: str | None = None, category_filter: str | None = None) -> list:
    """List all documents in the vector store, optionally filtered by category."""
    return await _list(collection, category_filter)


@mcp.tool()
async def delete_document(filename: str | None = None, file_hash: str | None = None) -> dict:
    """Delete a document from the vector store by filename or hash."""
    return await _delete(filename, file_hash)


@mcp.tool()
async def get_metadata(filename: str) -> dict:
    """Get metadata for a specific document in the vector store."""
    return await _get_meta(filename)


if __name__ == "__main__":
    mcp.run(transport="sse", port=8001)
