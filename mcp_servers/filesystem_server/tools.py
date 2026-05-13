"""Tool handlers for the Filesystem MCP server."""
import base64
import hashlib
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "app"))

from config.settings import settings
from src.ingestion.ingestion_src import ingest_document_from_base64, get_category
from src.retriever.retriever_rag import HybridRAGPipeline


def _get_vectorstore():
    pipeline = HybridRAGPipeline.get_instance()
    return pipeline.vectorstore


async def upload_document(
    file_path: str,
    filename: str,
    metadata: dict | None = None,
) -> dict:
    """
    Ingest a document into the vector store from a local file path.

    Args:
        file_path: Absolute path to the file on disk
        filename: Original filename (used for category detection and metadata)
        metadata: Optional extra metadata to attach to each chunk

    Returns:
        {"chunks_added": int, "file_hash": str, "category": str, "status": "success"|"duplicate"}
    """
    with open(file_path, "rb") as f:
        raw = f.read()

    file_hash = hashlib.sha256(raw).hexdigest()
    b64 = base64.b64encode(raw).decode()
    ext = pathlib.Path(filename).suffix or ".pdf"
    category = get_category(filename)
    vs = _get_vectorstore()

    chunks_added = ingest_document_from_base64(
        vs, b64, filename, ext, metadata or {}
    )
    status = "duplicate" if chunks_added == 0 else "success"
    return {
        "chunks_added": chunks_added,
        "file_hash": file_hash,
        "category": category,
        "status": status,
    }


async def list_documents(
    collection: str | None = None,
    category_filter: str | None = None,
) -> list[dict]:
    """
    List all documents in the vector store.

    Args:
        collection: Ignored (single-collection store); reserved for future use
        category_filter: If provided, only return documents from this category

    Returns:
        [{"filename": str, "hash": str, "category": str, "chunk_count": int}]
    """
    vs = _get_vectorstore()
    result = vs.get(include=["metadatas"])
    metadatas = result.get("metadatas") or []

    # Aggregate per filename
    docs: dict[str, dict] = {}
    for meta in metadatas:
        fname = meta.get("filename", "unknown")
        if category_filter and meta.get("category") != category_filter:
            continue
        if fname not in docs:
            docs[fname] = {
                "filename": fname,
                "hash": meta.get("file_hash", ""),
                "category": meta.get("category", "Other"),
                "chunk_count": 0,
            }
        docs[fname]["chunk_count"] += 1

    return list(docs.values())


async def delete_document(
    filename: str | None = None,
    file_hash: str | None = None,
) -> dict:
    """
    Delete a document from the vector store by filename or hash.

    Returns:
        {"deleted": bool, "chunks_removed": int}
    """
    if not filename and not file_hash:
        return {"deleted": False, "chunks_removed": 0}

    vs = _get_vectorstore()
    where: dict = {}
    if file_hash:
        where = {"file_hash": file_hash}
    elif filename:
        where = {"filename": filename}

    existing = vs.get(where=where, include=["metadatas"])
    ids = existing.get("ids") or []
    if not ids:
        return {"deleted": False, "chunks_removed": 0}

    vs.delete(ids=ids)
    return {"deleted": True, "chunks_removed": len(ids)}


async def get_metadata(filename: str) -> dict:
    """
    Retrieve metadata for a specific document.

    Returns:
        {"filename", "category", "file_hash", "chunk_count", "filetype"} or {"error": str}
    """
    vs = _get_vectorstore()
    result = vs.get(where={"filename": filename}, include=["metadatas"])
    metadatas = result.get("metadatas") or []
    if not metadatas:
        return {"error": f"Document '{filename}' not found"}

    first = metadatas[0]
    return {
        "filename": first.get("filename", filename),
        "category": first.get("category", "Other"),
        "file_hash": first.get("file_hash", ""),
        "chunk_count": len(metadatas),
        "filetype": first.get("filetype", ""),
    }
