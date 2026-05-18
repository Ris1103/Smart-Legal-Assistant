import os
import base64
import hashlib
import tempfile
import logging
from typing import Dict, Any

from langchain_community.document_loaders import PyPDFLoader

from config.settings import settings
from src.ingestion.chunker_factory import get_chunker
from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

CATEGORY_KEYWORDS = {
    "GST": ["gst", "cgst", "igst"],
    "Income Tax": ["income-tax", "income tax", "tax"],
    "Penal Code": ["ipc", "penal code"],
    "Company Act": ["ca act", "companies act", "companies_act", "moa"],
    "Shop Act": ["shop act"],
    "Rules": ["rules"],
    "Registration": ["registration"],
}


def get_category(filename: str) -> str:
    fname = filename.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in fname for kw in keywords):
            return category
    return "Other"


def ingest_document_from_base64(
    vectorstore: BaseVectorStore,
    base64_text: str,
    filename: str,
    file_type: str,
    metadata: Dict[str, Any],
) -> int:
    """
    Decodes a base64 string, validates the file, checks for duplicates,
    then ingests it into the vector store.

    Returns:
        Number of chunks added. Returns 0 if the document is a duplicate.

    Raises:
        ValueError: If the file exceeds the configured size limit or payload is invalid.
    """
    logger.info("Ingesting file='%s' type='%s'", filename, file_type)

    try:
        decoded_content = base64.b64decode(base64_text)
    except Exception as exc:
        logger.error("Failed to base64-decode payload for file='%s': %s", filename, exc)
        raise ValueError(f"Invalid base64 payload for '{filename}'.") from exc

    size_mb = len(decoded_content) / (1024 * 1024)
    logger.debug("Decoded file='%s' size=%.2f MB", filename, size_mb)
    if size_mb > settings.max_file_size_mb:
        logger.warning(
            "File='%s' rejected: size=%.1f MB exceeds limit=%d MB",
            filename, size_mb, settings.max_file_size_mb,
        )
        raise ValueError(
            f"File '{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{settings.max_file_size_mb} MB limit."
        )

    file_hash = hashlib.sha256(decoded_content).hexdigest()
    logger.debug("File='%s' sha256=%s…", filename, file_hash[:16])
    try:
        existing = vectorstore.get_by_metadata(where={"file_hash": file_hash}, limit=1)
        if existing and existing.get("ids"):
            logger.info(
                "Document '%s' (hash=%s…) already ingested — skipping.",
                filename, file_hash[:12],
            )
            return 0
    except Exception as exc:
        logger.warning(
            "Could not check for duplicate file='%s' (proceeding with ingestion): %s",
            filename, exc,
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp:
            tmp.write(decoded_content)
            tmp_path = tmp.name

        logger.debug("File='%s' written to tmp_path='%s'", filename, tmp_path)

        if settings.chunk_strategy == "layout":
            logger.debug("Using layout chunker for file='%s'", filename)
            from src.ingestion.layout_chunker import extract_layout_chunks
            chunks = extract_layout_chunks(
                tmp_path,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        else:
            logger.debug(
                "Using %s chunker (size=%d overlap=%d) for file='%s'",
                settings.chunk_strategy, settings.chunk_size,
                settings.chunk_overlap, filename,
            )
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = get_chunker()
            chunks = text_splitter.split_documents(docs)

        if not chunks:
            logger.warning("No chunks produced from file='%s' — PDF may be empty or unreadable", filename)
            return 0

        category = get_category(filename)
        ext = os.path.splitext(filename)[-1].lower()
        logger.debug("File='%s' category='%s' chunks=%d", filename, category, len(chunks))

        for i, chunk in enumerate(chunks):
            base_meta = {
                "filename": filename,
                "filetype": ext,
                "category": category,
                "source": filename,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "file_hash": file_hash,
            }
            base_meta.update(chunk.metadata)
            base_meta.update(metadata)
            chunk.metadata = base_meta

        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(chunks), batch_size)):
            batch = chunks[i: i + batch_size]
            logger.debug(
                "Storing batch %d/%d (%d chunks) for file='%s'",
                batch_idx + 1, total_batches, len(batch), filename,
            )
            try:
                vectorstore.add_documents(batch)
            except Exception as exc:
                logger.error(
                    "Failed to store batch %d/%d for file='%s': %s",
                    batch_idx + 1, total_batches, filename, exc, exc_info=True,
                )
                raise

        logger.info(
            "Ingested %d chunks from '%s' into category='%s'",
            len(chunks), filename, category,
        )
        return len(chunks)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug("Cleaned up tmp_path='%s'", tmp_path)
