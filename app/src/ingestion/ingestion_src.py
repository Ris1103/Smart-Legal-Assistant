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
        ValueError: If the file exceeds the configured size limit.
    """
    decoded_content = base64.b64decode(base64_text)

    size_mb = len(decoded_content) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise ValueError(
            f"File '{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{settings.max_file_size_mb} MB limit."
        )

    file_hash = hashlib.sha256(decoded_content).hexdigest()
    try:
        existing = vectorstore.get_by_metadata(where={"file_hash": file_hash}, limit=1)
        if existing and existing.get("ids"):
            logger.info(
                f"Document '{filename}' (hash={file_hash[:12]}…) "
                "already ingested. Skipping."
            )
            return 0
    except Exception as e:
        logger.warning(
            f"Could not check for duplicate (will proceed with ingestion): {e}"
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp:
            tmp.write(decoded_content)
            tmp_path = tmp.name

        logger.info(f"File '{filename}' saved to temporary path: {tmp_path}")

        # Choose chunking strategy
        if settings.chunk_strategy == "layout":
            from src.ingestion.layout_chunker import extract_layout_chunks
            chunks = extract_layout_chunks(
                tmp_path,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        else:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = get_chunker()
            chunks = text_splitter.split_documents(docs)

        category = get_category(filename)
        ext = os.path.splitext(filename)[-1].lower()

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
            # Preserve layout metadata (element_type, section_header, page) if present
            base_meta.update(chunk.metadata)
            base_meta.update(metadata)
            chunk.metadata = base_meta

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            vectorstore.add_documents(chunks[i: i + batch_size])

        logger.info(
            f"Successfully ingested {len(chunks)} chunks from '{filename}' "
            f"into category '{category}'."
        )
        return len(chunks)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"Cleaned up temporary file: {tmp_path}")
