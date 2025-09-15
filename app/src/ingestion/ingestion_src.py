import os
import base64
import tempfile
import logging
from typing import Dict, Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# --- CATEGORIZATION LOGIC: Moved directly from your original script ---
CATEGORY_KEYWORDS = {
    "GST": ["gst", "cgst", "igst"],
    "Income Tax": ["income-tax", "income tax", "tax"],
    "Penal Code": ["ipc", "penal code"],
    "Company Act": ["ca act", "companies act", "moa"],
    "Shop Act": ["shop act"],
    "Rules": ["rules"],
    "Registration": ["registration"],
}


def get_category(filename: str) -> str:
    """Assigns a category to a file based on its name."""
    fname = filename.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in fname for kw in keywords):
            return category
    return "Other"


def ingest_document_from_base64(
    vectorstore: Chroma,
    base64_text: str,
    filename: str,
    file_type: str,
    metadata: Dict[str, Any],
) -> int:
    """
    Decodes a base64 string, processes the file with full original logic,
    and ingests it into the vector store.
    """
    tmp_path = None
    try:
        # 1. Decode and save to a temporary file
        decoded_content = base64.b64decode(base64_text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp:
            tmp.write(decoded_content)
            tmp_path = tmp.name

        logger.info(f"File '{filename}' saved to temporary path: {tmp_path}")

        # 2. Load and Split the document
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # 3. --- CORE LOGIC: Replicating the original script's metadata creation ---
        category = get_category(filename)
        ext = os.path.splitext(filename)[-1].lower()

        for i, chunk in enumerate(chunks):
            # Create the detailed metadata dictionary for each chunk
            new_metadata = {
                "filename": filename,
                "filetype": ext,
                "category": category,
                # For API ingestion, 'source' is best represented by the filename
                "source": filename,
                "chunk_id": i,
                "total_chunks": len(chunks),
            }
            # Combine with any metadata passed from the API call
            new_metadata.update(metadata)
            chunk.metadata = new_metadata
        # -------------------------------------------------------------------------

        # 4. Add chunks to the vector store in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectorstore.add_documents(batch)

        logger.info(
            f"Successfully ingested {len(chunks)} chunks from '{filename}' "
            f"into category '{category}'."
        )
        return len(chunks)

    finally:
        # 5. Clean up the temporary file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info(f"Cleaned up temporary file: {tmp_path}")
