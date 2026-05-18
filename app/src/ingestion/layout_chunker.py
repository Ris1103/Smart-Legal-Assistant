"""
Layout-aware PDF chunker using pdfplumber.

Extracts text preserving document structure: section headers are prepended
to following chunks, tables are serialised to markdown, reading order is
maintained. Falls back to RecursiveCharacterTextSplitter on any error.
"""
import logging
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(
    r"^(?:\d+[\.\d]*\s+[A-Z]|[A-Z][A-Z\s]{4,}|Section\s+\d+|PART\s+[IVX]+)",
    re.MULTILINE,
)


def _table_to_markdown(table: list) -> str:
    """Convert pdfplumber table (list of rows) to markdown string."""
    if not table:
        return ""
    rows = [[str(cell or "").strip() for cell in row] for row in table]
    # Filter fully empty rows
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join([header, separator, body])


def extract_layout_chunks(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Extract chunks from a PDF with layout awareness.

    Returns a list of Documents with metadata:
      element_type: "text" | "table"
      section_header: nearest heading above the chunk
      page: page number (1-based)
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        raise

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    all_chunks: List[Document] = []
    current_header = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract tables first so we can exclude their bboxes from text
                tables = page.extract_tables()
                table_bboxes = [t.bbox for t in page.find_tables()] if tables else []

                # --- Tables ---
                for table in tables:
                    md = _table_to_markdown(table)
                    if md:
                        all_chunks.append(Document(
                            page_content=f"[Table]\n{md}",
                            metadata={
                                "element_type": "table",
                                "section_header": current_header,
                                "page": page_num,
                            },
                        ))

                # --- Text (excluding table areas) ---
                if table_bboxes:
                    # Crop away table regions
                    text_page = page
                    for bbox in table_bboxes:
                        try:
                            text_page = text_page.outside_bbox(bbox)
                        except Exception:
                            pass
                    raw_text = text_page.extract_text() or ""
                else:
                    raw_text = page.extract_text() or ""

                if not raw_text.strip():
                    continue

                # Detect section headers in this page's text
                for line in raw_text.splitlines():
                    if _HEADER_RE.match(line.strip()):
                        current_header = line.strip()[:120]
                        break

                # Split text into chunks
                text_docs = splitter.create_documents(
                    [raw_text],
                    metadatas=[{
                        "element_type": "text",
                        "section_header": current_header,
                        "page": page_num,
                    }],
                )
                # Prepend section header to each chunk for context
                for doc in text_docs:
                    if current_header and not doc.page_content.startswith(current_header):
                        doc.page_content = f"[Section: {current_header}]\n{doc.page_content}"
                all_chunks.extend(text_docs)

    except Exception as e:
        logger.error(f"Layout extraction failed for {pdf_path}: {e}. Falling back.")
        return _fallback_chunks(pdf_path, chunk_size, chunk_overlap)

    if not all_chunks:
        return _fallback_chunks(pdf_path, chunk_size, chunk_overlap)

    logger.info(f"Layout chunker produced {len(all_chunks)} chunks from {pdf_path}")
    return all_chunks


def _fallback_chunks(pdf_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
