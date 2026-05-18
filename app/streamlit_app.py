import logging
import time
import base64
import io

import requests
import streamlit as st
from pypdf import PdfReader

from config.settings import settings
from logging_config import setup_logging

setup_logging(
    level=settings.log_level,
    log_to_file=settings.log_to_file,
    log_dir=settings.log_dir,
    max_bytes=settings.log_max_bytes,
    backup_count=settings.log_backup_count,
)
logger = logging.getLogger(__name__)

# --- Configuration ---
st.set_page_config(page_title="Legal Assistant AI", layout="wide")
FASTAPI_URL = settings.fastapi_url
logger.info("Streamlit app started | api_url=%s", FASTAPI_URL)

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []
if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

# --- Sidebar ---
with st.sidebar:
    st.header("Actions")
    if st.button("Clear Conversation History"):
        logger.debug("User cleared conversation history")
        st.session_state.history = []
        st.success("History cleared!")

    st.header("About")
    st.info(
        "This application uses a Retrieval-Augmented Generation (RAG) "
        "pipeline. It first searches a local knowledge base. "
        "If no relevant information is found, it falls back to a live web search."
    )


# --- Helper Functions ---

def call_ingest_api(uploaded_file):
    """Encodes file to base64, extracts metadata, and calls the /ingest endpoint."""
    bytes_data = uploaded_file.getvalue()
    size_mb = len(bytes_data) / (1024 * 1024)
    logger.info("Ingesting file='%s' size=%.2f MB", uploaded_file.name, size_mb)

    base64_encoded_data = base64.b64encode(bytes_data).decode("utf-8")

    try:
        pdf_file = io.BytesIO(bytes_data)
        reader = PdfReader(pdf_file)
        num_pages = len(reader.pages)
        logger.debug("PDF metadata: file='%s' pages=%d", uploaded_file.name, num_pages)
    except Exception as exc:
        logger.warning("Could not read PDF metadata for '%s': %s", uploaded_file.name, exc)
        st.warning(f"Could not read PDF metadata (pages): {exc}")
        num_pages = "N/A"

    payload = {
        "base64_text": base64_encoded_data,
        "file_type": ".pdf",
        "filename": uploaded_file.name,
        "metadata": {
            "source": "fileupload",
            "upload_timestamp": str(time.time()),
            "no_of_pages": num_pages,
        },
    }

    try:
        t0 = time.time()
        response = requests.post(f"{FASTAPI_URL}/ingest", json=payload, timeout=60)
        response.raise_for_status()
        ingest_ms = int((time.time() - t0) * 1000)
        logger.info(
            "Ingest API response: file='%s' status=%s chunks=%s latency=%dms",
            uploaded_file.name,
            response.json().get("status"),
            response.json().get("chunks_added"),
            ingest_ms,
        )

        t1 = time.time()
        refresh_response = requests.post(f"{FASTAPI_URL}/refresh-index", timeout=30)
        refresh_response.raise_for_status()
        logger.debug("TF-IDF refresh completed in %dms", int((time.time() - t1) * 1000))

        return response.json()
    except requests.exceptions.Timeout:
        logger.error("Ingest API timed out for file='%s'", uploaded_file.name)
        st.error("Ingestion timed out. The file may be too large — try a smaller PDF.")
        return None
    except requests.exceptions.ConnectionError as exc:
        logger.error("Cannot connect to API at %s: %s", FASTAPI_URL, exc)
        st.error(f"Cannot reach the API server at {FASTAPI_URL}. Is the backend running?")
        return None
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        logger.error(
            "Ingest API HTTP %s for file='%s': %s", status, uploaded_file.name, detail
        )
        st.error(f"Ingestion failed (HTTP {status}): {detail or exc}")
        return None
    except requests.exceptions.RequestException as exc:
        logger.error("Ingest API request error for file='%s': %s", uploaded_file.name, exc)
        st.error(f"Ingestion failed: {exc}")
        return None


def call_retrieve_api(query: str, filename_filter: str = None):
    """Calls the /query endpoint (multi-agent pipeline)."""
    logger.info("Sending query len=%d filter=%r", len(query), filename_filter)
    payload = {"user_query": query, "response_style": "detailed"}

    try:
        t0 = time.time()
        response = requests.post(f"{FASTAPI_URL}/query", json=payload, timeout=120)
        response.raise_for_status()
        latency_ms = int((time.time() - t0) * 1000)

        data = response.json()
        logger.info(
            "Query response: domain=%s confidence=%.2f results=%d search_type=%s latency=%dms",
            data.get("domain"),
            data.get("confidence") or 0.0,
            len(data.get("results", [])),
            data.get("metadata", {}).get("search_type", "?"),
            latency_ms,
        )

        return {
            "query": data.get("query", query),
            "summary": data.get("summary", ""),
            "results": data.get("results", []),
            "citations": data.get("citations", []),
            "metadata": data.get("metadata", {}),
        }
    except requests.exceptions.Timeout:
        logger.error("Query API timed out for query len=%d", len(query))
        st.error("Query timed out. The request took too long — please try again.")
        return None
    except requests.exceptions.ConnectionError as exc:
        logger.error("Cannot connect to API at %s: %s", FASTAPI_URL, exc)
        st.error(f"Cannot reach the API server at {FASTAPI_URL}. Is the backend running?")
        return None
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        logger.error("Query API HTTP %s: %s", status, detail)
        st.error(f"Could not get answer (HTTP {status}): {detail or exc}")
        return None
    except requests.exceptions.RequestException as exc:
        logger.error("Query API request error: %s", exc)
        st.error(f"Could not get answer: {exc}")
        return None


# --- UI Components ---
st.title("AI-Powered Legal Assistant ⚖️")
st.markdown(
    "Upload a legal document (PDF) to add it to the knowledge base, then ask questions about its content or any general legal query."
)

# --- Main Application Flow ---

# 1. File Upload Section
st.header("1. Upload a Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None and uploaded_file.file_id != st.session_state.processed_file_id:
    with st.spinner(f"Processing and ingesting '{uploaded_file.name}'..."):
        ingest_result = call_ingest_api(uploaded_file)
        if ingest_result:
            st.session_state.processed_file_id = uploaded_file.file_id
            if ingest_result.get("status") == "duplicate":
                st.warning(
                    f"'{ingest_result['filename']}' is already in the "
                    "knowledge base — skipping re-ingestion."
                )
                st.session_state.last_uploaded_filename = ingest_result["filename"]
            else:
                st.success(
                    f"Successfully ingested '{ingest_result['filename']}'. "
                    f"Added {ingest_result['chunks_added']} text chunks "
                    "to the knowledge base."
                )
                st.session_state.last_uploaded_filename = ingest_result["filename"]
                st.info(
                    "Your next question will be specifically about "
                    f"**{ingest_result['filename']}**."
                )

st.divider()

# 2. Q&A Section
st.header("2. Ask a Question")
query_text = st.text_input("Enter your question here:", key="query_input")

if st.button("Get Answer"):
    if not query_text:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching for answers... This may take a moment."):
            filename_to_filter = st.session_state.last_uploaded_filename
            if filename_to_filter:
                st.info(f"Searching within the context of **{filename_to_filter}**...")

            retrieval_result = call_retrieve_api(query_text, filename_filter=filename_to_filter)

            if retrieval_result:
                st.session_state.history.insert(0, (query_text, retrieval_result))
                st.session_state.last_uploaded_filename = None

# 3. Display History
st.header("History")
if not st.session_state.history:
    st.info("Your conversation history will appear here.")
else:
    for i, (query, result) in enumerate(st.session_state.history):
        with st.expander(f"**Q: {query}**", expanded=(i == 0)):
            st.markdown("##### Summary")
            st.info(result["summary"])

            st.markdown("---")
            st.markdown("##### Sources")
            source_files = result["metadata"].get("source_files", ["N/A"])
            st.write(f"_{', '.join(source_files)}_")

            if result.get("results"):
                with st.container():
                    st.markdown("---")
                    st.markdown("##### Retrieved Context")
                    for doc_idx, doc in enumerate(result["results"]):
                        unique_key = f"doc_{i}_{doc_idx}"

                        if doc.get("metadata", {}).get("source") == "Perplexity Web Search":
                            st.markdown(f"**Source:** {doc['metadata']['source']}")
                        else:
                            st.markdown(
                                f"**Source:** `{doc.get('metadata', {}).get('filename', 'N/A')}` | **Category:** `{doc.get('metadata', {}).get('category', 'N/A')}`"
                            )

                        st.text_area(
                            label=f"Chunk {doc.get('metadata', {}).get('chunk_id', doc_idx)}",
                            value=doc["content"],
                            height=150,
                            disabled=True,
                            key=unique_key,
                        )
