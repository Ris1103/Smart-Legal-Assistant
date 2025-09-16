import streamlit as st
import requests
import base64
import time
import io
from pypdf import PdfReader

# --- Configuration ---
st.set_page_config(page_title="Legal Assistant AI", layout="wide")
FASTAPI_URL = "http://127.0.0.1:8000"

# --- Session State Initialization ---
# Initialize keys for conversation history and for managing file processing logic.
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
    base64_encoded_data = base64.b64encode(bytes_data).decode("utf-8")

    try:
        pdf_file = io.BytesIO(bytes_data)
        reader = PdfReader(pdf_file)
        num_pages = len(reader.pages)
    except Exception as e:
        st.warning(f"Could not read PDF metadata (pages): {e}")
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
        response = requests.post(f"{FASTAPI_URL}/ingest", json=payload, timeout=60)
        response.raise_for_status()
        refresh_response = requests.post(f"{FASTAPI_URL}/refresh-index", timeout=30)
        refresh_response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ingestion failed: {e}")
        return None


def call_retrieve_api(query: str, filename_filter: str = None):
    """Calls the /retrieve endpoint, optionally with a filename filter."""
    payload = {"user_query": query, "filename_filter": filename_filter}
    try:
        response = requests.post(f"{FASTAPI_URL}/retrieve", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not get answer: {e}")
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

# FIX: Check if the file has already been processed to prevent re-ingestion.
if uploaded_file is not None and uploaded_file.file_id != st.session_state.processed_file_id:
    with st.spinner(f"Processing and ingesting '{uploaded_file.name}'..."):
        ingest_result = call_ingest_api(uploaded_file)
        if ingest_result:
            st.success(
                f"Successfully ingested '{ingest_result['filename']}'. "
                f"Added {ingest_result['chunks_added']} text chunks to the knowledge base."
            )
            # ENHANCEMENT: Store the filename for the next query.
            st.session_state.last_uploaded_filename = ingest_result["filename"]
            # FIX: Mark this file as processed.
            st.session_state.processed_file_id = uploaded_file.file_id
            st.info(
                f"Your next question will be specifically about **{ingest_result['filename']}**."
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
            # ENHANCEMENT: Check if the last query should be scoped to the uploaded file.
            filename_to_filter = st.session_state.last_uploaded_filename
            if filename_to_filter:
                st.info(f"Searching within the context of **{filename_to_filter}**...")

            retrieval_result = call_retrieve_api(query_text, filename_filter=filename_to_filter)

            if retrieval_result:
                st.session_state.history.insert(0, (query_text, retrieval_result))
                # ENHANCEMENT: Clear the filter so the *next* search is global.
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
                        # Unique key for each text_area
                        unique_key = f"doc_{i}_{doc_idx}"

                        if doc.get("metadata", {}).get("source") == "Perplexity Web Search":
                            st.markdown(f"**Source:** {doc['metadata']['source']}")
                        else:
                            # FIX: Combined into a single f-string to remove SyntaxWarning.
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
