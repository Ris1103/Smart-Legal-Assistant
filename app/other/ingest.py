import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


# 1. Import the Google GenAI Embeddings class
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging

# Ensure the Google API key is set in the environment
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set. Please set it in your .env file."
    )

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to your data folder
# Note: Assumes the script is in the same root directory as the "Data" folder
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

# Heuristic function to assign category based on filename
CATEGORY_KEYWORDS = {
    "GST": ["gst", "cgst", "igst"],
    "Income Tax": ["income-tax", "income tax", "tax"],
    "Penal Code": ["ipc", "penal code"],
    "Company Act": ["ca act", "companies act", "moa"],
    "Shop Act": ["shop act"],
    "Rules": ["rules"],
    "Registration": ["registration"],
}


def get_category(filename):
    fname = filename.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in fname for kw in keywords):
            return category
    return "Other"


# Prepare ChromaDB directory
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

# 2. Updated: Use Google's text-embedding-004 model
# This model runs on Google's infrastructure, so no "device" argument is needed.
logger.info("Initializing Google Generative AI Embeddings...")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Initialize text splitter for better chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)

# Initialize ChromaDB with the new embedding function
# The interface is the same, so no other changes are needed here.
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
    collection_name="legal_documents",
)


def ingest_pdf(filepath):
    """Load and split PDF documents into chunks"""
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    # Split documents into chunks
    chunks = text_splitter.split_documents(docs)
    return chunks


def main():
    """Main ingestion function"""
    total_chunks = 0

    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory not found at: {DATA_DIR}")
        return

    file_list = os.listdir(DATA_DIR)
    if not file_list:
        logger.warning(f"No files found in the data directory: {DATA_DIR}")
        return

    for fname in tqdm(file_list, desc="Ingesting files"):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        ext = os.path.splitext(fname)[-1].lower()
        category = get_category(fname)

        if ext == ".pdf":
            try:
                chunks = ingest_pdf(fpath)

                # Add metadata to each chunk
                for i, chunk in enumerate(chunks):
                    # Combine existing metadata with new metadata
                    new_metadata = {
                        "filename": fname,
                        "filetype": ext,
                        "category": category,
                        "source": fpath,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                    }
                    chunk.metadata.update(new_metadata)

                # Add documents to vectorstore
                batch_size = 100
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    vectorstore.add_documents(batch)

                total_chunks += len(chunks)

                logger.info(f"Ingested {fname} → category={category}, chunks={len(chunks)}")

            except Exception as e:
                logger.error(f"Failed to ingest {fname}: {e}")
        else:
            logger.warning(f"Skipping unsupported file type: {fname}")

    # Persist the vectorstore
    if total_chunks > 0:
        logger.info("Persisting the vectorstore...")
        logger.info(f"Ingestion complete. Total chunks: {total_chunks}. Vector DB persisted.")
    else:
        logger.info("No new chunks were added. Ingestion finished.")


if __name__ == "__main__":
    main()
