import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings

# Path to your data folder
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")

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
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

# —– UPDATED: use an instruction‐tuned embedding model for better alignment with the LLM
embedding_model = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-large",
    model_kwargs={"device": "cuda"},  # or "cpu" if you lack a GPU
)

# Note: pass .embed_query as the low‐level function Chroma expects
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model.embed_query)


def ingest_pdf(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    return docs


def main():
    for fname in tqdm(os.listdir(DATA_DIR), desc="Ingesting files"):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[-1].lower()
        category = get_category(fname)
        metadata = {
            "filename": fname,
            "filetype": ext,
            "category": category,
        }
        if ext == ".pdf":
            try:
                docs = ingest_pdf(fpath)
                for doc in docs:
                    doc.metadata.update(metadata)
                vectorstore.add_documents(docs)
                print(f"Ingested {fname} → category={category}")
            except Exception as e:
                print(f"Failed to ingest {fname}: {e}")
        else:
            print(f"Skipping unsupported file type: {fname}")
    vectorstore.persist()
    print("Ingestion complete. Vector DB persisted.")


if __name__ == "__main__":
    main()
