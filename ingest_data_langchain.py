import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm
import torch

# Path to your data folder
DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')

# Heuristic function to assign category based on filename
CATEGORY_KEYWORDS = {
    'GST': ['gst', 'cgst', 'igst'],
    'Income Tax': ['income-tax', 'income tax', 'tax'],
    'Penal Code': ['ipc', 'penal code'],
    'Company Act': ['ca act', 'companies act', 'moa'],
    'Shop Act': ['shop act'],
    'Rules': ['rules'],
    'Registration': ['registration'],
}

def get_category(filename):
    fname = filename.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in fname for kw in keywords):
            return category
    return 'Other'

# Prepare ChromaDB directory
CHROMA_DIR = os.path.join(os.path.dirname(__file__), 'chroma_db')
os.makedirs(CHROMA_DIR, exist_ok=True)

# Initialize embedding model (using the same model for embedding and retrieval)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}
)

# Initialize Chroma vector store
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

def ingest_pdf(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    return docs

def main():
    for fname in tqdm(os.listdir(DATA_DIR), desc='Ingesting files'):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[-1].lower()
        category = get_category(fname)
        metadata = {
            'filename': fname,
            'filetype': ext,
            'category': category,
        }
        # Only handle PDFs for now
        if ext == '.pdf':
            try:
                docs = ingest_pdf(fpath)
                # Attach metadata to each doc
                for doc in docs:
                    doc.metadata.update(metadata)
                # Add to vectorstore
                vectorstore.add_documents(docs)
                print(f"Ingested {fname} as category {category}")
            except Exception as e:
                print(f"Failed to ingest {fname}: {e}")
        else:
            print(f"Skipping unsupported file type: {fname}")
    vectorstore.persist()
    print("Ingestion complete. Vector DB persisted.")

if __name__ == '__main__':
    main()

print(torch.cuda.is_available())
