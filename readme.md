# AI-Powered Legal Assistant for Indian SMBs

## 1. Project Goal

The ultimate goal of this project is to develop a sophisticated, AI-powered legal assistant specifically designed to meet the needs of Small and Medium-sized Businesses (SMBs) in India.

The system is envisioned as a **hybrid RAG (Retrieval-Augmented Generation) + Agent system** capable of:

- Providing contextually relevant legal advice based on a comprehensive knowledge base.
- Generating legal documents and contracts tailored to user needs.
- Offering a seamless, conversational user experience for complex legal queries.

To achieve this, the final architecture will implement technologies such as **ChromaDB** for semantic search, **pgVector** for advanced hybrid queries, **Redis** for high-speed caching, and **Celery** for asynchronous task processing. The knowledge base will be built using a rich corpus of Indian legal templates, acts, and regulations.

---

## 2. Current Status (What has been implemented so far)

A foundational version of the RAG pipeline has been successfully built and deployed as a web API. The current implementation includes:

- **Core RAG Pipeline:** A robust RAG system that can retrieve relevant information from a knowledge base and generate summarized answers.
- **Vector Database:** **ChromaDB** is used as the vector store to save, manage, and query document embeddings efficiently.
- **Hybrid Search:** The system employs a hybrid search mechanism that combines:
  - **Semantic Search:** Using vector similarity to find contextually relevant documents.
  - **Keyword Search:** Using traditional TF-IDF to match specific terms and phrases.
- **State-of-the-Art AI Models:**
  - **Embedding Model:** Google's `text-embedding-004` is used to create high-quality vector embeddings for all legal documents.
  - **Generative Model:** Google's `gemma-3-27b-it` is used to generate coherent, accurate summaries and answers from the retrieved context.
- **FastAPI Backend:** The entire pipeline is served through a high-performance API built with FastAPI, featuring two primary endpoints:
  - `/ingest`: A dynamic endpoint to upload new PDF documents, which are then processed and added to the knowledge base in real-time.
  - `/retrieve`: An endpoint to ask questions and receive context-aware, summarized answers from the AI.
- **Modular & Scalable Codebase:** The project is structured into logical, reusable modules (`main.py`, `retriever.py`, `ingestion_src.py`), promoting maintainability and future expansion.

---

## 3. How to Run the Project

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.12
- A virtual environment tool like `venv` (recommended)

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-directory>/app
```bash

### Step 2: Set Up the Python Environment
Create and activate a virtual environment within the app directory.

On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all the required packages from the requirements.txt file.

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a file named `.env` in the app directory. This file will hold your API key. Add the following line to it:

```bash

GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

Replace "YOUR_API_KEY_HERE" with your actual Google AI Studio API key.

### Step 5: Run the FastAPI Server

Start the web server using Uvicorn. The `--reload` flag will automatically restart the server when you make code changes.

```bash
uvicorn main:app --reload
```

The API will now be running at `http://127.0.0.1:8000`.

### Step 6: Interact with the API

You can interact with the API using any HTTP client (like Postman, Insomnia, or curl), or by using the auto-generated documentation.

**Interactive Docs (Swagger UI):** Open your browser and navigate to `http://127.0.0.1:8000/docs`. Here you can test the endpoints directly.

**Example curl Commands:**

To ingest a new document:
(You will need to first Base64-encode your PDF file)

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/ingest' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "base64_text": "YOUR_BASE64_ENCODED_PDF_STRING",
  "file_type": ".pdf",
  "filename": "my_new_legal_document.pdf",
  "metadata": {}
}'
```

To retrieve an answer:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/retrieve' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_query": "What are the penalties under the Income Tax Act?"
}'
```
