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

The project has evolved from a basic RAG pipeline into an intelligent, agent-driven system with access to real-time web information. The current implementation includes:

* **Intelligent Agent with Fallback Logic:** The core of the system is now an agent that uses a "fallback" retrieval strategy. It first searches the local knowledge base. If it determines the retrieved documents are not relevant to the user's query, it automatically falls back to performing a live web search to find the answer.
* **Web Search Capability:** The agent is integrated with the **Perplexity AI API**, giving it the ability to answer questions about recent events, new laws, or any topic not covered in the local document store.
* **LLM-Powered Relevance Checking:** An LLM is used as an intelligent judge to analyse the documents retrieved from the local database and decide if they are sufficient to answer the user's query before deciding to use the web search tool.
* **Vector Database & Hybrid Search:** **ChromaDB** remains the vector store for the local knowledge base, supporting a hybrid search that combines semantic and keyword retrieval.
* **State-of-the-Art AI Models:**
  * **Embedding Model:** Google's `text-embedding-004` creates high-quality vector embeddings.
  * **Generative Model:** Google's `gemma-3-27b-it` is used for both summarising local context and for the internal relevance-checking logic.
  * **Web Model:** The Perplexity model (configurable, e.g., `llama-3-sonar-large-32k-online`) is used for all web-based queries.
* **FastAPI Backend:** The entire system is served via a high-performance API with two primary endpoints:
  * `/ingest`: To dynamically upload new PDF documents to the local knowledge base.
  * `/retrieve`: The main endpoint that now orchestrates the "search local -> check relevance -> fallback to web" logic.
* **Modular & Scalable Codebase:** The project is structured into logical modules (`main.py`, `retriever_rag.py`, `ingestion_src.py`, `agent.py`) for maintainability.

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

GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
PERPLEXITY_API_KEY="YOUR_PERPLEXITY_API_KEY_HERE"
PERPLEXITY_MODEL_NAME="sonar"
```

Replace "YOUR_API_KEY_HERE" with your actual Google AI Studio API key.

### Step 5: Run the Application

You need to run the backend and frontend in  **two separate terminals** .

**Terminal 1: Start the FastAPI Backend**

(Make sure you are in the `app` directory with your virtual environment activated)

```bash
uvicorn main:app --reload
```

The backend API will now be running at `http://127.0.0.1:8000`

**Terminal 2: Start the Streamlit Frontend**
*(Open a new terminal, navigate to the `app` directory, and activate the same virtual environment)*

```
streamlit run streamlit_app.py
```

A new tab should automatically open in your browser at `http://localhost:8501`, displaying the user interface.

### Step 6: Interact with the Application

You can now use the Streamlit web interface to upload PDF documents and ask questions.

For direct API testing, you can still use the auto-generated documentation by navigating to `http://127.0.0.1:8000/docs` in your browser.

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
