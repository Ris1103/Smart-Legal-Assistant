from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.pipelines import pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import torch

# Use a publicly available LLAMA model or Mistral model
llama_model_name = "meta-llama/Llama-2-7b-hf"  # Public Llama-2 model (instead of gated meta-llama)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})

# Initialize Chroma vector store
CHROMA_DIR = './chroma_db'
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

# Load the model for summarization (LLAMA-2 or Mistral)
summarizer = AutoModelForSeq2SeqLM.from_pretrained(llama_model_name)
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)

# System prompt to guide summarization
system_prompt = """
You are a legal assistant for small businesses in India. Based on the retrieved legal documents from the [CATEGORY] category, summarize the key points that answer the user's query. The response should be concise, clear, and easy to understand for a non-expert. Focus on providing actionable insights and breaking down complex legal terms into simple language. The user query is as follows:

USER QUERY: [USER_QUERY]
SUMMARIZE the relevant details from the retrieved documents in the following manner:
1. Key points about the law or regulation.
2. Specific legal advice relevant to small businesses.
3. Any important compliance steps or recommendations.
"""

# Function to categorize the query
def categorize_query(user_query):
    CATEGORY_KEYWORDS = {
        'GST': ['gst', 'cgst', 'igst'],
        'Income Tax': ['income-tax', 'income tax', 'tax'],
        'Penal Code': ['ipc', 'penal code'],
        'Company Act': ['ca act', 'companies act', 'moa'],
        'Shop Act': ['shop act'],
        'Rules': ['rules'],
        'Registration': ['registration'],
    }
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in user_query.lower() for kw in keywords):
            return category
    return 'Other'

# Retrieve documents based on query
def retrieve_documents(user_query):
    # Categorize the query
    category = categorize_query(user_query)
    
    # Retrieve relevant documents based on the category
    docs = vectorstore.similarity_search(user_query, category=category)
    
    return docs

# Function to summarize the documents with system prompt
def summarize_documents(documents, user_query):
    # Extract text from documents
    doc_texts = [doc.page_content for doc in documents]
    
    # Combine text into a single string for summarization
    doc_content = " ".join(doc_texts)
    
    # Format the system prompt with user query and retrieved documents
    prompt = system_prompt.replace("[CATEGORY]", documents[0].metadata['category']).replace("[USER_QUERY]", user_query)
    
    # Tokenize and summarize the content using LLAMA model with system prompt
    inputs = tokenizer(prompt + "\n\n" + doc_content, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    summary_ids = summarizer.generate(inputs["input_ids"], max_length=300, min_length=100, length_penalty=2.0)
    
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Process the user query
def process_query(user_query):
    # Retrieve documents based on the query
    documents = retrieve_documents(user_query)
    
    # Summarize the retrieved documents with the system prompt
    summary = summarize_documents(documents, user_query)
    
    return summary

# Example usage
user_query = "What are the GST compliance requirements for small businesses?"
answer = process_query(user_query)
print(answer)
