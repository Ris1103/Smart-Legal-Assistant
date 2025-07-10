import os
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import Chroma  # Corrected import
from langchain_huggingface import HuggingFaceEmbeddings  # Corrected import
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# 1) Set up Hugging Face API client
client = InferenceClient(
    token="Your_token_here",  # Replace with your actual Hugging Face token
)

# 2) Embedding — using the correct embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-large",
    model_kwargs={"device": "cuda"},  # Or "cpu" if you lack a GPU
)

# 3) Vector store for storing and retrieving documents
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model.embed_query,  # Correct method to use
)


# 4) Function to interact with Hugging Face API for generating structured summaries
def generate_structured_summary(text):
    prompt = f"""
    Please summarize the following content in a structured format with the following sections:
    1. **Key Points**: Provide a list of the main points from the content.
    2. **Conclusion**: Provide the conclusion based on the content.
    3. **Important Notes**: Any notable or important details to remember.

    Content: {text}
    """

    try:
        # Use text generation API (Hugging Face Inference API)
        response = client.text_generation(
            model="google/flan-ul2",  # Choose the appropriate model
            prompt=prompt,
            max_new_tokens=150,  # Adjust token limit as needed
        )
        return response["generated_text"]
    except Exception as e:
        print(f"Error accessing the model: {e}")
        return None


# 5) Set up HuggingFacePipeline for the summarization task
summarization_pipeline = HuggingFacePipeline(pipeline=client.text_generation)

# 6) Build a RetrievalQA chain for hybrid search and summarization
qa_chain = RetrievalQA.from_chain_type(
    llm=summarization_pipeline,  # Pass HuggingFacePipeline as the LLM (Runnable)
    chain_type="stuff",  # or "map_reduce" for very large docs
    retriever=vectorstore.as_retriever(),
)


def process_query(user_query: str) -> str:
    """
    1. Run RetrievalQA to get the relevant documents.
    2. Generate a structured summary from the retrieved documents.
    3. Save the summary to summary.txt.
    4. Return the structured summary.
    """
    # Step 1: Retrieve documents using the RAG chain
    retrieved_docs = qa_chain.run(user_query)

    # Step 2: Generate a structured summary of the retrieved documents
    structured_summary = generate_structured_summary(retrieved_docs)

    if structured_summary:
        # Step 3: Save the structured summary to a file
        out_path = os.path.join(os.path.dirname(__file__), "summary.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(structured_summary)

        print(f"Summary written to {out_path}")
        return structured_summary
    else:
        return "Error: Unable to generate a summary."


if __name__ == "__main__":
    # Example query
    query = "What are the GST compliance requirements for small businesses?"
    answer = process_query(query)
    print("=== STRUCTURED SUMMARY ===\n", answer)
