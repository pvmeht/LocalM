from llama_cpp import Llama
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma  # Updated import
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp  # Added for compatibility
import os

# Initialize the local model with LlamaCpp wrapper
llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=10,  # Offload 10 layers to GPU (adjust based on VRAM)
    n_ctx=2048,       # Context length
    verbose=True
)

# Function to load documents
def load_documents(directory="docs"):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue
        documents.extend(loader.load())
    return documents

# Set up embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Process documents and create vector store
def create_vector_store(documents):
    texts = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
    return vector_store

# Query function with fallback to LLM
def query_knowledge(query, vector_store, fallback_llm=None):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain({"query": query})
    answer = result["result"]
    
    # If no good answer, fallback to a larger LLM (optional)
    if not answer.strip() or "I don't know" in answer.lower():
        if fallback_llm:
            print("Falling back to larger LLM...")
            return fallback_llm(query)  # Replace with API call if needed
        return "I don't have enough information to answer that."
    return answer

# Main execution
if __name__ == "__main__":
    # Create a docs folder and add some test files (PDF, DOCX, TXT)
    os.makedirs("docs", exist_ok=True)
    
    # Load and process documents
    documents = load_documents("docs")
    if documents:
        vector_store = create_vector_store(documents)
        print("Knowledge base created!")
    else:
        print("No documents found in 'docs' folder.")
        vector_store = Chroma(embedding_function=embeddings, persist_directory="chroma_db")

    # Interactive query loop
    while True:
        query = input("Ask me anything (or 'quit' to exit): ")
        if query.lower() == "quit":
            break
        answer = query_knowledge(query, vector_store)
        print(f"Answer: {answer}\n")