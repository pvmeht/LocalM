import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from utils import extract_text
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embeddings with a lightweight model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize persistent vector store with collection name for better management
client = chromadb.PersistentClient(path="chroma_db")
vector_store = Chroma(
    client=client,
    embedding_function=embeddings,
    collection_name="student_notes"  # Named collection for clarity
)

def add_document_to_vector_store(file_path, metadata):
    """Add a document to the vector store with metadata."""
    text = extract_text(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    vector_store.add_texts(texts=chunks, metadatas=[metadata] * len(chunks))
    logger.info(f"Added document: {file_path} with {len(chunks)} chunks")

# Define the prompt template for RAG
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Based on the following context from uploaded notes, answer the question:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)

def query_knowledge(question, llm, k=2, max_context_tokens=1500, temp=0.7, top_p=0.9):
    """Query the vector store and generate an answer using the LLM efficiently."""
    start_time = time.time()
    docs = vector_store.similarity_search(question, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    
    if len(context.split()) > max_context_tokens:
        context = " ".join(context.split()[:max_context_tokens])
    
    prompt = prompt_template.format(context=context, question=question)
    logger.info(f"Prompt prepared in {time.time() - start_time:.2f}s, length: {len(prompt.split())} words")
    
    # Set LLM parameters
    llm.temperature = temp
    llm.top_p = top_p
    
    start_time = time.time()
    response = llm(prompt)
    logger.info(f"LLM response generated in {time.time() - start_time:.2f}s")
    
    return response.strip() if response.strip() else "I don't have enough information to answer that."