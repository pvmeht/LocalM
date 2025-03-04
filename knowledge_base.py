# knowledge_base.py
import os
import logging
import shutil
import threading
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from PIL import Image
import torch
from .models import llm, clip_model, clip_processor, device
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Embeddings and text splitter
text_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Global vector store with thread lock
vector_store = None
vector_store_lock = threading.Lock()

# Cache embeddings (clear on refresh if needed)
@lru_cache(maxsize=1000)
def cached_embedding(text):
    return text_embeddings.embed_query(text)

# Load text documents with detailed logging
def load_text_documents(directory="docs"):
    documents = []
    try:
        logger.info(f"Loading documents from {directory}")
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist")
            return documents
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            logger.info(f"Processing file: {filepath}")
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath)
            else:
                logger.info(f"Skipping unsupported file: {filename}")
                continue
            loaded_docs = loader.load()
            logger.info(f"Loaded {len(loaded_docs)} documents from {filename}")
            documents.extend(loaded_docs)
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
    logger.info(f"Total documents loaded: {len(documents)}")
    for doc in documents:
        logger.debug(f"Document content: {doc.page_content[:100]}...")  # Log first 100 chars
    return documents

# Process image and generate caption
def process_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        
        candidate_texts = [
            "A sunset over a beach",
            "A dog in a park",
            "A person in a room",
            "A car on a road",
            "A mountain landscape"
        ]
        text_inputs = clip_processor(text=candidate_texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
        
        similarities = torch.nn.functional.cosine_similarity(image_features, text_features)
        best_match_idx = similarities.argmax().item()
        initial_caption = candidate_texts[best_match_idx]
        
        prompt = f"Based on the description '{initial_caption}', provide a brief description of the image."
        caption = llm(prompt, max_tokens=50, temperature=0.7, top_p=0.9)
        logger.info(f"Generated caption for {image_path}: {caption}")
        
        return Document(
            page_content=caption,
            metadata={"source": image_path}
        )
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return Document(
            page_content="Error processing image",
            metadata={"source": image_path}
        )

# Update vector store with detailed logging
def update_vector_store():
    global vector_store
    with vector_store_lock:
        try:
            documents = load_text_documents("docs")
            image_dir = "images"
            if os.path.exists(image_dir):
                logger.info(f"Loading images from {image_dir}")
                for filename in os.listdir(image_dir):
                    if filename.endswith((".png", ".jpg", ".jpeg")):
                        filepath = os.path.join(image_dir, filename)
                        image_doc = process_image(filepath)
                        documents.append(image_doc)
            
            if documents:
                logger.info(f"Indexing {len(documents)} documents into vector store")
                texts = text_splitter.split_documents(documents)
                logger.info(f"Split into {len(texts)} text chunks")
                if vector_store is None:
                    vector_store = Chroma.from_documents(texts, text_embeddings, persist_directory="chroma_db")
                    logger.info("Created new vector store")
                else:
                    vector_store.add_documents(texts)
                    logger.info("Added documents to existing vector store")
            else:
                logger.warning("No documents to index")
                if vector_store is None:
                    vector_store = Chroma(embedding_function=text_embeddings, persist_directory="chroma_db")
                    logger.info("Initialized empty vector store")
        except Exception as e:
            logger.error(f"Error updating vector store: {e}")

# Refresh memory and clear cache
def refresh_memory():
    global vector_store
    with vector_store_lock:
        try:
            logger.info("Refreshing memory: Clearing vector store...")
            vector_store = None
            chroma_db_path = "chroma_db"
            if os.path.exists(chroma_db_path):
                shutil.rmtree(chroma_db_path)
                logger.info("ChromaDB directory deleted.")
            os.makedirs(chroma_db_path, exist_ok=True)
            cached_embedding.cache_clear()  # Clear cached embeddings
            logger.info("Memory refresh complete, cache cleared.")
        except Exception as e:
            logger.error(f"Error refreshing memory: {e}")

# Query knowledge with detailed logging
def query_knowledge(query, temp=0.7, top_p=0.9):
    global vector_store
    with vector_store_lock:
        if vector_store is None:
            logger.info("Vector store is None, updating...")
            update_vector_store()
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        llm.temperature = temp
        llm.top_p = top_p
        try:
            logger.info(f"Processing query: {query}")
            result = qa_chain({"query": query})
            retrieved_docs = result["source_documents"]
            logger.info(f"Retrieved {len(retrieved_docs)} documents: {[doc.page_content[:100] for doc in retrieved_docs]}")
            answer = result["result"]
            logger.info(f"Generated answer: {answer[:100]}...")
            if not answer.strip() or "I don't know" in answer.lower():
                return "I don't have enough information to answer that."
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "Error processing query"