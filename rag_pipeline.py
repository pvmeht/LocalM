import os
import shutil
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_core.documents import Document
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn
import threading
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define Pydantic models for API responses
class QueryResponse(BaseModel):
    query: str
    answer: str
    temp: float
    top_p: float

class ImageUploadResponse(BaseModel):
    filename: str
    status: str

class RefreshResponse(BaseModel):
    status: str

# Initialize FastAPI app with Swagger UI metadata
app = FastAPI(
    title="Multimodal RAG API",
    description="A RESTful API for querying a multimodal (text + image) RAG system with Llama-2-7B and CLIP.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Local text model (Llama-2-7B-Chat-Q4_K_M)
llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=10,
    n_ctx=2048,
    verbose=True,
    n_batch=512
)

# CLIP model for image recognition
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Embeddings and text splitter for text data
text_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Global vector store with thread lock for safety
vector_store = None
vector_store_lock = threading.Lock()

# Cache embeddings for repeated queries
@lru_cache(maxsize=1000)
def cached_embedding(text):
    return text_embeddings.embed_query(text)

# Load text documents
def load_text_documents(directory="docs"):
    documents = []
    try:
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
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
    return documents

# Process image and generate a descriptive caption
def process_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        
        # Candidate descriptions for CLIP to rank
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
        
        # Compute similarity between image and text features
        similarities = torch.nn.functional.cosine_similarity(image_features, text_features)
        best_match_idx = similarities.argmax().item()
        initial_caption = candidate_texts[best_match_idx]
        
        # Refine with LLM
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

# Update vector store with text and image data
def update_vector_store():
    global vector_store
    with vector_store_lock:
        try:
            documents = load_text_documents("docs")
            image_dir = "images"
            if os.path.exists(image_dir):
                for filename in os.listdir(image_dir):
                    if filename.endswith((".png", ".jpg", ".jpeg")):
                        filepath = os.path.join(image_dir, filename)
                        image_doc = process_image(filepath)
                        documents.append(image_doc)
            
            if documents:
                logger.info("Updating vector store...")
                texts = text_splitter.split_documents(documents)
                if vector_store is None:
                    vector_store = Chroma.from_documents(texts, text_embeddings, persist_directory="chroma_db")
                else:
                    vector_store.add_documents(texts)
                logger.info("Vector store updated successfully!")
            elif vector_store is None:
                vector_store = Chroma(embedding_function=text_embeddings, persist_directory="chroma_db")
        except Exception as e:
            logger.error(f"Error updating vector store: {e}")

# Refresh memory by clearing vector store
def refresh_memory():
    global vector_store
    with vector_store_lock:
        try:
            logger.info("Refreshing memory: Clearing vector store...")
            vector_store = None  # Reset in-memory store
            chroma_db_path = "chroma_db"
            if os.path.exists(chroma_db_path):
                shutil.rmtree(chroma_db_path)  # Delete persisted data
                logger.info("ChromaDB directory deleted.")
            os.makedirs(chroma_db_path, exist_ok=True)  # Recreate empty directory
            logger.info("Memory refresh complete.")
        except Exception as e:
            logger.error(f"Error refreshing memory: {e}")

# File system watcher
class FileWatcher(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            if event.src_path.startswith(os.path.join(os.getcwd(), "docs")) or \
               event.src_path.startswith(os.path.join(os.getcwd(), "images")):
                logger.info(f"New file detected: {event.src_path}")
                update_vector_store()

def start_watcher():
    observer = Observer()
    observer.schedule(FileWatcher(), path="docs", recursive=False)
    observer.schedule(FileWatcher(), path="images", recursive=False)
    observer.start()
    logger.info("File watcher started for docs and images!")

# Query function with optimization
def query_knowledge(query, temp=0.7, top_p=0.9):
    global vector_store
    with vector_store_lock:
        if vector_store is None:
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
            logger.info(f"Retrieved documents: {[doc.page_content for doc in result['source_documents']]}")
            answer = result["result"]
            if not answer.strip() or "I don't know" in answer.lower():
                return "I don't have enough information to answer that."
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "Error processing query"

# FastAPI endpoints with Swagger documentation
@app.get("/query/", response_model=QueryResponse, 
         summary="Query the multimodal RAG system",
         description="Submit a text query to retrieve answers from text documents and image descriptions.")
async def api_query(query: str = "What’s in my files?", 
                    temp: float = 0.7, 
                    top_p: float = 0.9):
    try:
        answer = query_knowledge(query, temp, top_p)
        return {"query": query, "answer": answer, "temp": temp, "top_p": top_p}
    except Exception as e:
        logger.error(f"API query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-image/", response_model=ImageUploadResponse,
          summary="Upload an image",
          description="Upload an image file (PNG/JPG) to be processed and added to the knowledge base.")
async def upload_image(file: UploadFile = File(...)):
    try:
        os.makedirs("images", exist_ok=True)
        file_path = os.path.join("images", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logger.info(f"Image uploaded: {file_path}")
        threading.Thread(target=update_vector_store, daemon=True).start()
        return {"filename": file.filename, "status": "Image uploaded and processing started"}
    except Exception as e:
        logger.error(f"Image upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh-memory/", response_model=RefreshResponse,
          summary="Refresh system memory",
          description="Clear all stored data in the vector store to reset the knowledge base.")
async def refresh_memory_endpoint():
    try:
        refresh_memory()
        return {"status": "Memory refreshed successfully"}
    except Exception as e:
        logger.error(f"Refresh memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Interactive console
def interactive_mode():
    logger.info("Interactive mode started! Type 'quit' to exit.")
    while True:
        query = input("Ask me anything (or 'quit' to exit): ")
        if query.lower() == "quit":
            break
        temp = float(input("Temperature (0.0-1.0, default 0.7): ") or 0.7)
        top_p = float(input("Top_p (0.0-1.0, default 0.9): ") or 0.9)
        answer = query_knowledge(query, temp, top_p)
        print(f"Answer: {answer}\n")

# Main execution
if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    update_vector_store()
    
    watcher_thread = threading.Thread(target=start_watcher, daemon=True)
    watcher_thread.start()
    
    interactive_thread = threading.Thread(target=interactive_mode)
    interactive_thread.start()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)