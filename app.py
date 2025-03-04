# app.py
import os
import logging
import threading
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from .schemas import QueryResponse, ImageUploadResponse, RefreshResponse
from .knowledge_base import query_knowledge, update_vector_store, refresh_memory
from .file_watcher import start_watcher

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PM MMR  Multimodal RAG API",
    description="A RESTful API for querying a multimodal (text + image) RAG system with Llama-2-7B and CLIP.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# FastAPI endpoints
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