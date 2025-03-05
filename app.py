import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from schemas import QueryRequest, QueryResponse, NotesUploadResponse, ImageUploadRequest, ImageUploadResponse, RefreshResponse
from rag_engine import update_vector_store, query_rag
from utils import process_image
import uvicorn
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="StudentRAG-V3",
    description="A multimodal RAG system for students.",
    version="3.0.0"
)

@app.post("/upload-notes/", response_model=NotesUploadResponse)
async def upload_notes(file: UploadFile = File(...)):
    os.makedirs("docs", exist_ok=True)
    file_path = os.path.join("docs", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    threading.Thread(target=update_vector_store, daemon=True).start()
    return {"filename": file.filename, "status": "Notes uploaded"}

@app.post("/upload-image/", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...), topic: str | None = None):
    os.makedirs("images", exist_ok=True)
    file_path = os.path.join("images", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    image_doc = process_image(file_path, topic)
    threading.Thread(target=update_vector_store, daemon=True).start()
    return {"filename": file.filename, "topic": topic, "description": image_doc.page_content, "status": "Image processed"}

@app.post("/query/", response_model=QueryResponse)
async def query(request: QueryRequest):
    answer = query_rag(request.query, request.temp, request.top_p)
    return {"query": request.query, "answer": answer}

@app.post("/refresh-memory/", response_model=RefreshResponse)
async def refresh_memory():
    global vector_store
    from rag_engine import vector_store, vector_store_lock
    with vector_store_lock:
        vector_store = None
        if os.path.exists("chroma_db"):
            import shutil
            shutil.rmtree("chroma_db")
        os.makedirs("chroma_db", exist_ok=True)
    return {"status": "Memory refreshed"}

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    raise HTTPException(status_code=501, detail="Audio support not yet implemented")

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing StudentRAG-V3...")
    try:
        update_vector_store()
        logger.info("Vector store initialized.")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting application...")
    uvicorn.run(app, host="0.0.0.0", port=8000)