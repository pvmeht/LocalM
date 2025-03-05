from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import os
from datetime import datetime
from models import load_llm, load_blip2
from rag_engine import add_document_to_vector_store, query_knowledge
from utils import generate_image_description

app = FastAPI(title="LocalM - Student AI Assistant")

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    app.state.llm = load_llm()
    app.state.processor, app.state.model = load_blip2()
    print("Models loaded successfully.")

@app.post("/upload-notes/")
async def upload_notes(file: UploadFile = File(...)):
    """Upload a note file and add it to the vector store."""
    os.makedirs("docs", exist_ok=True)
    file_path = os.path.join("docs", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    metadata = {
        "filename": file.filename,
        "upload_timestamp": datetime.now().isoformat()
    }
    add_document_to_vector_store(file_path, metadata)
    return {"message": "Note uploaded successfully"}

class Question(BaseModel):
    question: str
    temp: float = 0.3  # Default temperature 0.7
    top_p: float = 0.3  # Default top-p  0.9

@app.post("/ask-question/")
async def ask_question(question: Question):
    """Answer a question based on uploaded notes with customizable temp and top_p."""
    response = query_knowledge(
        question.question,
        app.state.llm,
        k=2,
        max_context_tokens=1500
    )
    app.state.llm.temperature = question.temp  # Set temperature
    app.state.llm.top_p = question.top_p       # Set top-p
    return {"answer": response}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...), topic: str = Form(...)):
    """Upload an image and generate a topic-specific description for the current image."""
    os.makedirs("images", exist_ok=True)
    image_path = os.path.join("images", file.filename)
    with open(image_path, "wb") as f:
        f.write(await file.read())
    description = generate_image_description(
        image_path, topic, app.state.processor, app.state.model, app.state.llm
    )
    return {"description": description}

@app.post("/upload-audio/")
async def upload_audio():
    """Placeholder for future audio upload functionality."""
    return {"message": "Audio upload not implemented yet. Planned for future updates."}