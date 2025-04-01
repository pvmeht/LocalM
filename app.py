from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from qa_rag import QARAG
from image_recognition import ImageRecognizer
from fastapi import HTTPException
import os

app = FastAPI(
    title="Simple AI Application",
    description="APIs for QA, RAG, and Image Recognition",
    version="1.0.0"
)

# Create uploads directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Initialize components
qa_rag = QARAG()
image_recognizer = ImageRecognizer()

class QueryRequest(BaseModel):
    question: str

@app.post("/qa")
async def qa_endpoint(query: QueryRequest):
    """Answer a question without context."""
    answer = qa_rag.run_qa(query.question)
    return {"question": query.question, "answer": answer}

@app.post("/rag")
async def rag_endpoint(query: QueryRequest):
    """Answer a question with retrieved context."""
    result = qa_rag.run_rag(query.question)
    return {"question": query.question, "result": result}

@app.post("/upload-rag-file")
async def upload_rag_file(file: UploadFile = File(...)):
    """Upload a text file for RAG."""
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    try:
        qa_rag.add_document(file_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": f"File '{file.filename}' uploaded and processed for RAG"}

@app.delete("/clear-rag-data")
async def clear_rag_data():
    """Clear all RAG data."""
    qa_rag.clear_rag_data()
    return {"message": "RAG data cleared"}

@app.post("/classify-image")
async def classify_image_endpoint(file: UploadFile = File(...)):
    """Classify an uploaded image."""
    image_data = await file.read()
    result = image_recognizer.classify(image_data)
    return {"filename": file.filename, "prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)