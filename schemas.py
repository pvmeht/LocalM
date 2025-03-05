from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    query: str
    temp: float = 0.7
    top_p: float = 0.9

class QueryResponse(BaseModel):
    query: str
    answer: str

class NotesUploadResponse(BaseModel):
    filename: str
    status: str

class ImageUploadRequest(BaseModel):
    topic: Optional[str] = None

class ImageUploadResponse(BaseModel):
    filename: str
    topic: Optional[str]
    description: str
    status: str

class RefreshResponse(BaseModel):
    status: str