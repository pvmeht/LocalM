# schemas.py
from pydantic import BaseModel

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