# from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from passlib.context import CryptContext
# from jose import JWTError, jwt
# from datetime import datetime, timedelta
# import shutil
# import os
# from models.qa import answer_question
# from models.rag import rag_answer
# from models.image_rec import recognize_image
# from models.image_gen import generate_image
# from utils.file_processing import process_file
# from utils.data_cleaning import clean_data

# app = FastAPI()

# # Security Setup
# SECRET_KEY = "your_secret_key_here"  # Replace with a secure key
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 30

# # Use pbkdf2_sha256 to avoid requiring native bcrypt backend during startup
# pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# # Dummy user database
# # Create a simple user entry (hash computed with the selected scheme)
# users_db = {
#     "user1": {"username": "user1", "hashed_password": pwd_context.hash("password1")}
# }

# def verify_password(plain_password, hashed_password):
#     return pwd_context.verify(plain_password, hashed_password)

# def get_user(username: str):
#     return users_db.get(username)

# def authenticate_user(username: str, password: str):
#     user = get_user(username)
#     if not user or not verify_password(password, user["hashed_password"]):
#         return False
#     return user

# def create_access_token(data: dict, expires_delta: timedelta = None):
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# @app.post("/token")
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
#     user = authenticate_user(form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     token = create_access_token({"sub": user["username"]}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
#     return {"access_token": token, "token_type": "bearer"}

# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username = payload.get("sub")
#         if not username or not get_user(username):
#             raise HTTPException(status_code=401, detail="Invalid token")
#         return username
#     except JWTError:
#         raise HTTPException(status_code=401, detail="Invalid token")

# # Endpoints
# @app.post("/qa", dependencies=[Depends(get_current_user)])
# def qa_endpoint(question: str, context: str):
#     return {"answer": answer_question(question, context)}

# @app.post("/rag", dependencies=[Depends(get_current_user)])
# def rag_endpoint(question: str):
#     return {"answer": rag_answer(question)}

# @app.post("/upload", dependencies=[Depends(get_current_user)])
# async def upload_file(file: UploadFile = File(...)):
#     os.makedirs("uploads", exist_ok=True)
#     file_path = f"uploads/{file.filename}"
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     return process_file(file_path)

# @app.post("/clean_data", dependencies=[Depends(get_current_user)])
# async def clean_data_endpoint(file: UploadFile = File(...)):
#     file_path = f"uploads/{file.filename}"
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     return clean_data(file_path)

# @app.post("/train_new_model", dependencies=[Depends(get_current_user)])
# async def train_model(file: UploadFile = File(...)):
#     # Placeholder: Add real training logic (e.g., scikit-learn)
#     return {"status": "Training not implemented yet"}

# @app.post("/image_recognition", dependencies=[Depends(get_current_user)])
# async def image_recognition_endpoint(file: UploadFile = File(...)):
#     file_path = f"uploads/{file.filename}"
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     label = recognize_image(file_path)
#     return {"label": label}

# @app.post("/image_generation", dependencies=[Depends(get_current_user)])
# def image_generation_endpoint(prompt: str):
#     image = generate_image(prompt)
#     image.save("generated_image.png")
#     return {"message": "Image saved as generated_image.png"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




# from fastapi import FastAPI, UploadFile, File, HTTPException, Query
# import shutil
# import os
# import logging
# import base64
# from io import BytesIO
# from typing import Optional
# import faiss
# import numpy as np
# import torch
# from sentence_transformers import SentenceTransformer
# from models.qa import answer_question
# from models.image_rec import recognize_image
# from models.image_gen import generate_image
# from utils.file_processing import process_file
# from utils.data_cleaning import clean_data

# app = FastAPI()

# # Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # In-memory user data for RAG isolation (simplified without auth)
# user_indices = {}  # user_id -> FAISS index
# user_documents = {}  # user_id -> list of documents/chunks
# encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")  # For document embeddings

# # Default user for testing (since auth removed)
# DEFAULT_USER = "user1"

# # Updated rag_answer function (local, no import from rag.py to avoid conflicts)
# def rag_answer(user_id: str, question: str) -> str:
#     if user_id not in user_indices or not user_documents.get(user_id):
#         return f"No documents uploaded for user '{user_id}'. Please upload a file first."
    
#     q_embedding = encoder.encode([question])  # Shape: [1, 384]
    
#     index = user_indices[user_id]
#     documents = user_documents[user_id]
    
#     if q_embedding.shape[1] != index.d:
#         logger.warning(f"Dimension mismatch: query={q_embedding.shape[1]}, index={index.d}")
#         return "Embedding dimension mismatch—re-upload documents."
    
#     distances, indices = index.search(q_embedding, k=1)
#     if indices.shape[1] == 0 or indices[0][0] >= len(documents):
#         return "No relevant context found in uploaded documents."
    
#     context = documents[indices[0][0]]
#     logger.info(f"Retrieved context: {context[:100]}...")  # Debug: Log retrieved chunk
#     return answer_question(question, context)

# # Endpoints
# @app.post("/qa")
# async def qa_endpoint(
#     question: str = Query(..., description="Question to answer"),
#     context: str = Query(..., description="Context for QA")
# ):
#     try:
#         answer = answer_question(question, context)
#         logger.info(f"QA: {question[:50]}...")
#         return {"answer": answer}
#     except Exception as e:
#         logger.error(f"QA error: {e}")
#         raise HTTPException(status_code=500, detail="QA processing failed")

# @app.post("/rag")
# async def rag_endpoint(
#     question: str = Query(..., description="Question for RAG")
# ):
#     try:
#         # Use default user for isolation
#         answer = rag_answer(DEFAULT_USER, question)
#         logger.info(f"RAG: {question[:50]}...")
#         return {"answer": answer}
#     except Exception as e:
#         logger.error(f"RAG error: {e}")
#         raise HTTPException(status_code=500, detail="RAG processing failed")

# @app.post("/upload")
# async def upload_file(
#     file: UploadFile = File(..., description="File to upload")
# ):
#     global user_indices, user_documents  # Ensure access
#     try:
#         os.makedirs("uploads", exist_ok=True)
#         file_path = f"uploads/{DEFAULT_USER}_{file.filename}"  # Default user-specific
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         logger.info(f"Processing file: {file_path}")
        
#         # Process file to get chunks (now returns list[str] directly)
#         chunks = process_file(file_path)
#         logger.info(f"Extracted {len(chunks)} chunks from {file.filename}")
        
#         if isinstance(chunks, list) and chunks:
#             # Create/reset embeddings and index
#             embeddings = encoder.encode(chunks)
#             dim = embeddings.shape[1]
#             if DEFAULT_USER not in user_indices:
#                 user_indices[DEFAULT_USER] = faiss.IndexFlatIP(dim)
#             else:
#                 user_indices[DEFAULT_USER].reset()  # Reset for re-upload
            
#             user_indices[DEFAULT_USER].add(embeddings)
#             user_documents[DEFAULT_USER] = chunks
#             logger.info(f"Indexed {len(chunks)} chunks for user {DEFAULT_USER}. Dim: {dim}")
#         else:
#             logger.warning(f"No chunks from {file.filename}: {type(chunks)}")
#             chunks = []  # Ensure defined
        
#         # Cleanup
#         if os.path.exists(file_path):
#             os.remove(file_path)
        
#         return {"message": "Upload successful", "chunks_processed": len(chunks)}
#     except Exception as e:
#         logger.error(f"Upload error: {e}")
#         if 'file_path' in locals() and os.path.exists(file_path):
#             os.remove(file_path)  # Cleanup on error
#         raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# # ... (rest of endpoints unchanged: /clean_data, /train_new_model, /image_recognition, /image_generation, root)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)







# from fastapi import FastAPI, UploadFile, File, HTTPException, Query
# import shutil
# import os
# import logging
# import base64
# from io import BytesIO
# from typing import Optional
# import faiss
# import numpy as np
# import torch
# from sentence_transformers import SentenceTransformer
# from models.qa import answer_question
# from models.rag import rag_answer
# from models.image_rec import recognize_image
# from models.image_gen import generate_image
# from utils.file_processing import process_file
# from utils.data_cleaning import clean_data

# app = FastAPI()

# # Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # In-memory user data for RAG isolation (simplified without auth)
# user_indices = {}  # user_id -> FAISS index
# user_documents = {}  # user_id -> list of documents/chunks
# encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")  # For document embeddings

# # Default user for testing (since auth removed)
# DEFAULT_USER = "user1"

# # Updated rag_answer function (local, no import from rag.py to avoid conflicts)
# def rag_answer(user_id: str, question: str) -> str:
#     if user_id not in user_indices or not user_documents.get(user_id):
#         return f"No documents uploaded for user '{user_id}'. Please upload a file first."
    
#     q_embedding = encoder.encode([question])  # Shape: [1, 384]
    
#     index = user_indices[user_id]
#     documents = user_documents[user_id]
    
#     if q_embedding.shape[1] != index.d:
#         logger.warning(f"Dimension mismatch: query={q_embedding.shape[1]}, index={index.d}")
#         return "Embedding dimension mismatch—re-upload documents."
    
#     distances, indices = index.search(q_embedding, k=1)
#     if indices.shape[1] == 0 or indices[0][0] >= len(documents):
#         return "No relevant context found in uploaded documents."
    
#     context = documents[indices[0][0]]
#     logger.info(f"Retrieved context: {context[:100]}...")  # Debug: Log retrieved chunk
#     return answer_question(question, context)

# # Endpoints
# @app.get("/")
# def root():
#     return {"message": "AI App is running!"}

# @app.post("/qa")
# async def qa_endpoint(
#     question: str = Query(..., description="Question to answer"),
#     context: str = Query(..., description="Context for QA")
# ):
#     try:
#         answer = answer_question(question, context)
#         logger.info(f"QA: {question[:50]}...")
#         return {"answer": answer}
#     except Exception as e:
#         logger.error(f"QA error: {e}")
#         raise HTTPException(status_code=500, detail="QA processing failed")

# @app.post("/rag")
# async def rag_endpoint(
#     question: str = Query(..., description="Question for RAG")
# ):
#     try:
#         # Use default user for isolation
#         answer = rag_answer(DEFAULT_USER, question)
#         logger.info(f"RAG: {question[:50]}...")
#         return {"answer": answer}
#     except Exception as e:
#         logger.error(f"RAG error: {e}")
#         raise HTTPException(status_code=500, detail="RAG processing failed")

# @app.post("/upload")
# async def upload_file(
#     file: UploadFile = File(..., description="File to upload")
# ):
#     global user_indices, user_documents  # Ensure access
#     try:
#         os.makedirs("uploads", exist_ok=True)
#         file_path = f"uploads/{DEFAULT_USER}_{file.filename}"  # Default user-specific
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         logger.info(f"Processing file: {file_path}")
        
#         # Process file to get chunks (now returns list[str] directly)
#         chunks = process_file(file_path)
#         logger.info(f"Extracted {len(chunks)} chunks from {file.filename}")
        
#         if isinstance(chunks, list) and chunks:
#             # Create/reset embeddings and index
#             embeddings = encoder.encode(chunks)
#             dim = embeddings.shape[1]
#             if DEFAULT_USER not in user_indices:
#                 user_indices[DEFAULT_USER] = faiss.IndexFlatIP(dim)
#             else:
#                 user_indices[DEFAULT_USER].reset()  # Reset for re-upload
            
#             user_indices[DEFAULT_USER].add(embeddings)
#             user_documents[DEFAULT_USER] = chunks
#             logger.info(f"Indexed {len(chunks)} chunks for user {DEFAULT_USER}. Dim: {dim}")
#         else:
#             logger.warning(f"No chunks from {file.filename}: {type(chunks)}")
#             chunks = []  # Ensure defined
        
#         # Cleanup
#         if os.path.exists(file_path):
#             os.remove(file_path)
        
#         return {"message": "Upload successful", "chunks_processed": len(chunks)}
#     except Exception as e:
#         logger.error(f"Upload error: {e}")
#         if 'file_path' in locals() and os.path.exists(file_path):
#             os.remove(file_path)  # Cleanup on error
#         raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# @app.post("/clean_data")
# async def clean_data_endpoint(
#     file: UploadFile = File(..., description="File to clean")
# ):
#     try:
#         file_path = f"uploads/{DEFAULT_USER}_{file.filename}"
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         result = clean_data(file_path)
#         os.remove(file_path)  # Cleanup
#         logger.info(f"Data cleaning: {file.filename}")
#         return result
#     except Exception as e:
#         logger.error(f"Clean data error: {e}")
#         raise HTTPException(status_code=500, detail="Data cleaning failed")

# @app.post("/train_new_model")
# async def train_model(
#     file: UploadFile = File(..., description="Dataset for training")
# ):
#     try:
#         file_path = f"uploads/{DEFAULT_USER}_{file.filename}"
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # Simple placeholder with scikit-learn (install: pip install scikit-learn)
#         from sklearn.feature_extraction.text import TfidfVectorizer
#         from sklearn.naive_bayes import MultinomialNB
#         # Example: Load text data, train classifier (adapt to your needs)
#         with open(file_path, 'r') as f:
#             texts = f.readlines()
#         vectorizer = TfidfVectorizer()
#         X = vectorizer.fit_transform(texts[:10])  # Sample small data
#         y = [1] * X.shape[0]  # Dummy labels
#         model = MultinomialNB().fit(X, y)
        
#         os.remove(file_path)
#         logger.info(f"Model training: {file.filename}")
#         return {"status": "Model trained (placeholder)", "model_type": "Naive Bayes"}
#     except Exception as e:
#         logger.error(f"Training error: {e}")
#         raise HTTPException(status_code=500, detail="Training failed")

# @app.post("/image_recognition")
# async def image_recognition_endpoint(
#     file: UploadFile = File(..., description="Image to recognize")
# ):
#     try:
#         os.makedirs("uploads", exist_ok=True)
#         file_path = f"uploads/{DEFAULT_USER}_{file.filename}"
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         result = recognize_image(file_path)  # Now returns dict
#         os.remove(file_path)
#         logger.info(f"Image rec: {result['label']} (confidence: {result.get('confidence', 'N/A')})")
#         return result  # {"label": "...", "confidence": 0.95}
#     except Exception as e:
#         logger.error(f"Image rec error: {e}")
#         raise HTTPException(status_code=500, detail="Image recognition failed")

# @app.post("/image_generation")
# async def image_generation_endpoint(
#     prompt: str = Query(..., description="Prompt for image generation"),
#     num_inference_steps: Optional[int] = Query(20, description="Steps for generation (10-50)")
# ):
#     try:
#         image = generate_image(prompt, num_inference_steps)
#         # Return base64 instead of saving
#         buffered = BytesIO()
#         image.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#         logger.info(f"Image gen: {prompt[:50]}...")
#         return {"image_b64": f"data:image/png;base64,{img_str}"}
#     except Exception as e:
#         logger.error(f"Image gen error: {e}")
#         raise HTTPException(status_code=500, detail="Image generation failed")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




# from fastapi import FastAPI, UploadFile, File, HTTPException, Query
# import os
# import logging
# import base64
# from io import BytesIO
# from typing import Optional
# import chromadb
# from chromadb.utils import embedding_functions
# from utils.file_processing import process_file  # Returns list[str] chunks
# from models.qa import answer_question
# from models.image_rec import recognize_image
# from models.image_gen import generate_image

# app = FastAPI()

# # Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Persistent storage root
# STORAGE_ROOT = "storage"
# os.makedirs(STORAGE_ROOT, exist_ok=True)

# # Embedding function (same as before)
# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="multi-qa-MiniLM-L6-cos-v1"
# )

# # In-memory cache of Chroma clients/collections per user
# user_clients = {}   # user_id -> chromadb.Client
# user_collections = {}  # user_id -> chromadb.Collection

# def get_user_collection(user_id: str):
#     """Get or create persistent Chroma collection for a user."""
#     if user_id in user_collections:
#         return user_collections[user_id]

#     user_path = os.path.join(STORAGE_ROOT, user_id)
#     os.makedirs(user_path, exist_ok=True)

#     client = chromadb.PersistentClient(path=user_path)
#     collection = client.get_or_create_collection(
#         name="documents",
#         embedding_function=embedding_function
#     )

#     user_clients[user_id] = client
#     user_collections[user_id] = collection
#     return collection

# def rag_answer(user_id: str, question: str) -> str:
#     collection = get_user_collection(user_id)
    
#     if collection.count() == 0:
#         return f"No documents uploaded for user '{user_id}'. Please upload files first."

#     try:
#         results = collection.query(
#             query_texts=[question],
#             n_results=3,  # Top 3 relevant chunks
#             include=["documents", "distances", "metadatas"]
#         )

#         contexts = results["documents"][0]
#         if not contexts:
#             return "No relevant information found in your documents."

#         combined_context = "\n\n".join(contexts)
#         logger.info(f"Retrieved {len(contexts)} chunks for user {user_id}")

#         return answer_question(question, combined_context)

#     except Exception as e:
#         logger.error(f"RAG query error for {user_id}: {e}")
#         return "Error processing your question. Please try again."

# # Endpoints
# @app.get("/")
# def root():
#     return {"message": "LocalM AI App is running! Multi-user persistent storage enabled."}

# @app.post("/upload")
# async def upload_file(
#     file: UploadFile = File(..., description="Upload document (PDF, DOCX, TXT, etc.)"),
#     user_id: str = Query("user1", description="User ID (e.g., user1, user2)")
# ):
#     try:
#         # Temporary save
#         os.makedirs("uploads", exist_ok=True)
#         temp_path = f"uploads/{user_id}_{file.filename}"
#         with open(temp_path, "wb") as buffer:
#             content = await file.read()
#             buffer.write(content)

#         # Extract chunks
#         chunks = process_file(temp_path)
#         logger.info(f"Extracted {len(chunks)} chunks from {file.filename} for {user_id}")

#         if not chunks:
#             raise ValueError("No text extracted from file.")

#         # Get user collection and add chunks
#         collection = get_user_collection(user_id)
#         ids = [f"{user_id}_chunk_{i}" for i in range(len(chunks))]
#         metadatas = [{"source": file.filename, "chunk_index": i} for i in range(len(chunks))]

#         collection.add(
#             documents=chunks,
#             ids=ids,
#             metadatas=metadatas
#         )

#         logger.info(f"Stored {len(chunks)} chunks in Chroma for user {user_id}")

#         # Cleanup temp file
#         os.remove(temp_path)

#         return {
#             "message": "File uploaded and indexed successfully",
#             "user_id": user_id,
#             "filename": file.filename,
#             "chunks_stored": len(chunks),
#             "total_documents": collection.count()
#         }

#     except Exception as e:
#         logger.error(f"Upload error for {user_id}: {e}")
#         if 'temp_path' in locals() and os.path.exists(temp_path):
#             os.remove(temp_path)
#         raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# @app.post("/rag")
# async def rag_endpoint(
#     question: str = Query(..., description="Your question"),
#     user_id: str = Query("user1", description="User ID")
# ):
#     try:
#         answer = rag_answer(user_id, question)
#         logger.info(f"RAG query for {user_id}: {question[:50]}...")
#         return {"answer": answer, "user_id": user_id}
#     except Exception as e:
#         logger.error(f"RAG error: {e}")
#         raise HTTPException(status_code=500, detail="RAG query failed")

# @app.post("/list_users")
# async def list_users():
#     """Debug: List all users with stored data"""
#     users = [d for d in os.listdir(STORAGE_ROOT) if os.path.isdir(os.path.join(STORAGE_ROOT, d))]
#     return {"stored_users": users}

# @app.post("/clear_user")
# async def clear_user(user_id: str = Query(..., description="User ID to clear")):
#     """Debug: Clear all data for a user"""
#     user_path = os.path.join(STORAGE_ROOT, user_id)
#     if os.path.exists(user_path):
#         import shutil
#         shutil.rmtree(user_path)
#         if user_id in user_collections:
#             del user_collections[user_id]
#         if user_id in user_clients:
#             del user_clients[user_id]
#         return {"message": f"Cleared data for {user_id}"}
#     return {"message": f"No data found for {user_id}"}

# # Keep your existing image endpoints unchanged
# @app.post("/image_recognition")
# async def image_recognition_endpoint(
#     file: UploadFile = File(..., description="Image to recognize"),
#     user_id: str = Query("user1", description="Optional user ID")
# ):
#     try:
#         os.makedirs("uploads", exist_ok=True)
#         file_path = f"uploads/{user_id}_{file.filename}"
#         with open(file_path, "wb") as buffer:
#             content = await file.read()
#             buffer.write(content)
        
#         result = recognize_image(file_path)
#         os.remove(file_path)
#         logger.info(f"Image recognition for {user_id}: {result}")
#         return result
#     except Exception as e:
#         logger.error(f"Image rec error: {e}")
#         raise HTTPException(status_code=500, detail="Image recognition failed")

# @app.post("/image_generation")
# async def image_generation_endpoint(
#     prompt: str = Query(..., description="Prompt for image generation"),
#     num_inference_steps: Optional[int] = Query(20, description="Steps (10-50)"),
#     user_id: str = Query("user1", description="Optional user ID")
# ):
#     try:
#         image = generate_image(prompt, num_inference_steps)
#         buffered = BytesIO()
#         image.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#         logger.info(f"Image generated for {user_id}: {prompt[:50]}...")
#         return {"image_b64": f"data:image/png;base64,{img_str}"}
#     except Exception as e:
#         logger.error(f"Image gen error: {e}")
#         raise HTTPException(status_code=500, detail="Image generation failed")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)






from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import shutil
import os
import logging
import base64
from io import BytesIO
from typing import Literal, Optional
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from models.qa import answer_question
from models.rag import rag_answer  # Optional; overridden locally
from models.image_rec import recognize_image
from models.image_gen import generate_image
from utils.file_processing import process_file
from utils.data_cleaning import clean_data
import pickle  # Add at top
import re

# Create data directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Default user for testing (since auth removed)
DEFAULT_USER = "user1"

# File paths for persistence
INDEX_FILE = os.path.join(DATA_DIR, f"{DEFAULT_USER}_index.faiss")
DOCS_FILE = os.path.join(DATA_DIR, f"{DEFAULT_USER}_docs.pkl")
app = FastAPI()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory user data for RAG isolation (simplified without auth)
user_indices = {}  # user_id -> FAISS index
user_documents = {}  # user_id -> list of documents/chunks
# encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")  # For document embeddings
encoder = SentenceTransformer("all-mpnet-base-v2")  # Better quality, 768 dim

# Load persisted data on startup
if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
    user_indices[DEFAULT_USER] = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, "rb") as f:
        user_documents[DEFAULT_USER] = pickle.load(f)
    logger.info(f"Loaded persisted index and {len(user_documents[DEFAULT_USER])} documents for {DEFAULT_USER}")


# Updated rag_answer function (local, no import from rag.py to avoid conflicts)
def rag_answer(user_id: str, question: str) -> str:
    if user_id not in user_indices or not user_documents.get(user_id):
        return f"No documents uploaded for user '{user_id}'. Please upload a file first."
    
    q_embedding = encoder.encode([question], normalize_embeddings=True)  # Normalize for cosine
    q_embedding = np.array(q_embedding, dtype=np.float32)  # Ensure float32
    
    index = user_indices[user_id]
    documents = user_documents[user_id]
    
    if q_embedding.shape[1] != index.d:
        logger.warning(f"Dimension mismatch: query={q_embedding.shape[1]}, index={index.d}")
        return "Embedding dimension mismatch—re-upload documents."
    
    distances, indices = index.search(q_embedding, k=3)  # Increase k for better recall
    logger.info(f"Search distances: {distances[0]}, indices: {indices[0]}")  # Debug retrieval quality
    
    if indices.shape[1] == 0 or indices[0][0] == -1:  # FAISS returns -1 for no results
        return "No relevant context found in uploaded documents."
    
    # Concatenate top contexts (improves "power")
    relevant_indices = [i for i in indices[0] if i < len(documents)]
    if not relevant_indices:
        return "No relevant context found."
    
    context = "\n\n".join(documents[i] for i in relevant_indices)
    logger.info(f"Retrieved context: {context[:100]}...")  # Debug: Log retrieved chunk
    return answer_question(question, context)

# Endpoints
@app.get("/")
def root():
    return {"message": "AI App is running!"}

@app.post("/qa")
async def qa_endpoint(
    question: str = Query(..., description="Question to answer"),
    context: str = Query(..., description="Context for QA")
):
    try:
        answer = answer_question(question, context)
        logger.info(f"QA: {question[:50]}...")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"QA error: {e}")
        raise HTTPException(status_code=500, detail="QA processing failed")

# @app.post("/rag")
# async def rag_endpoint(
#     question: str = Query(..., description="Question for RAG")
# ):
#     try:
#         # Use default user for isolation
#         answer = rag_answer(DEFAULT_USER, question)
#         logger.info(f"RAG: {question[:50]}...")
#         return {"answer": answer}
#     except Exception as e:
#         logger.error(f"RAG error: {e}")
#         raise HTTPException(status_code=500, detail="RAG processing failed")




@app.post("/rag")
async def rag_endpoint(
    question: str = Query(..., description="Question for RAG"),
    mode: Optional[Literal["generate", "extract", "hybrid", "auto"]] = Query(
        "hybrid",  # NEW DEFAULT: Best of both worlds!
        description="Mode: 'generate' (AI only), 'extract' (raw text), 'hybrid' (extract → AI summarize), 'auto' (old intent detection)"
    ),
    k: int = Query(5, ge=1, le=10, description="Number of chunks to retrieve (increase for better coverage)"),
    raw: bool = Query(False, description="Return raw context for debugging")
):
    try:
        if DEFAULT_USER not in user_indices or not user_documents.get(DEFAULT_USER):
            raise HTTPException(status_code=404, detail="No document uploaded yet. Please use /upload first.")

        # === Retrieval: Use higher k for better coverage ===
        q_embedding = encoder.encode([question], normalize_embeddings=True)
        q_embedding = np.array(q_embedding, dtype=np.float32)

        index = user_indices[DEFAULT_USER]
        documents = user_documents[DEFAULT_USER]

        distances, indices = index.search(q_embedding, k=k)
        logger.info(f"Search distances: {distances[0].tolist()}, indices: {indices[0].tolist()}")

        relevant_chunks = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(documents) and idx != -1:
                relevant_chunks.append((dist, documents[idx]))
        relevant_chunks.sort(key=lambda x: x[0], reverse=True)

        full_context = "\n\n".join(chunk for _, chunk in relevant_chunks)
        logger.info(f"Retrieved {len(relevant_chunks)} chunks, top similarity: {distances[0][0]:.3f}")

        if raw:
            return {
                "mode": "raw",
                "question": question,
                "retrieved_context": full_context,
                "num_chunks": len(relevant_chunks),
                "similarities": [float(d) for d, _ in relevant_chunks]
            }

        # === EXTRACT MODE: Raw limited text ===
        if mode == "extract":
            extracted = full_context.strip()
            if len(extracted) > 700:
                # Cut mid-word for "incomplete" feel
                extracted = extracted[:700].rsplit(' ', 1)[0] + "..."
            return {"mode": "extract", "answer": extracted}

        # === HYBRID MODE: Extract → Feed to AI for clean answer ===
        elif mode == "hybrid" or mode == "auto":
            # Step 1: Get clean extracted text (full, not limited)
            extracted_context = full_context.strip()

            # Step 2: Ask AI to generate structured answer using ONLY this real text
            structured_prompt = (
                "Using ONLY the following text from the resume, answer the question clearly and accurately. "
                "Extract and format the relevant information. Do not add anything not present.\n\n"
                f"Resume Text:\n{extracted_context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

            hybrid_answer = answer_question(structured_prompt, extracted_context)  # Reuse your qa.py function

            return {
                "mode": "hybrid",
                "answer": hybrid_answer,
                "source": "extracted resume text → AI structured",
                "retrieved_chunks": len(relevant_chunks),
                "top_similarity": float(distances[0][0])
            }

        # === GENERATE MODE: Old way (direct AI on retrieved context) ===
        else:
            answer = answer_question(question, full_context)
            return {
                "mode": "generate",
                "answer": answer,
                "retrieved_chunks": len(relevant_chunks)
            }

    except Exception as e:
        logger.error(f"RAG error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")






@app.post("/upload")
async def upload_file(
    file: UploadFile = File(..., description="File to upload")
):
    global user_indices, user_documents  # Ensure access
    try:

        # Load persisted data on startup
        if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
            user_indices[DEFAULT_USER] = faiss.read_index(INDEX_FILE)
            with open(DOCS_FILE, "rb") as f:
                user_documents[DEFAULT_USER] = pickle.load(f)
            logger.info(f"Loaded persisted index and {len(user_documents[DEFAULT_USER])} documents for {DEFAULT_USER}")

        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{DEFAULT_USER}_{file.filename}"  # Default user-specific
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing file: {file_path}")
        
        # Process file to get chunks (now returns list[str] directly)
        chunks = process_file(file_path)
        logger.info(f"Extracted {len(chunks)} chunks from {file.filename}")
        
        if isinstance(chunks, list) and chunks:
            embeddings = encoder.encode(chunks, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype=np.float32)
            dim = embeddings.shape[1]
            
            # Create or reset index
            if DEFAULT_USER not in user_indices:
                user_indices[DEFAULT_USER] = faiss.IndexFlatIP(dim)
            else:
                user_indices[DEFAULT_USER].reset()
            
            user_indices[DEFAULT_USER].add(embeddings)
            user_documents[DEFAULT_USER] = chunks
            
            # === NEW: Save to disk ===
            faiss.write_index(user_indices[DEFAULT_USER], INDEX_FILE)
            with open(DOCS_FILE, "wb") as f:
                pickle.dump(chunks, f)
            
            logger.info(f"Indexed and SAVED {len(chunks)} chunks to disk for {DEFAULT_USER}")
        else:
            # Clear saved files if no chunks
            if os.path.exists(INDEX_FILE):
                os.remove(INDEX_FILE)
            if os.path.exists(DOCS_FILE):
                os.remove(DOCS_FILE)
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {"message": "Upload successful", "chunks_processed": len(chunks)}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)  # Cleanup on error
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    

@app.post("/clean_data")
async def clean_data_endpoint(
    file: UploadFile = File(..., description="File to clean")
):
    try:
        file_path = f"uploads/{DEFAULT_USER}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = clean_data(file_path)
        os.remove(file_path)  # Cleanup
        logger.info(f"Data cleaning: {file.filename}")
        return result
    except Exception as e:
        logger.error(f"Clean data error: {e}")
        raise HTTPException(status_code=500, detail="Data cleaning failed")

@app.post("/train_new_model")
async def train_model(
    file: UploadFile = File(..., description="Dataset for training")
):
    try:
        file_path = f"uploads/{DEFAULT_USER}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Simple placeholder with scikit-learn (install: pip install scikit-learn)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        # Example: Load text data, train classifier (adapt to your needs)
        with open(file_path, 'r') as f:
            texts = f.readlines()
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts[:10])  # Sample small data
        y = [1] * X.shape[0]  # Dummy labels
        model = MultinomialNB().fit(X, y)
        
        os.remove(file_path)
        logger.info(f"Model training: {file.filename}")
        return {"status": "Model trained (placeholder)", "model_type": "Naive Bayes"}
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail="Training failed")

@app.post("/image_recognition")
async def image_recognition_endpoint(
    file: UploadFile = File(..., description="Image to recognize")
):
    try:
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{DEFAULT_USER}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = recognize_image(file_path)  # Now returns dict
        os.remove(file_path)
        logger.info(f"Image rec: {result['label']} (confidence: {result.get('confidence', 'N/A')})")
        return result  # {"label": "...", "confidence": 0.95}
    except Exception as e:
        logger.error(f"Image rec error: {e}")
        raise HTTPException(status_code=500, detail="Image recognition failed")

@app.post("/image_generation")
async def image_generation_endpoint(
    prompt: str = Query(..., description="Prompt for image generation"),
    num_inference_steps: Optional[int] = Query(20, description="Steps for generation (10-50)")
):
    try:
        image = generate_image(prompt, num_inference_steps)
        # Return base64 instead of saving
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        logger.info(f"Image gen: {prompt[:50]}...")
        return {"image_b64": f"data:image/png;base64,{img_str}"}
    except Exception as e:
        logger.error(f"Image gen error: {e}")
        raise HTTPException(status_code=500, detail="Image generation failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
