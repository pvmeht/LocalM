from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import shutil
import os
from models.qa import answer_question
from models.rag import rag_answer
from models.image_rec import recognize_image
from models.image_gen import generate_image
from utils.file_processing import process_file
from utils.data_cleaning import clean_data

app = FastAPI()

# Security Setup
SECRET_KEY = "your_secret_key_here"  # Replace with a secure key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dummy user database
users_db = {
    "user1": {"username": "user1", "hashed_password": pwd_context.hash("password1")}
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str):
    return users_db.get(username)

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user["username"]}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username or not get_user(username):
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Endpoints
@app.post("/qa", dependencies=[Depends(get_current_user)])
def qa_endpoint(question: str, context: str):
    return {"answer": answer_question(question, context)}

@app.post("/rag", dependencies=[Depends(get_current_user)])
def rag_endpoint(question: str):
    return {"answer": rag_answer(question)}

@app.post("/upload", dependencies=[Depends(get_current_user)])
async def upload_file(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return process_file(file_path)

@app.post("/clean_data", dependencies=[Depends(get_current_user)])
async def clean_data_endpoint(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return clean_data(file_path)

@app.post("/train_new_model", dependencies=[Depends(get_current_user)])
async def train_model(file: UploadFile = File(...)):
    # Placeholder: Add real training logic (e.g., scikit-learn)
    return {"status": "Training not implemented yet"}

@app.post("/image_recognition", dependencies=[Depends(get_current_user)])
async def image_recognition_endpoint(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    label = recognize_image(file_path)
    return {"label": label}

@app.post("/image_generation", dependencies=[Depends(get_current_user)])
def image_generation_endpoint(prompt: str):
    image = generate_image(prompt)
    image.save("generated_image.png")
    return {"message": "Image saved as generated_image.png"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)