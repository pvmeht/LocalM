

# LocalM - Student AI Assistant

**LocalM** is a local AI-powered application designed to assist students by enabling them to upload notes (PDF, DOCX, TXT) and images, then query the content or generate topic-specific image descriptions. Built with FastAPI, it leverages Llama-2-7B for text generation and BLIP for image captioning, with a Retrieval-Augmented Generation (RAG) pipeline for answering questions based on uploaded documents.

## Features
- **Document-Based Q&A**: Upload notes and ask questions answered directly from the content.
- **Image Description**: Upload images with a topic (e.g., "biology diagram") to get refined descriptions.
- **Future-Ready**: Placeholder for audio recognition (e.g., lecture recordings).
- **Customizable Responses**: Control creativity with `temp` and `top_p` parameters for text generation.

## Project Structure
```
F:\Codes\LocalM\
├── app.py                # Main FastAPI app
├── models.py             # Model initialization (Llama-2, BLIP)
├── rag_engine.py         # RAG pipeline and vector store logic
├── utils.py              # Helper functions (file loading, image processing)
├── requirements.txt      # Dependencies
├── README.md             # This file
├── docs/                 # Student notes (auto-created)
├── images/               # Uploaded images (auto-created)
├── chroma_db/            # Persistent vector store (auto-created)
└── models/               # Model files
    └── llama-2-7b-chat.Q4_K_M.gguf  # Llama-2 model (downloaded)
```

## Prerequisites
- **Python**: 3.12.6 (recommended, use `pyenv` to set locally)
- **System Requirements**: 
  - Minimum 8GB RAM (16GB+ recommended for BLIP)
  - Optional: NVIDIA GPU with CUDA for faster processing
- **Internet**: Required initially for downloading models and dependencies

## Setup Instructions

### 1. Clone the Repository
Clone or create the project directory:
```bash
mkdir F:\Codes\LocalM
cd F:\Codes\LocalM
```
(If using Git, replace with `git clone <repo-url>`.)

### 2. Set Up Python Environment
Set the Python version (optional with `pyenv`):
```bash
pyenv local 3.12.6
python --version  # Should show 3.12.6
```

Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
Create a `requirements.txt` file with the following content:
```
fastapi
uvicorn
langchain
langchain-community
langchain-chroma
chromadb
ctransformers
torch
transformers
PyPDF2
python-docx
pillow
pydantic
huggingface-hub
sentence-transformers
python-multipart
llama-cpp-python
langchain-huggingface
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 4. Download the Llama-2 Model
Download the Llama-2-7B GGUF model and place it in the `models/` directory:
```bash
mkdir models
cd models
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir .
cd ..
```

### 5. Run the Application
Start the FastAPI server with auto-reload for development:
```bash
uvicorn app:app --reload
```

The app will be available at `http://127.0.0.1:8000`. You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
Models loaded successfully.
INFO:     Application startup complete.
```

## Usage

### API Endpoints
Use tools like Postman, curl, or the interactive Swagger UI (`http://127.0.0.1:8000/docs`) to interact with the API.

#### 1. Upload Notes
- **Endpoint**: `POST /upload-notes/`
- **Request**: 
  ```
  Content-Type: multipart/form-data
  file: <select a PDF, DOCX, or TXT file>
  ```
- **Response**: 
  ```json
  {"message": "Note uploaded successfully"}
  ```

#### 2. Ask a Question
- **Endpoint**: `POST /ask-question/`
- **Request**: 
  ```json
  {
    "question": "What is the main topic of the uploaded notes?",
    "temp": 0.7,
    "top_p": 0.9
  }
  ```
- **Response**: 
  ```json
  {"answer": "The main topic is..."}
  ```

#### 3. Upload an Image
- **Endpoint**: `POST /upload-image/`
- **Request**: 
  ```
  Content-Type: multipart/form-data
  file: <select an image file>
  topic: "biology diagram"
  ```
- **Response**: 
  ```json
  {"description": "This image shows a biology diagram depicting..."}
  ```

#### 4. Upload Audio (Placeholder)
- **Endpoint**: `POST /upload-audio/`
- **Response**: 
  ```json
  {"message": "Audio upload not implemented yet. Planned for future updates."}
  ```

## Performance Tips
- **Speed Up Q&A**: If `/ask-question/` is slow, increase `n_batch` in `models.py` (e.g., to 1024) or use a GPU with `n_gpu_layers=10`.
- **Memory Issues**: If BLIP loading hangs, switch to a smaller model (e.g., `Salesforce/blip-image-captioning-base` already in use) or disable it temporarily:
  ```python
  # In models.py
  def load_blip2():
      return None, None
  ```

## Future Improvements
- **Audio Support**: Implement `/upload-audio/` with `speechrecognition` or `openai-whisper`.
- **User Authentication**: Add multi-user support with a login system.
- **Advanced Retrieval**: Use `RetrievalQA` from LangChain for better context handling.
- **Smaller Models**: Test lighter LLMs (e.g., 13B GGUF) for faster responses.

## Troubleshooting
- **Deprecation Warnings**: Ensure all LangChain imports use the latest packages (e.g., `langchain-huggingface`, `langchain-chroma`).
- **Model Loading Errors**: Verify `llama-2-7b-chat.Q4_K_M.gguf` is in `models/`. Re-download if missing.
- **Token Limit Issues**: If "exceeded maximum context length" appears, adjust `max_context_tokens` in `rag_engine.py`.

## License
This project is for educational purposes and uses open-source models under their respective licenses (e.g., Llama-2, BLIP).

---

### Notes
- This README reflects the current setup with `LlamaCpp` for Llama-2, `Salesforce/blip-image-captioning-base` for BLIP, and the latest LangChain packages (`langchain-huggingface`, `langchain-chroma`).
- Adjust paths (e.g., `F:\Codes\LocalM`) if your directory differs.
- Add a `LICENSE` file if you plan to distribute this publicly.

Save this as `README.md` in `F:\Codes\LocalM\`, and it’ll serve as a comprehensive guide for your project! Let me know if you’d like to tweak anything further.