# Project Overview: StudentRAG-V3
Purpose: Assist students by processing notes (PDF, DOCX, TXT), describing images with topic context, and preparing for audio input.
### Tools:
    1. LLM: Llama-2-7B-Chat (GGUF, quantized) for text generation.
    2. Image Model: BLIP-2 for image captioning.
    3. Embedding: Sentence-Transformers for text embeddings.
    4. Vector Store: ChromaDB for RAG.
    5. Framework: FastAPI with Swagger UI for API access.
    6. Hardware: Compatible with your setup (Windows 11, Intel i7-9750H, GTX 1650, 16 GB RAM, CUDA 12.1).
### Features:
* Upload and query notes.
* Upload images with optional topics for contextual descriptions.
* Future-proof audio integration.
* Project Setup


## Folder Structure

    F:\Codes\StudentRAG-V3\
    ├── app.py                # Main FastAPI app
    ├── models.py             # Model initialization (LLM, BLIP-2)
    ├── rag_engine.py         # RAG pipeline and vector store logic
    ├── utils.py              # Helper functions (file loading, etc.)
    ├── schemas.py            # Pydantic models for API
    ├── requirements.txt      # Dependencies
    ├── docs/                 # Student notes (auto-created)
    ├── images/               # Uploaded images (auto-created)
    └── chroma_db/            # Persistent vector store (auto-created)

### Prerequisites
    OS: Windows 11
    Python: 3.12.6 (via pyenv)
    Hardware: As specified (GTX 1650 with CUDA support)
    Software: CUDA Toolkit 12.1, cuDNN
#### Setup Steps
#### Create Directory and Virtual Environment:


    mkdir F:\Codes\StudentRAG-V3
    cd F:\Codes\StudentRAG-V3
    pyenv local 3.12.6
    python -m venv venv
    venv\Scripts\activate

### Install Dependencies: 
Create requirements.txt:

    fastapi
    uvicorn
    langchain
    langchain-community
    langchain-huggingface
    llama-cpp-python
    pypdf
    docx2txt
    sentence-transformers
    chromadb
    transformers
    torch
    torchvision
    pillow
    accelerate  
Install:

    pip install -r requirements.txt
For GPU support with llama-cpp-python:

    pip uninstall llama-cpp-python -y
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python-cuBLAS-wheels/ --force-reinstall
Download Models:
Llama-2-7B-Chat:

    mkdir models
    cd models
    pip install huggingface-hub
    huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir .
    cd ..


### ⚠️ To set cache to specific drive 
    Remove-Item -Path "C:\Users\Lenovo\.cache\huggingface" -Recurse -Force
    $env:HF_HOME = "F:\Codes\LocalM\models"