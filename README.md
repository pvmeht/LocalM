# LocalM
Testing SLM

# Local Small Language Model with RAG Setup

This project sets up a small language model (SLM) locally on Windows 11 using `llama.cpp` and integrates it with a Retrieval-Augmented Generation (RAG) pipeline to process files (PDF, DOCX, TXT) and answer questions based on their content. If the SLM lacks information, it can optionally fall back to a larger LLM (not implemented here).

## Prerequisites

- **OS**: Windows 11
- **Hardware**: 
  - CPU: Intel i7-9750H or similar
  - RAM: 16 GB
  - GPU: NVIDIA GTX 1650 (4 GB VRAM) with CUDA support
- **Software**: 
  - Python 3.12.6 (via `pyenv`)
  - Visual Studio Code (optional)
  - CUDA Toolkit 12.1 and cuDNN (for GPU support)

## Setup Steps

### 1. Create and Activate Virtual Environment
Open Command Prompt (CMD) and run:

cd F:\Codes\LocalM
pyenv local 3.12.6
python -m venv venv
venv\Scripts\activate

### 2. Install Dependencies
   pip install llama-cpp-python langchain langchain-community pypdf docx2txt sentence-transformers chromadb

# For GPU support, reinstall llama-cpp-python with CUDA:

pip uninstall llama-cpp-python -y
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python-cuBLAS-wheels/ --force-reinstall


### 3. Install CUDA Toolkit and cuDNN (Optional for GPU)

1. Download and install CUDA Toolkit 12.1.
2. Download cuDNN for CUDA 12.1, extract, and copy files to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`.
3. Add CUDA to System PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`

### 4. Download the Model

Download a quantized model (e.g., Llama-2-7B-Chat-GGUF Q4_K_M):

    mkdir models
    cd models
    pip install huggingface-hub
    huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir .
    cd ..

### 5. Prepare the RAG Pipeline Script
    Create rag_pipeline.py

### 6. Add Test Files
    mkdir docs

### 7. Run the Pipeline
    venv\Scripts\activate
    python rag_pipeline.py