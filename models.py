# models.py
import torch
import os
from langchain_community.llms import LlamaCpp
from transformers import CLIPProcessor, CLIPModel

# Get the absolute path to the 'models' directory relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "llama-2-7b-chat.Q4_K_M.gguf")

# Local text model (Llama-2-7B-Chat-Q4_K_M)
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=10,
    n_ctx=2048,
    verbose=True,
    n_batch=512
)

# CLIP model for image recognition
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")