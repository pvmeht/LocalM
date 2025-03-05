import torch
from langchain_community.llms import LlamaCpp
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

# Llama-2 with reduced GPU layers
llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=5,  # Reduced from 10 to fit 4 GB VRAM
    n_ctx=2048,      # Keep at 2048 for now
    verbose=True,    # Enable for debugging
    n_batch=512
)

blip_processor = None
blip_model = None

def init_blip2():
    global blip_processor, blip_model
    if blip_processor is None:
        blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
        blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b-coco").to(device)