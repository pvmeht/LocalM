# def load_llm():
#     """Load the Llama-2-7B model from a local GGUF file using ctransformers."""
#     from ctransformers import AutoModelForCausalLM
#     model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"  # Local file path
#     return AutoModelForCausalLM.from_pretrained(
#         model_path,
#         local_files_only=True,  # Ensure it only looks locally
#         model_type="llama"
#     )

# # def load_blip2():
# #     """Load the BLIP-2 model and processor."""
# #     from transformers import Blip2Processor, Blip2ForConditionalGeneration
# #     processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# #     model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
# #     return processor, model

# def load_blip2():
#     """Load a smaller BLIP model."""
#     from transformers import BlipProcessor, BlipForConditionalGeneration
#     processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#     model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#     return processor, model

import torch


def load_llm():
    """Load the Llama-2-7B model using LlamaCpp."""
    from langchain_community.llms import LlamaCpp
    model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"  # Local file path
    return LlamaCpp(
        model_path=model_path,
        n_ctx=2048,  # Larger context length
        n_batch=512,
        verbose=True,
        n_gpu_layers=10 if torch.cuda.is_available() else 0  # Use GPU if available
    )

def load_blip2():
    """Load a smaller BLIP model."""
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# def load_blip2():
#     from transformers import Blip2Processor, Blip2ForConditionalGeneration
#     import torch
#     processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
#     model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
#     if torch.cuda.is_available():
#         model = model.to("cuda")
#         print("BLIP-2 loaded on GPU.")
#     return processor, model 
