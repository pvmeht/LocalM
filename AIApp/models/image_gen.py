# from PIL import Image
# from diffusers import StableDiffusionPipeline
# import torch

# # Choose device safely
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # Use float16 on GPU, float32 on CPU
# dtype = torch.float16 if device == "cuda" else torch.float32

# # Load pipeline with an appropriate dtype for the target device
# pipe = StableDiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4",
#     torch_dtype=dtype,
# )

# if device == "cuda":
#     pipe = pipe.to(device)
# else:
#     # CPU-friendly settings (will be slow for SD models)
#     try:
#         pipe.enable_attention_slicing()
#     except Exception:
#         pass
#     pipe = pipe.to(device)

# def generate_image(prompt: str) -> Image.Image:
#     image = pipe(prompt, num_inference_steps=50).images[0]
#     return image



from PIL import Image
from diffusers import StableDiffusionPipeline
import torch

# Choose device safely
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use float16 on GPU, float32 on CPU
dtype = torch.float16 if device == "cuda" else torch.float32

# Load pipeline with an appropriate dtype for the target device
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=dtype,
)

if device == "cuda":
    pipe = pipe.to(device)
else:
    # CPU-friendly settings (will be slow for SD models)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    pipe = pipe.to(device)

def generate_image(prompt: str, num_inference_steps: int = 20) -> Image.Image:  # Reduced default steps for speed
    """
    Generate image from prompt.
    num_inference_steps: Controls quality/speed (20=fast, 50=high-quality).
    """
    try:
        image = pipe(
            prompt, 
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5  # Strength of prompt adherence
        ).images[0]
        return image
    except Exception as e:
        raise ValueError(f"Image generation failed: {str(e)}")