import os
import torch  # Add this line
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from models import blip_processor, blip_model, llm, device, init_blip2

def load_text_document(filepath):
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    elif filepath.endswith(".txt"):
        loader = TextLoader(filepath)
    else:
        return []
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = os.path.basename(filepath)
    return docs

def process_image(image_path, topic=None):
    init_blip2()
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = blip_model.generate(**inputs, max_length=50)
    caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    
    if topic:
        prompt = f"Refine this description '{caption}' for the topic '{topic}'."
        caption = llm(prompt, max_tokens=100, temperature=0.7, top_p=0.9)
    
    return Document(
        page_content=caption,
        metadata={"source": os.path.basename(image_path), "type": "image"}
    )