import PyPDF2
from docx import Document
from PIL import Image

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Handle cases where extract_text returns None
        return text

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(file_path):
    """Extract text from a TXT file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def extract_text(file_path):
    """Extract text based on file extension."""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")

def generate_image_description(image_path, topic, processor, model, llm):
    """Generate a description for the current image based on the given topic."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    refine_prompt = f"Describe this image related to '{topic}': '{caption}'"
    refined_description = llm(refine_prompt)
    return refined_description.strip()