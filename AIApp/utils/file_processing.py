import PyPDF2
import docx
import pandas as pd
import json
import os

def process_file(file_path: str) -> dict:
    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join(page.extract_text() for page in reader.pages)
        return {"text": text}
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        return {"text": text}
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return {"data": df.to_dict()}
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)
        return {"data": data}
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            text = f.read()
        return {"text": text}
    return {"error": "Unsupported file type"}