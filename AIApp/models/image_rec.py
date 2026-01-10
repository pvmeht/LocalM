# from transformers import ViTFeatureExtractor, ViTForImageClassification
# from PIL import Image
# import torch

# feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)

# def recognize_image(image_path: str) -> str:
#     image = Image.open(image_path).convert("RGB")
#     inputs = feature_extractor(images=image, return_tensors="pt")
#     for k, v in inputs.items():
#         inputs[k] = v.to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     predicted_class = outputs.logits.argmax(-1).item()
#     return model.config.id2label[predicted_class]



from transformers import ViTImageProcessor, ViTForImageClassification  # Updated to ViTImageProcessor
from PIL import Image
import torch
import torch.nn.functional as F  # For confidence

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")  # Renamed from feature_extractor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)

def recognize_image(image_path: str) -> dict:
    """
    Recognize image and return label + confidence.
    Returns dict for richer response.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
        probabilities = F.softmax(logits, dim=-1)
        confidence = probabilities[0][predicted_class].item()
        
        label = model.config.id2label[predicted_class]
        return {"label": label, "confidence": round(confidence, 4)}
    
    except Exception as e:
        raise ValueError(f"Image recognition failed: {str(e)}")