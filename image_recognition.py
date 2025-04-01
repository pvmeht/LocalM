import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, UnidentifiedImageError
import io
import requests

class ImageRecognizer:
    def __init__(self):
        # Use 'weights' instead of deprecated 'pretrained'
        self.classifier = models.resnet18(weights="IMAGENET1K_V1")
        self.classifier.eval()
        self.class_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Download and load ImageNet labels
        with open("imagenet_classes.txt", "w") as f:
            f.write(requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text)
        with open("imagenet_classes.txt") as f:
            self.labels = [line.strip() for line in f.readlines()]

    def classify(self, image_data):
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except UnidentifiedImageError:
            return "Invalid image file: Unable to identify the image. Please ensure it's a valid image (e.g., JPEG, PNG)."
        image = self.class_transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.classifier(image)
            _, predicted = outputs.max(1)
        return self.labels[predicted.item()]