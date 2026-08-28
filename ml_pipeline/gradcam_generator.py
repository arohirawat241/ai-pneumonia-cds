import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import io
import base64
from PIL import Image

def get_model_for_gradcam(device):
    """Loads the trained ResNet50 model for Grad-CAM."""
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    model_path = "ml_pipeline/models/best_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        print("⚠️ Model not found yet.")
        return None

def generate_heatmap_base64(image_bytes: bytes, model, device):
    """Generates a Grad-CAM heatmap."""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb_image = np.float32(pil_image) / 255.0
    
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Define target layer
    target_layers = [model.layer4[-1]]
    
    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # CRITICAL FIX: Enable gradients for Grad-CAM
    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    
    grayscale_cam = grayscale_cam[0, :]
    
    # Resize and overlay
    grayscale_cam_resized = cv2.resize(grayscale_cam, (pil_image.width, pil_image.height))
    visualization = show_cam_on_image(rgb_image, grayscale_cam_resized, use_rgb=True)
    
    # Convert to base64
    result_image = Image.fromarray(visualization)
    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return f"data:image/jpeg;base64,{img_base64}"