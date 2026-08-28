import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes, pretrained=True):
    # Load pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    
    # Freeze early layers to save compute (optional, but good for small datasets)
    # for param in model.parameters():
    #     param.requires_grad = False

    # Replace the final fully connected layer
    # ResNet50 has 2048 features in its final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model