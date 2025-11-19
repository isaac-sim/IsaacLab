import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import CubePositionNet

def load_model (model_path='best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CubePositionNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


#Model will be loaded prior to calling play function
def play(imgA, imgB, imgC, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imgA = transform(imgA).to(device)
    imgB = transform(imgB).to(device)
    imgC = transform(imgC).to(device)

    # Make prediction
    with torch.no_grad():
        pred = model(imgA.unsqueeze(0), imgB.unsqueeze(0), imgC.unsqueeze(0))

    position = pred[:3]
    

    print(f"Predicted Position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")
    return position
