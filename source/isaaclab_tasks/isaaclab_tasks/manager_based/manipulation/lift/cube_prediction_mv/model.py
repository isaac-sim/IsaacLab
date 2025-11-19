import torch
import torch.nn as nn
import torchvision.models as models

class CubePositionNet(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Load pretrained CNN
        resnet = models.__dict__[backbone_name](pretrained=pretrained)
        # Remove final classifier (FC layer)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # output: [B, 512, 1, 1]
        
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(2048*3, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 3)   # x, y, z (position)
        )

    def forward(self, imgA, imgB, imgC):
        featA = self.feature_extractor(imgA).flatten(1)  # [B,2048]
        featB = self.feature_extractor(imgB).flatten(1)  # [B,2048]
        featC = self.feature_extractor(imgC).flatten(1)  # [B,2048]
        out = self.fc(torch.cat([featA, featB, featC], dim=1))
        return out
