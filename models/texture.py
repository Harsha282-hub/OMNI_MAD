# models/texture.py
import torch
import torch.nn as nn

class TextureBranch(nn.Module):
    """
    Multi-scale processing of LBP (Local Binary Pattern) maps.
    Uses dilated convolutions to capture texture anomalies at different scales 
    without losing spatial resolution immediately.
    """
    def __init__(self, in_channels=1, embed_dim=256):
        super(TextureBranch, self).__init__()
        
        # Scale 1: Fine micro-textures (pores, fine wrinkles)
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Scale 2: Medium textures (larger blending spots) using Dilation
        self.scale2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Scale 3: Macro structures
        self.scale3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        # Align spatial dimension for fusion
        self.projector = nn.Sequential(
            nn.Conv2d(128, embed_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((7, 7))
        )

    def forward(self, x):
        # Input x: [Batch, 1, 224, 224] (LBP Map)
        x = self.scale1(x)
        x = self.scale2(x)
        x = self.scale3(x)
        
        # Target output: [Batch, embed_dim, 7, 7]
        return self.projector(x)