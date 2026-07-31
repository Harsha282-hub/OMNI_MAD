# models/frequency.py
import torch
import torch.nn as nn

class FrequencyBranch(nn.Module):
    """
    Analyzes the FFT magnitude spectrum. 
    High-frequency artifacts (blending seams, compression mismatches) form specific 
    geometric patterns in the frequency domain that are invisible in the RGB domain.
    """
    def __init__(self, in_channels=1, embed_dim=256):
        super(FrequencyBranch, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        # Squeeze-and-Excitation like spectral attention
        self.spectral_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128 // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128 // 4, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Project to target embedding dimension and align spatial size (7x7)
        self.projector = nn.Sequential(
            nn.Conv2d(128, embed_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((7, 7)) 
        )

    def forward(self, x):
        # Input x: [Batch, 1, 224, 224] (FFT Map)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Apply spectral attention
        attention_weights = self.spectral_attention(x)
        x = x * attention_weights
        
        # Target output: [Batch, embed_dim, 7, 7]
        x_proj = self.projector(x)
        return x_proj