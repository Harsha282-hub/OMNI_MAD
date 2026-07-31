# models/edge.py
import torch
import torch.nn as nn

class EdgeBranch(nn.Module):
    """
    Processes structural boundary maps (Sobel). 
    Designed to highlight 'ghosting' artifacts and splicing seams around fiducial points.
    """
    def __init__(self, in_channels=1, embed_dim=256):
        super(EdgeBranch, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            # Keep spatial resolution high initially to not lose fine edges
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        # Spatial Attention Module (SAM) to focus on corrupted edge regions
        self.sam = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        self.projector = nn.Sequential(
            nn.Conv2d(128, embed_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((7, 7))
        )

    def forward(self, x):
        # Input x: [Batch, 1, 224, 224] (Sobel Edge Map)
        features = self.conv_blocks(x)
        
        # Highlight regions with anomalous edges
        spatial_attention = self.sam(features)
        features = features * spatial_attention
        
        # Target output: [Batch, embed_dim, 7, 7]
        return self.projector(features)