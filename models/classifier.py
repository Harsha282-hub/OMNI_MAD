# models/classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArtifactLocalizationHead(nn.Module):
    """
    Segmentation head that upsamples spatial features [B, C, 7, 7] to [B, 1, 224, 224] 
    to localize morphed regions (e.g., eye blending, skin seams) for explainability.
    """
    def __init__(self, in_channels=256):
        super(ArtifactLocalizationHead, self).__init__()
        
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            # 14x14 -> 28x28
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # 28x28 -> 56x56
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            # 56x56 -> 112x112
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            
            # 112x112 -> 224x224
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # Probability map [0, 1]
        )

    def forward(self, spatial_map):
        return self.decoder(spatial_map) # [B, 1, 224, 224]


class ClassificationHead(nn.Module):
    """
    Maps globally pooled fusion representations into class predictions and 
    normalized embeddings for Sub-Center ArcFace and CDCL losses.
    """
    def __init__(self, in_features=512, embedding_dim=512, num_classes=2):
        super(ClassificationHead, self).__init__()
        
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Dropout(0.3)
        )
        
        self.fc_logits = nn.Linear(embedding_dim, num_classes)

    def forward(self, fused_features):
        # Input fused_features: [B, in_features]
        embeddings = self.embedding_layer(fused_features) # [B, embedding_dim]
        logits = self.fc_logits(embeddings) # [B, num_classes]
        
        return logits, embeddings