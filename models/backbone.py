# models/backbone.py
import torch
import torch.nn as nn
import torchvision.models as models

class RGBBackbone(nn.Module):
    """
    Extracts deep semantic spatial features from RGB images.
    Returns both the 2D feature map (for spatial cross-attention) 
    and the globally pooled embedding (for baseline classification).
    """
    def __init__(self, backbone_name='convnext_tiny', pretrained=True):
        super(RGBBackbone, self).__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == 'convnext_tiny':
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base_model = models.convnext_tiny(weights=weights)
            self.features = base_model.features
            self.out_channels = 768
            
        elif backbone_name == 'efficientnet_v2_s':
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_v2_s(weights=weights)
            self.features = base_model.features
            self.out_channels = 1280
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Adaptive pooling to ensure flat embeddings are consistent
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Input x: [Batch, 3, 224, 224]
        feat_map = self.features(x)  # Output: [Batch, C, 7, 7] for 224 input
        
        pooled = self.global_pool(feat_map)
        flat_emb = torch.flatten(pooled, 1) # Output: [Batch, C]
        
        return feat_map, flat_emb