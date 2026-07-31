# models/omni_mad.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import RGBBackbone
from .frequency import FrequencyBranch
from .texture import TextureBranch
from .edge import EdgeBranch
from .fusion import FeaturePyramidFusion, CrossDomainAttention, TransformerContextEncoder
from .classifier import ClassificationHead, ArtifactLocalizationHead

class OmniMAD(nn.Module):
    """
    Omni-Domain Morphing Attack Detection (OMNI-MAD) Architecture.
    Fuses RGB Spatial, Frequency (FFT), Micro-Texture (LBP), and Edge (Sobel) features 
    via Cross-Domain Attention and Transformer Context Modeling.
    """
    def __init__(self, config):
        super(OmniMAD, self).__init__()
        self.cfg = config
        
        # 1. Multi-Domain Feature Extractors
        self.rgb_backbone = RGBBackbone(
            backbone_name=config.BACKBONE, 
            pretrained=config.PRETRAINED
        )
        rgb_dim = self.rgb_backbone.out_channels # 768 for convnext_tiny
        
        branch_dim = 256
        self.use_freq = config.USE_FREQ_BRANCH
        self.use_texture = config.USE_TEXTURE_BRANCH
        self.use_edge = config.USE_EDGE_BRANCH
        
        if self.use_freq:
            self.freq_branch = FrequencyBranch(in_channels=1, embed_dim=branch_dim)
        if self.use_texture:
            self.texture_branch = TextureBranch(in_channels=1, embed_dim=branch_dim)
        if self.use_edge:
            self.edge_branch = EdgeBranch(in_channels=1, embed_dim=branch_dim)

        # Domain Projection for Loss Calculations (CDCL)
        self.rgb_proj = nn.Linear(rgb_dim, config.EMBEDDING_DIM)
        self.freq_proj = nn.Linear(branch_dim, config.EMBEDDING_DIM)

        # 2. Auxiliary Fusion
        self.pyramid_fusion = FeaturePyramidFusion(in_dim=branch_dim, out_dim=branch_dim)

        # 3. Cross-Domain Attention (RGB Queries, Aux Keys/Values)
        self.cross_attention = CrossDomainAttention(
            rgb_dim=rgb_dim,
            aux_dim=branch_dim,
            embed_dim=branch_dim,
            num_heads=config.TRANSFORMER_HEADS
        )

        # 4. Transformer Context Encoder
        self.transformer_encoder = TransformerContextEncoder(
            embed_dim=branch_dim,
            num_heads=config.TRANSFORMER_HEADS,
            num_layers=config.TRANSFORMER_LAYERS
        )

        # 5. Output Heads
        # Combined feature size = RGB pooled (768) + Transformer global token (256) = 1024
        combined_dim = rgb_dim + branch_dim
        
        self.classifier = ClassificationHead(
            in_features=combined_dim, 
            embedding_dim=config.EMBEDDING_DIM, 
            num_classes=config.NUM_CLASSES
        )
        
        self.localization_head = ArtifactLocalizationHead(in_channels=branch_dim)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict (dict): Batch containing 'rgb', 'freq', 'edge', 'texture' tensors.
        Returns:
            dict containing logits, embeddings, domain projections, and artifact mask.
        """
        rgb_img = batch_dict['rgb']       # [B, 3, 224, 224]
        freq_img = batch_dict['freq']     # [B, 1, 224, 224]
        edge_img = batch_dict['edge']     # [B, 1, 224, 224]
        texture_img = batch_dict['texture'] # [B, 1, 224, 224]

        # 1. Extract RGB Features
        rgb_map, rgb_flat = self.rgb_backbone(rgb_img) # [B, 768, 7, 7], [B, 768]

        # 2. Extract Auxiliary Domain Features
        freq_feat = self.freq_branch(freq_img) if self.use_freq else torch.zeros(rgb_img.shape[0], 256, 7, 7, device=rgb_img.device)
        texture_feat = self.texture_branch(texture_img) if self.use_texture else torch.zeros(rgb_img.shape[0], 256, 7, 7, device=rgb_img.device)
        edge_feat = self.edge_branch(edge_img) if self.use_edge else torch.zeros(rgb_img.shape[0], 256, 7, 7, device=rgb_img.device)

        # 3. Domain Projections for CDCL Loss
        freq_flat = F.adaptive_avg_pool2d(freq_feat, (1, 1)).flatten(1)
        rgb_emb_proj = self.rgb_proj(rgb_flat)
        freq_emb_proj = self.freq_proj(freq_flat)

        # 4. Auxiliary Feature Pyramid Fusion
        aux_fused_map = self.pyramid_fusion(freq_feat, texture_feat, edge_feat) # [B, 256, 7, 7]

        # 5. Cross-Domain Attention
        cross_attended_map = self.cross_attention(rgb_map, aux_fused_map) # [B, 256, 7, 7]

        # 6. Transformer Context Encoder
        transformed_map, context_token = self.transformer_encoder(cross_attended_map) # [B, 256, 7, 7], [B, 256]

        # 7. Feature Concatenation for Classification
        fused_global = torch.cat([rgb_flat, context_token], dim=1) # [B, 1024]

        # 8. Classification & Metric Learning Embeddings
        logits, embeddings = self.classifier(fused_global)

        # 9. Artifact Localization Mask Generation
        artifact_mask = self.localization_head(transformed_map) # [B, 1, 224, 224]

        return {
            "logits": logits,
            "embeddings": embeddings,
            "rgb_emb": rgb_emb_proj,
            "freq_emb": freq_emb_proj,
            "artifact_mask": artifact_mask
        }