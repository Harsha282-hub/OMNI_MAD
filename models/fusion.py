# models/fusion.py
import torch
import torch.nn as nn

class CrossDomainAttention(nn.Module):
    """
    Cross-Attention Mechanism where RGB spatial features (Queries) dynamically query 
    and aggregate features from non-spatial domain branches (Keys/Values).
    """
    def __init__(self, rgb_dim=768, aux_dim=256, embed_dim=256, num_heads=8, dropout=0.1):
        super(CrossDomainAttention, self).__init__()
        self.embed_dim = embed_dim
        
        # Projections to unified embedding dimension
        self.q_proj = nn.Conv2d(rgb_dim, embed_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(aux_dim, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(aux_dim, embed_dim, kernel_size=1)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, rgb_feat, aux_feat):
        """
        Args:
            rgb_feat: [B, rgb_dim, H, W]
            aux_feat: [B, aux_dim, H, W] (Fused Frequency, Texture, Edge features)
        Returns:
            fused_spatial: [B, embed_dim, H, W]
        """
        B, _, H, W = rgb_feat.shape
        N = H * W
        
        # Project and reshape to [B, N, D]
        q = self.q_proj(rgb_feat).flatten(2).permute(0, 2, 1) # [B, N, embed_dim]
        k = self.k_proj(aux_feat).flatten(2).permute(0, 2, 1) # [B, N, embed_dim]
        v = self.v_proj(aux_feat).flatten(2).permute(0, 2, 1) # [B, N, embed_dim]
        
        # Multi-head cross-attention
        attn_out, _ = self.multihead_attn(query=q, key=k, value=v) # [B, N, embed_dim]
        
        # Residual connection + LayerNorm
        attn_out = self.layer_norm(q + attn_out)
        
        # Reshape back to 2D Spatial Map: [B, embed_dim, H, W]
        fused_spatial = attn_out.permute(0, 2, 1).reshape(B, self.embed_dim, H, W)
        return self.out_proj(fused_spatial)


class TransformerContextEncoder(nn.Module):
    """
    Transformer Encoder operating over multi-modal token sequences to model 
    long-range contextual dependencies across spatial patches.
    """
    def __init__(self, embed_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super(TransformerContextEncoder, self).__init__()
        
        # 2D Learnable Positional Embeddings (for 7x7 spatial maps = 49 tokens)
        self.pos_embed = nn.Parameter(torch.randn(1, 49, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x: [B, embed_dim, H, W] (where H=7, W=7 -> N=49 tokens)
        Returns:
            x_transformed: [B, embed_dim, H, W]
            global_token: [B, embed_dim]
        """
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1) # [B, N, C]
        
        # Add positional embedding
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        
        # Pass through Transformer layers
        tokens = self.transformer(tokens) # [B, N, C]
        
        # Reshape back to 2D
        x_transformed = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Global contextual vector (Mean pooling over spatial tokens)
        global_token = tokens.mean(dim=1)
        
        return x_transformed, global_token


class FeaturePyramidFusion(nn.Module):
    """
    Combines outputs from Frequency, Texture, and Edge branches into a unified 
    auxiliary feature map prior to Cross-Attention with RGB features.
    """
    def __init__(self, in_dim=256, out_dim=256):
        super(FeaturePyramidFusion, self).__init__()
        
        # Concatenated dimension = in_dim * 3 (Freq + Texture + Edge)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim * 3, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, freq_feat, texture_feat, edge_feat):
        # All inputs are [B, 256, 7, 7]
        concat_feat = torch.cat([freq_feat, texture_feat, edge_feat], dim=1) # [B, 768, 7, 7]
        return self.fusion_conv(concat_feat) # [B, 256, 7, 7]