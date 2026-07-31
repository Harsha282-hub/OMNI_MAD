# losses.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss to handle hard-to-classify high-quality morphs.
    Down-weights well-classified examples and focuses on hard negatives.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SubCenterArcFace(nn.Module):
    """
    Morph-Centric Sub-Center ArcFace.
    Standard ArcFace assumes unimodal distributions per class. However, morphed faces 
    come from various algorithms (StyleGAN, LMA, OpenCV), forming distinct sub-clusters.
    This module creates K sub-centers for the Morph class to prevent representation collapse.
    """
    def __init__(self, in_features, num_classes=2, k_sub_centers=3, s=64.0, m=0.50):
        super(SubCenterArcFace, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.k_sub_centers = k_sub_centers
        self.s = s
        self.m = m
        
        # We assign 1 center for Bona Fide (Class 0) and K centers for Morph (Class 1)
        self.total_centers = 1 + self.k_sub_centers
        self.weight = nn.Parameter(torch.FloatTensor(self.total_centers, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weights)
        
        # Combine sub-centers for the Morph class by taking the max similarity
        # Class 0: index 0
        # Class 1: max of indices 1 to K
        cosine_class_0 = cosine[:, 0].unsqueeze(1)
        cosine_class_1, _ = torch.max(cosine[:, 1:], dim=1, keepdim=True)
        cosine_merged = torch.cat([cosine_class_0, cosine_class_1], dim=1)

        # Add margin penalty
        sine = torch.sqrt(1.0 - torch.pow(cosine_merged, 2))
        phi = cosine_merged * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine_merged > self.th, phi, cosine_merged - self.mm)

        # Apply margin only to the ground truth class
        one_hot = torch.zeros(cosine_merged.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine_merged)
        output *= self.s
        
        return F.cross_entropy(output, labels)


class CrossDomainConsistencyLoss(nn.Module):
    """
    Novel Contribution: CDCL
    Forces RGB and Frequency domain embeddings to be consistent for Real faces, 
    but explicitly pushes them apart for Morphed faces (as morphing creates frequency anomalies).
    """
    def __init__(self, margin=1.0):
        super(CrossDomainConsistencyLoss, self).__init__()
        self.margin = margin

    def forward(self, rgb_emb, freq_emb, labels):
        # Normalize embeddings
        rgb_emb = F.normalize(rgb_emb, p=2, dim=1)
        freq_emb = F.normalize(freq_emb, p=2, dim=1)
        
        # Pairwise euclidean distance between domains
        distances = (rgb_emb - freq_emb).pow(2).sum(1)
        
        # For real (label=0): minimize distance
        # For morph (label=1): maximize distance up to a margin
        real_loss = (1 - labels) * distances
        morph_loss = labels * F.relu(self.margin - torch.sqrt(distances + 1e-9)).pow(2)
        
        return (real_loss + morph_loss).mean()


class OmniMADLoss(nn.Module):
    """
    Wrapper module that combines the multi-objective losses based on config weights.
    """
    def __init__(self, config):
        super(OmniMADLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        self.focal = FocalLoss() if config.W_FOCAL > 0 else None
        self.arcface = SubCenterArcFace(in_features=config.EMBEDDING_DIM)
        self.cdcl = CrossDomainConsistencyLoss()
        self.cfg = config

    def forward(self, logits, embeddings, rgb_emb, freq_emb, labels):
        loss_ce = self.ce(logits, labels)
        
        total_loss = self.cfg.W_CE * loss_ce
        
        if self.focal is not None:
            total_loss += self.cfg.W_FOCAL * self.focal(logits, labels)
            
        if self.cfg.W_ARCFACE > 0 and embeddings is not None:
            total_loss += self.cfg.W_ARCFACE * self.arcface(embeddings, labels)
            
        if self.cfg.W_CDCL > 0 and rgb_emb is not None and freq_emb is not None:
            total_loss += self.cfg.W_CDCL * self.cdcl(rgb_emb, freq_emb, labels)
            
        return total_loss