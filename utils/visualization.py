# utils/visualization.py
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def visualize_artifact_mask(rgb_tensor, mask_tensor, save_path):
    """
    Overlays the predicted artifact heatmap onto the original RGB image.
    rgb_tensor: [3, H, W] normalized tensor
    mask_tensor: [1, H, W] probability mask [0, 1]
    """
    # Denormalize RGB
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_img = rgb_tensor.cpu() * std + mean
    rgb_img = torch.clamp(rgb_img, 0, 1).permute(1, 2, 0).numpy()
    rgb_img = (rgb_img * 255).astype(np.uint8)
    
    # Process Mask
    mask = mask_tensor.squeeze().cpu().detach().numpy()
    mask = np.uint8(255 * mask)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
    # Convert RGB for OpenCV blending
    rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    
    # Blend
    overlay = cv2.addWeighted(rgb_bgr, 0.5, heatmap, 0.5, 0)
    
    # Save
    cv2.imwrite(save_path, overlay)

def plot_tsne(embeddings, labels, save_path):
    """
    Generates a t-SNE scatter plot of the extracted features.
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_emb = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_emb[:, 0], y=reduced_emb[:, 1], 
        hue=labels, palette=['blue', 'red'], alpha=0.7
    )
    plt.title("t-SNE Projection of OMNI-MAD Embeddings")
    plt.legend(title="Class", labels=["Bona Fide (Real)", "Morph (Attack)"])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()