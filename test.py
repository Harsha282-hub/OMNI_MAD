# test.py
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from config import cfg
from dataset import get_dataloaders
from models import OmniMAD
from metrics import MADMetricsTracker
from utils.visualization import plot_tsne

def load_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    
    # Load EMA weights if available (they generalize better), else normal weights
    if 'ema_state_dict' in checkpoint:
        print("-> Found EMA weights. Loading EMA state dict for testing.")
        state_dict = checkpoint['ema_state_dict']
    else:
        state_dict = checkpoint['state_dict']
        
    # Handle DataParallel prefix removal if model was trained on Multi-GPU
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    return model

def main():
    print("--- OMNI-MAD Evaluation ---")
    
    # 1. Load Data
    _, _, test_loader = get_dataloaders(cfg)
    
    # 2. Build Model
    model = OmniMAD(cfg).to(cfg.DEVICE)
    checkpoint_path = os.path.join(cfg.SAVE_DIR, cfg.RUN_NAME, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")
        
    model = load_checkpoint(model, checkpoint_path)
    model.eval()
    
    metrics_tracker = MADMetricsTracker()
    
    all_embeddings = []
    all_labels = []

    print("Running Inference on Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(cfg.DEVICE) for k, v in batch.items()}
            labels = batch['label']
            
            # Forward pass
            outputs = model(batch)
            
            probabilities = F.softmax(outputs['logits'], dim=1)[:, 1] # Class 1 (Morph) prob
            predictions = torch.argmax(outputs['logits'], dim=1)
            
            metrics_tracker.update(labels, probabilities, predictions)
            
            # Collect for t-SNE
            all_embeddings.append(outputs['embeddings'].cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 3. Compute and Print Metrics
    results = metrics_tracker.compute_all()
    
    print("\n" + "="*40)
    print("TEST SET RESULTS (ISO/IEC 30107-3)")
    print("="*40)
    for k, v in results.items():
        print(f"{k:>20}: {v:.4f}")
    print("="*40)
    
    # 4. Generate Visualizations
    out_dir = os.path.join(cfg.LOG_DIR, cfg.RUN_NAME, "plots")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving plots to {out_dir}...")
    metrics_tracker.plot_roc_curve(os.path.join(out_dir, "roc_curve.png"))
    metrics_tracker.plot_confusion_matrix(os.path.join(out_dir, "confusion_matrix.png"))
    metrics_tracker.plot_calibration_curve(os.path.join(out_dir, "calibration_curve.png"))
    
    # Generate t-SNE
    embeddings_cat = np.concatenate(all_embeddings, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)
    plot_tsne(embeddings_cat, labels_cat, os.path.join(out_dir, "tsne_embeddings.png"))
    
    print("Evaluation Complete!")

if __name__ == "__main__":
    main()