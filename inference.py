# inference.py
import os
import cv2
import time
import torch
import torch.nn.functional as F

from config import cfg
from models import OmniMAD
from transforms import get_val_transforms, DomainExtractors
from utils.visualization import visualize_artifact_mask

class OMNIMADPredictor:
    def __init__(self, checkpoint_path=None):
        self.device = cfg.DEVICE
        self.model = OmniMAD(cfg).to(self.device)
        self.transform = get_val_transforms(cfg.IMAGE_SIZE)
        self.domain_extractor = DomainExtractors(image_size=cfg.IMAGE_SIZE)
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(cfg.SAVE_DIR, cfg.RUN_NAME, "best_model.pth")
            
        self._load_weights(checkpoint_path)
        self.model.eval()
        print(f"Predictor initialized on {self.device}.")

    def _load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights not found at {path}")
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False
        )
        state_dict = checkpoint.get('ema_state_dict', checkpoint.get('state_dict'))
        
        # Strip DDP 'module.' prefix if it exists
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state_dict)

    def preprocess(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Read and convert to RGB
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Extract non-spatial priors
        edge_tensor = self.domain_extractor.extract_edge(img_rgb)
        texture_tensor = self.domain_extractor.extract_texture(img_rgb)
        
        # 2. RGB augmentations (Normalization & Resize)
        augmented = self.transform(image=img_rgb)
        rgb_tensor = augmented['image']
        
        # 3. Frequency prior
        freq_tensor = self.domain_extractor.extract_frequency(rgb_tensor)
        
        # Add batch dimension and move to device
        batch = {
            "rgb": rgb_tensor.unsqueeze(0).to(self.device),
            "freq": freq_tensor.unsqueeze(0).to(self.device),
            "edge": edge_tensor.unsqueeze(0).to(self.device),
            "texture": texture_tensor.unsqueeze(0).to(self.device),
        }
        return batch

    @torch.no_grad()
    def predict(self, image_path, save_heatmap=True, out_path="artifact_heatmap.jpg"):
        batch = self.preprocess(image_path)
        
        start_time = time.time()
        outputs = self.model(batch)
        inference_time = (time.time() - start_time) * 1000 # ms
        
        # Calculate probabilities
        probs = F.softmax(outputs['logits'], dim=1).squeeze(0)
        morph_prob = probs[1].item()
        bona_fide_prob = probs[0].item()
        
        prediction_label = "Morph (Attack)" if morph_prob > 0.5 else "Bona Fide (Real)"
        confidence = max(morph_prob, bona_fide_prob)
        
        # Generate heatmap overlay
        if save_heatmap:
            rgb_tensor = batch["rgb"].squeeze(0)
            mask_tensor = outputs["artifact_mask"].squeeze(0)
            visualize_artifact_mask(rgb_tensor, mask_tensor, out_path)
            
        return {
            "prediction": prediction_label,
            "confidence": f"{confidence * 100:.2f}%",
            "morph_probability": morph_prob,
            "inference_time_ms": f"{inference_time:.2f}",
            "heatmap_path": out_path if save_heatmap else None
        }

if __name__ == "__main__":
    # Test the pipeline locally
    # Replace with an actual test image path
    sample_image = "sample_test_image.jpg" 
    
    if os.path.exists(sample_image):
        predictor = OMNIMADPredictor()
        result = predictor.predict(sample_image, save_heatmap=True, out_path="result_heatmap.jpg")
        print("\n--- Prediction Result ---")
        for k, v in result.items():
            print(f"{k.capitalize()}: {v}")
    else:
        print(f"Put a sample image named '{sample_image}' in the root directory to test inference standalone.")