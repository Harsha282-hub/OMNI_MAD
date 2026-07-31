# transforms.py
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size=256):
    """Training augmentations with robust domain perturbations."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(60, 100), p=0.5),
        A.GaussNoise(p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(20, 40), hole_width_range=(20, 40), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=256):
    """Validation and test transformations (Strict Resizing & Normalization)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

class DomainExtractors:
    """Extracts frequency, edge, and texture priors for OMNI-MAD."""
    def __init__(self, image_size=256):
        self.image_size = image_size

    def extract_frequency(self, rgb_tensor):
        """Computes Fast Fourier Transform (FFT) magnitude spectrum."""
        gray = torch.mean(rgb_tensor, dim=0, keepdim=True)
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)
        magnitude = torch.log(magnitude + 1.0)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        return magnitude

    def extract_edge(self, img_rgb):
        """Extracts Canny edge maps."""
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.resize(img_gray, (self.image_size, self.image_size))
        edges = cv2.Canny(img_gray, 100, 200)
        edge_tensor = torch.from_numpy(edges).float().unsqueeze(0) / 255.0
        return edge_tensor

    def extract_texture(self, img_rgb):
        """Extracts high-pass texture maps via Laplacian filter."""
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.resize(img_gray, (self.image_size, self.image_size))
        laplacian = cv2.Laplacian(img_gray, cv2.CV_32F)
        texture_tensor = torch.from_numpy(laplacian).float().unsqueeze(0)
        texture_tensor = (texture_tensor - texture_tensor.min()) / (texture_tensor.max() - texture_tensor.min() + 1e-8)
        return texture_tensor