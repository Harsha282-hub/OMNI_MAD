# dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from transforms import get_train_transforms, get_val_transforms, DomainExtractors

class SMDDDataset(Dataset):
    def __init__(self, data_dir, image_size=256, is_train=True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.is_train = is_train
        
        self.transform = get_train_transforms(image_size) if is_train else get_val_transforms(image_size)
        self.domain_extractor = DomainExtractors(image_size=image_size)
        
        self.samples = []
        self._load_dataset()

    def _load_dataset(self):
        """Scans directory structure and maps paths to labels (0: Bona Fide, 1: Morph)."""
        if not os.path.exists(self.data_dir):
            raise NotADirectoryError(f"Dataset directory not found: {self.data_dir}")

        for category in os.listdir(self.data_dir):
            cat_path = os.path.join(self.data_dir, category)
            if not os.path.isdir(cat_path):
                continue
            
            # Determine label based on folder name
            cat_lower = category.lower()
            if "bona_fide" in cat_lower or "real" in cat_lower or cat_lower == "0":
                label = 0
            elif "morph" in cat_lower or "attack" in cat_lower or cat_lower == "1":
                label = 1
            else:
                continue  # Skip unknown folders

            for img_name in os.listdir(cat_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    img_path = os.path.join(cat_path, img_name)
                    self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} images from {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 1. Read image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Fallback for corrupted image paths to prevent crashes
            img_bgr = cv2.zeros((self.image_size, self.image_size, 3), dtype=cv2.uint8)
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. STRICTLY FORCE UNIFORM RESIZING FIRST to prevent size mismatches
        img_rgb = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # 3. Extract non-spatial multi-domain priors
        edge = self.domain_extractor.extract_edge(img_rgb)
        texture = self.domain_extractor.extract_texture(img_rgb)
        
        # 4. Apply spatial RGB augmentations & Normalization
        augmented = self.transform(image=img_rgb)
        rgb = augmented['image']  # Tensor [3, H, W]
        
        # 5. Frequency domain prior (computed from normalized tensor)
        freq = self.domain_extractor.extract_frequency(rgb)

        return {
            "rgb": rgb,
            "freq": freq,
            "edge": edge,
            "texture": texture,
            "label": torch.tensor(label, dtype=torch.long)
        }

def get_dataloaders(cfg):
    """Factory method to build Train, Validation, and Test DataLoaders."""
    train_dataset = SMDDDataset(
        data_dir=os.path.join(cfg.DATA_ROOT, "train"), 
        image_size=cfg.IMAGE_SIZE, 
        is_train=True
    )
    val_dataset = SMDDDataset(
        data_dir=os.path.join(cfg.DATA_ROOT, "val"), 
        image_size=cfg.IMAGE_SIZE, 
        is_train=False
    )
    test_dataset = SMDDDataset(
        data_dir=os.path.join(cfg.DATA_ROOT, "test"), 
        image_size=cfg.IMAGE_SIZE, 
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS, 
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.NUM_WORKERS, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.NUM_WORKERS, 
        pin_memory=True
    )

    return train_loader, val_loader, test_loader