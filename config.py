# config.py
import os
import torch
from dataclasses import dataclass

@dataclass
class Config:
    # ---------------------------------------------------------
    # Project Info
    # ---------------------------------------------------------
    PROJECT_NAME: str = "OMNI-MAD-SMDD"
    RUN_NAME: str = "Omni-Domain-Fusion-v1"
    SEED: int = 42
    
    # ---------------------------------------------------------
    # System & Hardware
    # ---------------------------------------------------------
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = 4 if os.name == 'nt' else 8 # Adjust for Windows
    PIN_MEMORY: bool = True
    MULTI_GPU: bool = torch.cuda.device_count() > 1
    
    # ---------------------------------------------------------
    # Dataset (SMDD)
    # ---------------------------------------------------------
    # TODO: Update these paths to your local SMDD directories
    DATA_ROOT: str = "./data/SMDD"
    TRAIN_DIR: str = os.path.join(DATA_ROOT, "train")
    VAL_DIR: str = os.path.join(DATA_ROOT, "val")
    TEST_DIR: str = os.path.join(DATA_ROOT, "test")
    
    IMAGE_SIZE: int = 224
    IN_CHANNELS: int = 3
    NUM_CLASSES: int = 2 # 0: Bona Fide (Real), 1: Morph
    
    # ---------------------------------------------------------
    # Model Architecture Flags
    # ---------------------------------------------------------
    BACKBONE: str = "convnext_tiny" # Alternatives: efficientnet_v2_s
    PRETRAINED: bool = True
    
    USE_FREQ_BRANCH: bool = True
    USE_TEXTURE_BRANCH: bool = True
    USE_EDGE_BRANCH: bool = True
    USE_TRANSFORMER_FUSION: bool = True
    
    EMBEDDING_DIM: int = 512
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 2
    
    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    EPOCHS: int = 10
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    
    # Optimizer & Scheduler
    OPTIMIZER: str = "AdamW" # AdamW is superior for Transformer hybrid models
    MIN_LR: float = 1e-6
    WARMUP_EPOCHS: int = 5
    
    # Regularization & Stability
    MIXED_PRECISION: bool = True
    GRAD_CLIP: float = 1.0
    USE_EMA: bool = True
    EMA_DECAY: float = 0.999
    EARLY_STOPPING_PATIENCE: int = 10
    
    # ---------------------------------------------------------
    # Loss Function Weights
    # ---------------------------------------------------------
    # Total Loss = (W_CE * CE) + (W_ARCFACE * ArcFace) + (W_CDCL * CrossDomainConsistency)
    W_CE: float = 1.0
    W_FOCAL: float = 0.0 # Can be toggled
    W_ARCFACE: float = 0.5
    W_CDCL: float = 0.2
    
    LABEL_SMOOTHING: float = 0.1
    
    # ---------------------------------------------------------
    # Augmentations (MixUp / CutMix)
    # ---------------------------------------------------------
    MIXUP_ALPHA: float = 0.2
    CUTMIX_ALPHA: float = 1.0
    PROB_MIXUP: float = 0.5
    
    # ---------------------------------------------------------
    # Logging & Checkpoints
    # ---------------------------------------------------------
    SAVE_DIR: str = "./checkpoints"
    LOG_DIR: str = "./logs"
    USE_WANDB: bool = True # Highly recommended for tracking
    WANDB_PROJECT: str = "Morph-Detection-SMDD"
    
    def __post_init__(self):
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

# Instantiate a global config object
cfg = Config()