# train.py
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import cfg
from dataset import get_dataloaders
from models import OmniMAD
from trainer import OMNIMADTrainer
from utils.logger import ExperimentLogger
from utils.saver import ModelSaver

def set_seed(seed):
    """Ensures complete reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    print(f"Initializing Project: {cfg.PROJECT_NAME} | Run: {cfg.RUN_NAME}")
    set_seed(cfg.SEED)

    # 1. Setup Utilities
    logger = ExperimentLogger(cfg)
    saver = ModelSaver(save_dir=cfg.SAVE_DIR, run_name=cfg.RUN_NAME)

    # 2. Data Loaders
    logger.info("Loading Datasets (SMDD)...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # 3. Initialize Model
    logger.info(f"Building OMNI-MAD Architecture with {cfg.BACKBONE} backbone...")
    model = OmniMAD(cfg)
    
    if cfg.MULTI_GPU and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = torch.nn.DataParallel(model)

    # 4. Optimizer & Scheduler
    # We use AdamW which handles weight decay better for Transformer modules
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.LEARNING_RATE, 
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=cfg.EPOCHS, 
        eta_min=cfg.MIN_LR
    )

    # 5. Trainer setup
    trainer = OMNIMADTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg,
        logger=logger,
        saver=saver
    )

    # 6. Start Training
    try:
        trainer.fit()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Shutting down gracefully...")
    finally:
        logger.info("Training complete.")

if __name__ == "__main__":
    main()