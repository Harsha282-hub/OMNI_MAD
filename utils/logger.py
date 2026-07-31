# utils/logger.py
import logging
import sys
import wandb
from torch.utils.tensorboard import SummaryWriter
import os

class ExperimentLogger:
    def __init__(self, config):
        self.cfg = config
        self.use_wandb = config.USE_WANDB
        
        # Setup Python Logging
        self.logger = logging.getLogger(config.PROJECT_NAME)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # File handler
        log_file = os.path.join(config.LOG_DIR, f"{config.RUN_NAME}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Setup TensorBoard
        self.tb_writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, config.RUN_NAME))

        # Setup WandB
        if self.use_wandb:
            wandb.init(
                project=config.WANDB_PROJECT,
                name=config.RUN_NAME,
                config=vars(config),
                reinit=True
            )

    def info(self, message):
        self.logger.info(message)

    def log_metrics(self, metrics, step, phase="Train"):
        """Logs metrics to Console, TensorBoard, and WandB."""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"[{phase}] Epoch {step} -> {metrics_str}")
        
        for k, v in metrics.items():
            self.tb_writer.add_scalar(f"{phase}/{k}", v, step)
            
        if self.use_wandb:
            wandb.log({f"{phase}/{k}": v for k, v in metrics.items()}, step=step)

    def close(self):
        self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()