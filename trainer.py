# trainer.py
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import copy
from tqdm import tqdm

from losses import OmniMADLoss
from metrics import MADMetricsTracker

class ModelEMA:
    """Exponential Moving Average of model weights for training stability."""
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
                if ema_v.dtype.is_floating_point:
                    ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * model_v)


class OMNIMADTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, config, logger, saver):
        self.cfg = config
        self.device = config.DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.saver = saver
        
        self.criterion = OmniMADLoss(config).to(self.device)
        self.metrics = MADMetricsTracker()
        
        self.scaler = GradScaler(enabled=config.MIXED_PRECISION)
        self.ema = ModelEMA(model, decay=config.EMA_DECAY) if config.USE_EMA else None

        # Early stopping tracking
        self.best_eer = float('inf')
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg.EPOCHS} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch['label']

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Forward Pass
            with autocast(enabled=self.cfg.MIXED_PRECISION):
                outputs = self.model(batch)
                
                loss = self.criterion(
                    outputs['logits'], 
                    outputs['embeddings'], 
                    outputs['rgb_emb'], 
                    outputs['freq_emb'], 
                    labels
                )

            # Backward and Optimize
            self.scaler.scale(loss).backward()
            
            # Unscale before gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.GRAD_CLIP)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema:
                self.ema.update(self.model)

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        return {"Loss": total_loss / len(self.train_loader)}

    @torch.no_grad()
    def validate_epoch(self, epoch):
        # Use EMA model for validation if enabled, else use normal model
        eval_model = self.ema.ema if self.ema else self.model
        eval_model.eval()
        
        total_loss = 0.0
        self.metrics.reset()
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.cfg.EPOCHS} [Val]")
        
        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch['label']
            
            outputs = eval_model(batch)
            
            loss = self.criterion(
                outputs['logits'], 
                outputs['embeddings'], 
                outputs['rgb_emb'], 
                outputs['freq_emb'], 
                labels
            )
            total_loss += loss.item()

            # Process predictions
            probabilities = F.softmax(outputs['logits'], dim=1)[:, 1] # Prob of Morph
            predictions = torch.argmax(outputs['logits'], dim=1)
            
            self.metrics.update(labels, probabilities, predictions)

        val_results = self.metrics.compute_all()
        val_results["Loss"] = total_loss / len(self.val_loader)
        
        return val_results

    def fit(self):
        self.logger.info("Starting OMNI-MAD Training Pipeline...")
        
        for epoch in range(1, self.cfg.EPOCHS + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            self.logger.log_metrics(train_metrics, step=epoch, phase="Train")
            self.logger.log_metrics(val_metrics, step=epoch, phase="Val")
            
            self.scheduler.step()
            
            # Checkpoint and Early Stopping
            is_best = self.saver.update_and_save(
                model=self.model, 
                optimizer=self.optimizer, 
                epoch=epoch, 
                metrics=val_metrics,
                ema_model=self.ema.ema if self.ema else None
            )
            
            if is_best:
                self.patience_counter = 0
                self.logger.info(f"New Best EER: {val_metrics['EER']:.4f}! Model saved.")
            else:
                self.patience_counter += 1
                self.logger.info(f"No improvement. Patience: {self.patience_counter}/{self.cfg.EARLY_STOPPING_PATIENCE}")
                
                if self.patience_counter >= self.cfg.EARLY_STOPPING_PATIENCE:
                    self.logger.info("Early stopping triggered. Training stopped.")
                    break
        
        self.logger.close()