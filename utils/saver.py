# utils/saver.py
import os
import torch
import shutil

class ModelSaver:
    def __init__(self, save_dir, run_name):
        self.save_dir = os.path.join(save_dir, run_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_eer = float('inf')

    def save_checkpoint(self, state, is_best, filename="checkpoint.pth"):
        """Saves checkpoint and optionally a best_model.pth if is_best is True."""
        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)
        
        if is_best:
            best_filepath = os.path.join(self.save_dir, "best_model.pth")
            shutil.copyfile(filepath, best_filepath)

    def update_and_save(self, model, optimizer, epoch, metrics, ema_model=None):
        """Checks if current EER is the best, and saves the models."""
        current_eer = metrics.get('EER', float('inf'))
        is_best = current_eer < self.best_eer
        
        if is_best:
            self.best_eer = current_eer

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_eer': self.best_eer,
            'metrics': metrics
        }
        
        if ema_model is not None:
            state['ema_state_dict'] = ema_model.state_dict()

        self.save_checkpoint(state, is_best)
        return is_best