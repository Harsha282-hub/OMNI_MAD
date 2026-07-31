# metrics.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, auc, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve

class MADMetricsTracker:
    """
    Computes Standard and Biometrics ISO/IEC 30107-3 Metrics for Morphing Attack Detection.
    Bona Fide (Real) = 0, Attack (Morph) = 1.
    """
    def __init__(self):
        self.y_true = []
        self.y_prob = []  # Probabilities of class 1 (Morph)
        self.y_pred = []  # Thresholded predictions

    def update(self, labels, probabilities, predictions):
        """Update tracker with batch results."""
        self.y_true.extend(labels.detach().cpu().numpy())
        self.y_prob.extend(probabilities.detach().cpu().numpy())
        self.y_pred.extend(predictions.detach().cpu().numpy())

    def reset(self):
        self.y_true = []
        self.y_prob = []
        self.y_pred = []

    def compute_all(self):
        y_true = np.array(self.y_true)
        y_prob = np.array(self.y_prob)
        y_pred = np.array(self.y_pred)

        # Standard Metrics
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # ROC and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # ISO Metrics
        # APCER (Attack Presentation Classification Error Rate) - False Accept Rate for Morphs
        # BPCER (Bona Fide Presentation Classification Error Rate) - False Reject Rate for Reals
        fnr = 1 - tpr
        
        # Find Equal Error Rate (EER) where FPR == FNR
        eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = fpr[eer_idx]
        optimal_threshold = thresholds[eer_idx]

        # Calculate APCER and BPCER at the 0.5 threshold
        morph_indices = (y_true == 1)
        bona_fide_indices = (y_true == 0)
        
        apcer = np.mean(y_pred[morph_indices] == 0) if np.sum(morph_indices) > 0 else 0.0
        bpcer = np.mean(y_pred[bona_fide_indices] == 1) if np.sum(bona_fide_indices) > 0 else 0.0
        acer = (apcer + bpcer) / 2.0

        return {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "AUC": roc_auc,
            "EER": eer,
            "APCER": apcer,
            "BPCER": bpcer,
            "ACER": acer,
            "Optimal_Threshold": optimal_threshold
        }

    def plot_roc_curve(self, save_path="roc_curve.png"):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (BPCER)')
        plt.ylabel('True Positive Rate (1 - APCER)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, save_path="confusion_matrix.png"):
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Bona Fide', 'Morph'], 
                    yticklabels=['Bona Fide', 'Morph'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_calibration_curve(self, save_path="calibration_curve.png"):
        prob_true, prob_pred = calibration_curve(self.y_true, self.y_prob, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model Calibration')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives (Morphs)')
        plt.title('Calibration Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()