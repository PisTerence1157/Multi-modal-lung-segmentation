import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SegmentationMetrics:
    """Metric accumulator for segmentation tasks."""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.dice_scores = []
        self.iou_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.accuracy_scores = []
        self.f1_scores = []
    
    def update(self, pred, target):
        """
        Update metrics with a new batch.
        Args:
            pred: predictions [B, 1, H, W] or [B, H, W]
            target: ground truth [B, 1, H, W] or [B, H, W]
        """
        # Ensure proper input format
        if isinstance(pred, torch.Tensor):
            pred = torch.sigmoid(pred)
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Ensure shapes are consistent
        if pred.ndim == 4:
            pred = pred.squeeze(1)
        if target.ndim == 4:
            target = target.squeeze(1)
        
        # Binarize predictions
        pred_binary = (pred > self.threshold).astype(np.float32)
        target_binary = target.astype(np.float32)
        
        # Compute metrics per sample
        batch_size = pred.shape[0]
        for i in range(batch_size):
            pred_i = pred_binary[i].flatten()
            target_i = target_binary[i].flatten()
            
            # Dice Score
            dice = self._calculate_dice(pred_i, target_i)
            self.dice_scores.append(dice)
            
            # IoU Score
            iou = self._calculate_iou(pred_i, target_i)
            self.iou_scores.append(iou)
            
            # Precision, Recall, Accuracy, F1
            if pred_i.sum() > 0 or target_i.sum() > 0:
                precision = precision_score(target_i, pred_i, zero_division=0)
                recall = recall_score(target_i, pred_i, zero_division=0)
                accuracy = accuracy_score(target_i, pred_i)
                f1 = f1_score(target_i, pred_i, zero_division=0)
            else:
                # If both prediction and target are empty, treat as perfect prediction
                precision = 1.0
                recall = 1.0
                accuracy = 1.0
                f1 = 1.0
            
            self.precision_scores.append(precision)
            self.recall_scores.append(recall)
            self.accuracy_scores.append(accuracy)
            self.f1_scores.append(f1)
    
    def _calculate_dice(self, pred, target, smooth=1e-8):
        """Compute Dice coefficient."""
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice
    
    def _calculate_iou(self, pred, target, smooth=1e-8):
        """Compute IoU score."""
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou
    
    def compute(self):
        """Compute average metrics across all accumulated batches."""
        if not self.dice_scores:
            return {
                'dice': 0.0,
                'iou': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0,
                'f1': 0.0
            }
        
        return {
            'dice': np.mean(self.dice_scores),
            'iou': np.mean(self.iou_scores),
            'precision': np.mean(self.precision_scores),
            'recall': np.mean(self.recall_scores),
            'accuracy': np.mean(self.accuracy_scores),
            'f1': np.mean(self.f1_scores),
            'dice_std': np.std(self.dice_scores),
            'iou_std': np.std(self.iou_scores)
        }
    
    def get_detailed_results(self):
        """Return per-sample metric arrays."""
        return {
            'dice_scores': self.dice_scores,
            'iou_scores': self.iou_scores,
            'precision_scores': self.precision_scores,
            'recall_scores': self.recall_scores,
            'accuracy_scores': self.accuracy_scores,
            'f1_scores': self.f1_scores
        }

def dice_coefficient(pred, target, threshold=0.5, smooth=1e-8):
    """
    Compute the Dice coefficient for a batch.
    Args:
        pred: predictions
        target: ground truth
        threshold: binarization threshold
        smooth: smoothing term
    """
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred)
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = target.astype(np.float32)
    
    intersection = (pred_binary * target_binary).sum()
    dice = (2. * intersection + smooth) / (pred_binary.sum() + target_binary.sum() + smooth)
    
    return dice

def iou_score(pred, target, threshold=0.5, smooth=1e-8):
    """
    Compute the IoU score for a batch.
    """
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred)
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = target.astype(np.float32)
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def calculate_metrics_batch(pred, target, threshold=0.5):
    """
    Compute all metrics for a single batch.
    """
    metrics = SegmentationMetrics(threshold=threshold)
    metrics.update(pred, target)
    return metrics.compute()

# Utilities for model comparison
def compare_predictions(pred1, pred2, target, threshold=0.5):
    """
    Compare predictions from two models on the same ground truth.
    """
    metrics1 = calculate_metrics_batch(pred1, target, threshold)
    metrics2 = calculate_metrics_batch(pred2, target, threshold)
    
    comparison = {}
    for key in metrics1.keys():
        if not key.endswith('_std'):
            comparison[f'model1_{key}'] = metrics1[key]
            comparison[f'model2_{key}'] = metrics2[key]
            comparison[f'improvement_{key}'] = metrics2[key] - metrics1[key]
    
    return comparison

