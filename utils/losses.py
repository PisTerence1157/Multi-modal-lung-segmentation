import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: predicted tensor [B, 1, H, W]
            target: ground truth tensor [B, 1, H, W]
        """
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice

class IoULoss(nn.Module):
    """Intersection over Union Loss (IoU Loss)"""
    
    def __init__(self, smooth=1e-8):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou

class CombinedLoss(nn.Module):
    """Composite loss: Dice + BCE"""
    
    def __init__(self, dice_weight=0.7, bce_weight=0.3, pos_weight=None):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.dice_loss = DiceLoss()
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduce=False)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduce:
            return focal_loss.mean()
        else:
            return focal_loss

def get_loss_function(config, pos_weight=None):
    """Return the loss function specified by config."""
    loss_type = config['loss']['type']
    
    if loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'bce':
        if pos_weight is not None:
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            return nn.BCEWithLogitsLoss()
    elif loss_type == 'dice_bce':
        return CombinedLoss(
            dice_weight=config['loss']['dice_weight'],
            bce_weight=config['loss']['bce_weight'],
            pos_weight=pos_weight
        )
    elif loss_type == 'focal':
        return FocalLoss()
    elif loss_type == 'iou':
        return IoULoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Utility for comparing loss functions
def calculate_all_losses(pred, target, pos_weight=None):
    """Compute all supported loss values for comparison."""
    results = {}
    
    dice_loss = DiceLoss()
    results['dice'] = dice_loss(pred, target).item()
    
    if pos_weight is not None:
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    else:
        bce_loss = nn.BCEWithLogitsLoss()
    results['bce'] = bce_loss(pred, target).item()
    
    iou_loss = IoULoss()
    results['iou'] = iou_loss(pred, target).item()
    
    focal_loss = FocalLoss()
    results['focal'] = focal_loss(pred, target).item()
    
    combined_loss = CombinedLoss(pos_weight=pos_weight)
    results['combined'] = combined_loss(pred, target).item()
    
    return results
