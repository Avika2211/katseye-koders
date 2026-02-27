"""
Loss functions for semantic segmentation
Hybrid loss: 0.5 * CrossEntropy + 0.5 * Dice Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    Helps with class imbalance and focuses on overlap
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets, num_classes):
        """
        Args:
            predictions: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
            num_classes: number of classes
        """
        # Convert logits to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = (targets != 255)
        
        # One-hot encode targets (only valid pixels)
        targets_valid = targets.clone()
        targets_valid[~valid_mask] = 0  # Set ignored pixels to 0 temporarily
        targets_one_hot = F.one_hot(targets_valid, num_classes=num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        
        # Calculate Dice coefficient for each class
        dice_loss = 0
        for class_idx in range(num_classes):
            pred_class = predictions[:, class_idx, :, :] * valid_mask.squeeze(1)
            target_class = targets_one_hot[:, class_idx, :, :] * valid_mask.squeeze(1)
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice_coeff)
        
        # Average over all classes
        dice_loss = dice_loss / num_classes
        return dice_loss


class HybridLoss(nn.Module):
    """
    Hybrid loss combining Cross Entropy and Dice Loss
    Formula: 0.5 * CE + 0.5 * Dice
    
    Benefits:
    - CE: pixel-wise classification accuracy
    - Dice: regional overlap, handles class imbalance
    """
    def __init__(self, class_weights=None, ce_weight=0.5, dice_weight=0.5, num_classes=10, ignore_index=255):
        super(HybridLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Cross Entropy with class weights
        if class_weights is not None:
            if class_weights is not None:
                if not isinstance(class_weights, torch.Tensor):
                    class_weights = torch.tensor(class_weights, dtype=torch.float32)
                cclass_weights = class_weights.float()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index, reduction='mean')
        
        # Dice Loss
        self.dice_loss = DiceLoss(smooth=1.0)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) - model logits
            targets: (B, H, W) - ground truth class indices
        
        Returns:
            Combined loss value
        """
        # Cross Entropy Loss
        ce = self.ce_loss(predictions, targets)
        
        # Dice Loss
        dice = self.dice_loss(predictions, targets, self.num_classes)
        
        # Weighted combination
        total_loss = self.ce_weight * ce + self.dice_weight * dice
        
        return total_loss, ce, dice


def compute_class_weights(train_loader, num_classes, device):
    """
    Compute class weights based on pixel frequency in training set
    Inverse frequency weighting to handle class imbalance
    
    Args:
        train_loader: DataLoader for training data
        num_classes: number of classes
        device: torch device
    
    Returns:
        class_weights: tensor of shape (num_classes,)
    """
    print("Computing class weights from training data...")
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        for class_id in range(num_classes):
            class_counts[class_id] += (masks == class_id).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(train_loader)} batches")
    
    # Compute weights (inverse frequency)
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts)
    
    # Normalize weights to prevent extreme values
    class_weights = class_weights / class_weights.sum() * num_classes
    
    # Clip extreme weights
    class_weights = torch.clamp(class_weights, min=0.1, max=10.0)
    
    print("\nClass weights computed:")
    for i, weight in enumerate(class_weights):
        print(f"  Class {i}: {weight:.4f} (pixels: {int(class_counts[i])})")
    
    return class_weights.to(device)
