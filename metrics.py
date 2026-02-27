"""
Evaluation metrics for semantic segmentation
- Per-class IoU
- Mean IoU
- Confusion Matrix
- Precision and Recall per class
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SegmentationMetrics:
    """
    Comprehensive metrics for semantic segmentation evaluation
    """
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, predictions, targets):
        """
        Update confusion matrix with batch predictions
        
        Args:
            predictions: (B, H, W) - predicted class indices
            targets: (B, H, W) - ground truth class indices
        """
        # Flatten arrays
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Update confusion matrix
        mask = (targets >= 0) & (targets < self.num_classes)
        cm = confusion_matrix(
            targets[mask],
            predictions[mask],
            labels=np.arange(self.num_classes)
        )
        self.confusion_matrix += cm
    
    def compute_iou(self):
        """
        Compute Intersection over Union (IoU) for each class
        
        Returns:
            iou_per_class: array of shape (num_classes,)
            mean_iou: scalar
        """
        iou_per_class = np.zeros(self.num_classes, dtype=np.float32)
        
        for i in range(self.num_classes):
            true_positive = self.confusion_matrix[i, i]
            false_positive = self.confusion_matrix[:, i].sum() - true_positive
            false_negative = self.confusion_matrix[i, :].sum() - true_positive
            
            denominator = true_positive + false_positive + false_negative
            
            if denominator > 0:
                iou_per_class[i] = true_positive / denominator
            else:
                iou_per_class[i] = 0.0
        
        # Mean IoU (average over all classes)
        mean_iou = np.mean(iou_per_class)
        
        return iou_per_class, mean_iou
    
    def compute_precision_recall(self):
        """
        Compute precision and recall for each class
        
        Returns:
            precision_per_class: array of shape (num_classes,)
            recall_per_class: array of shape (num_classes,)
        """
        precision_per_class = np.zeros(self.num_classes, dtype=np.float32)
        recall_per_class = np.zeros(self.num_classes, dtype=np.float32)
        
        for i in range(self.num_classes):
            true_positive = self.confusion_matrix[i, i]
            false_positive = self.confusion_matrix[:, i].sum() - true_positive
            false_negative = self.confusion_matrix[i, :].sum() - true_positive
            
            # Precision: TP / (TP + FP)
            if (true_positive + false_positive) > 0:
                precision_per_class[i] = true_positive / (true_positive + false_positive)
            else:
                precision_per_class[i] = 0.0
            
            # Recall: TP / (TP + FN)
            if (true_positive + false_negative) > 0:
                recall_per_class[i] = true_positive / (true_positive + false_negative)
            else:
                recall_per_class[i] = 0.0
        
        return precision_per_class, recall_per_class
    
    def compute_f1_score(self):
        """
        Compute F1 score for each class
        
        Returns:
            f1_per_class: array of shape (num_classes,)
        """
        precision, recall = self.compute_precision_recall()
        f1_per_class = np.zeros(self.num_classes, dtype=np.float32)
        
        for i in range(self.num_classes):
            if (precision[i] + recall[i]) > 0:
                f1_per_class[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            else:
                f1_per_class[i] = 0.0
        
        return f1_per_class
    
    def get_results(self):
        """
        Get all computed metrics
        
        Returns:
            results: dictionary with all metrics
        """
        iou_per_class, mean_iou = self.compute_iou()
        precision_per_class, recall_per_class = self.compute_precision_recall()
        f1_per_class = self.compute_f1_score()
        
        results = {
            'iou_per_class': iou_per_class,
            'mean_iou': mean_iou,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': self.confusion_matrix
        }
        
        return results
    
    def print_results(self):
        """Print formatted results"""
        results = self.get_results()
        
        print("\n" + "="*80)
        print("SEGMENTATION METRICS")
        print("="*80)
        
        print(f"\nMean IoU: {results['mean_iou']:.4f}")
        
        print("\nPer-class metrics:")
        print(f"{'Class':<20} {'IoU':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-"*80)
        
        for i in range(self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
            print(f"{class_name:<20} "
                  f"{results['iou_per_class'][i]:<10.4f} "
                  f"{results['precision_per_class'][i]:<10.4f} "
                  f"{results['recall_per_class'][i]:<10.4f} "
                  f"{results['f1_per_class'][i]:<10.4f}")
        
        print("="*80 + "\n")
    
    def plot_confusion_matrix(self, save_path=None, normalize=True):
        """
        Plot confusion matrix
        
        Args:
            save_path: path to save plot
            normalize: whether to normalize by row (recall)
        """
        cm = self.confusion_matrix.copy()
        
        if normalize:
            cm = cm.astype('float')
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            cm = cm / row_sums
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
        )
        
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()


def compute_batch_iou(predictions, targets, num_classes):
    """
    Fast IoU computation for a single batch (used during training)
    
    Args:
        predictions: (B, H, W) - predicted class indices
        targets: (B, H, W) - ground truth class indices
        num_classes: number of classes
    
    Returns:
        mean_iou: scalar
    """
    ious = []
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou.item())
        else:
            ious.append(0.0)
    
    return np.mean(ious)
