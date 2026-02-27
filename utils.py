"""
Utility functions for visualization and analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from config import Config


def visualize_augmentations(dataset, num_samples=4):
    """
    Visualize augmented samples from dataset
    
    Args:
        dataset: Dataset object
        num_samples: number of samples to visualize
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        image, mask = dataset[i]
        
        # Convert tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * std + mean) * 255
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        else:
            image_np = image
        
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        
        # Create colored mask
        colored_mask = colorize_mask(mask_np, len(Config.CLASSES))
        
        # Plot
        if num_samples == 1:
            axes[0].imshow(image_np)
            axes[0].set_title('Image')
            axes[0].axis('off')
            
            axes[1].imshow(mask_np, cmap='tab10')
            axes[1].set_title('Mask (Grayscale)')
            axes[1].axis('off')
            
            axes[2].imshow(colored_mask)
            axes[2].set_title('Mask (Colored)')
            axes[2].axis('off')
        else:
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f'Sample {i+1}: Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_np, cmap='tab10')
            axes[i, 1].set_title(f'Sample {i+1}: Mask')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(colored_mask)
            axes[i, 2].set_title(f'Sample {i+1}: Colored')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Augmentation samples saved to augmentation_samples.png")


def colorize_mask(mask, num_classes):
    """
    Colorize segmentation mask for visualization
    """
    palette = np.array([
        [128, 64, 128],   # Trees - purple
        [34, 139, 34],    # Lush Bushes - forest green
        [255, 215, 0],    # Dry Grass - gold
        [210, 180, 140],  # Dry Bushes - tan
        [139, 69, 19],    # Ground Clutter - brown
        [255, 105, 180],  # Flowers - pink
        [160, 82, 45],    # Logs - sienna
        [128, 128, 128],  # Rocks - gray
        [244, 164, 96],   # Landscape - sandy brown
        [135, 206, 235]   # Sky - sky blue
    ], dtype=np.uint8)
    
    if num_classes > len(palette):
        extra_colors = np.random.randint(0, 255, size=(num_classes - len(palette), 3), dtype=np.uint8)
        palette = np.vstack([palette, extra_colors])
    
    colored_mask = palette[mask]
    return colored_mask


def visualize_predictions(image_path, mask_path, pred_path, save_path=None):
    """
    Visualize image, ground truth, and prediction side by side
    
    Args:
        image_path: path to original image
        mask_path: path to ground truth mask
        pred_path: path to predicted mask
        save_path: path to save visualization
    """
    # Load images
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    
    # Colorize masks
    num_classes = len(Config.CLASSES)
    mask_colored = colorize_mask(mask, num_classes)
    pred_colored = colorize_mask(pred, num_classes)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask_colored)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_colored)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Difference map (errors)
    diff = (mask != pred).astype(np.uint8) * 255
    axes[3].imshow(diff, cmap='Reds')
    axes[3].set_title('Errors (Red = Wrong)')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_class_distribution(mask_dir, num_classes):
    """
    Analyze class distribution in dataset
    
    Args:
        mask_dir: directory containing masks
        num_classes: number of classes
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)
    
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    print(f"Analyzing {len(mask_files)} masks...")
    
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        for class_id in range(num_classes):
            class_counts[class_id] += (mask == class_id).sum()
    
    total_pixels = class_counts.sum()
    
    print("\nClass Distribution:")
    print(f"{'Class':<20} {'Pixels':<15} {'Percentage':<10}")
    print("-" * 50)
    
    for i in range(num_classes):
        class_name = Config.CLASSES[i] if i < len(Config.CLASSES) else f"Class {i}"
        percentage = (class_counts[i] / total_pixels) * 100
        print(f"{class_name:<20} {class_counts[i]:<15} {percentage:>8.2f}%")
    
    # Plot distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    class_names = [Config.CLASSES[i] if i < len(Config.CLASSES) else f"Class {i}" 
                   for i in range(num_classes)]
    percentages = [(class_counts[i] / total_pixels) * 100 for i in range(num_classes)]
    
    bars = ax.bar(class_names, percentages)
    ax.set_xlabel('Class')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Class Distribution')
    ax.tick_params(axis='x', rotation=45)
    
    # Color bars
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nClass distribution plot saved to class_distribution.png")


def plot_training_history(history_path):
    """
    Plot training history from JSON file
    
    Args:
        history_path: path to training_history.json
    """
    import json
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # mIoU
    axes[0, 1].plot(epochs, history['val_miou'], label='Val mIoU', marker='o', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].set_title('Validation mIoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(epochs, history['learning_rates'], marker='o', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss vs mIoU
    axes[1, 1].scatter(history['val_loss'], history['val_miou'], alpha=0.6)
    axes[1, 1].set_xlabel('Validation Loss')
    axes[1, 1].set_ylabel('Validation mIoU')
    axes[1, 1].set_title('Loss vs mIoU')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Training history plot saved to training_history.png")


if __name__ == "__main__":
    # Example usage
    print("Utility functions loaded successfully!")
    
    # Visualize augmentations
    from dataset import OffroadSegmentationDataset, get_training_augmentation
    
    config = Config()
    
    if os.path.exists(config.TRAIN_IMG_DIR):
        print("\nVisualizing augmentations...")
        train_transform = get_training_augmentation(config.IMAGE_SIZE, config)
        train_dataset = OffroadSegmentationDataset(
            config.TRAIN_IMG_DIR,
            config.TRAIN_MASK_DIR,
            transform=train_transform
        )
        visualize_augmentations(train_dataset, num_samples=4)
        
        print("\nAnalyzing class distribution...")
        analyze_class_distribution(config.TRAIN_MASK_DIR, config.NUM_CLASSES)
    
    # Plot training history if available
    history_path = os.path.join(config.LOG_DIR, 'training_history.json')
    if os.path.exists(history_path):
        print("\nPlotting training history...")
        plot_training_history(history_path)
