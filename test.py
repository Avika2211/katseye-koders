"""
Test script for Offroad Semantic Segmentation Hackathon
Includes Test-Time Augmentation (TTA) for improved predictions
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from dataset import create_test_dataloader, get_validation_augmentation
from model import create_model
from metrics import SegmentationMetrics


def dense_to_sparse_labels(dense_mask, config):
    """
    Map dense predictions [0, 1, 2, 3, 4, 5] back to sparse labels [0, 1, 2, 3, 27, 39]
    
    Args:
        dense_mask: (H, W) array with dense labels
        config: Config object
    
    Returns:
        sparse_mask: (H, W) array with sparse labels
    """
    # Create reverse mapping
    dense_to_sparse = {v: k for k, v in config.SPARSE_TO_DENSE_MAPPING.items() if k != 255}
    
    sparse_mask = np.zeros_like(dense_mask, dtype=np.uint8)
    for dense_label, sparse_label in dense_to_sparse.items():
        sparse_mask[dense_mask == dense_label] = sparse_label
    
    return sparse_mask


def tta_predict(model, image, device, config):
    """
    Test-Time Augmentation prediction
    Averages predictions from original and horizontally flipped image
    
    Args:
        model: trained model
        image: input image tensor (1, C, H, W)
        device: torch device
        config: Config object
    
    Returns:
        prediction: averaged prediction (H, W)
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original prediction
        if config.USE_AMP:
            with autocast():
                output = model(image)
        else:
            output = model(image)
        
        prob = F.softmax(output, dim=1)
        predictions.append(prob)
        
        # Horizontal flip prediction
        image_flipped = torch.flip(image, dims=[3])  # Flip width dimension
        
        if config.USE_AMP:
            with autocast():
                output_flipped = model(image_flipped)
        else:
            output_flipped = model(image_flipped)
        
        prob_flipped = F.softmax(output_flipped, dim=1)
        prob_flipped = torch.flip(prob_flipped, dims=[3])  # Flip back
        predictions.append(prob_flipped)
    
    # Average predictions
    avg_prob = torch.mean(torch.stack(predictions), dim=0)
    prediction = torch.argmax(avg_prob, dim=1)
    
    return prediction.squeeze(0)  # (H, W)


def simple_predict(model, image, device, config):
    """
    Simple prediction without TTA
    
    Args:
        model: trained model
        image: input image tensor (1, C, H, W)
        device: torch device
        config: Config object
    
    Returns:
        prediction: prediction (H, W)
    """
    model.eval()
    
    with torch.no_grad():
        if config.USE_AMP:
            with autocast():
                output = model(image)
        else:
            output = model(image)
        
        prediction = torch.argmax(output, dim=1)
    
    return prediction.squeeze(0)  # (H, W)


def colorize_mask(mask, num_classes):
    """
    Colorize segmentation mask for visualization
    
    Args:
        mask: (H, W) numpy array with class indices
        num_classes: number of classes
    
    Returns:
        colored_mask: (H, W, 3) RGB image
    """
    # Create color palette
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
    
    # Ensure we have enough colors
    if num_classes > len(palette):
        # Generate random colors for additional classes
        extra_colors = np.random.randint(0, 255, size=(num_classes - len(palette), 3), dtype=np.uint8)
        palette = np.vstack([palette, extra_colors])
    
    # Map mask to colors
    colored_mask = palette[mask]
    
    return colored_mask


def test(config, checkpoint_path=None):
    """
    Main testing function
    
    Args:
        config: Config object
        checkpoint_path: path to model checkpoint (if None, uses best_model.pth)
    """
    
    # Initialize device
    device = config.DEVICE
    print(f"\nUsing device: {device}")
    
    # Create test dataloader
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)
    test_loader = create_test_dataloader(config)
    
    # Create model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model = create_model(config)
    model = model.to(device)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation mIoU: {checkpoint['val_miou']:.4f}")
    
    # Set model to eval mode
    model.eval()
    
    # Create output directory for predictions
    pred_dir = config.PREDICTIONS_DIR
    pred_masks_dir = os.path.join(pred_dir, 'masks')
    pred_colored_dir = os.path.join(pred_dir, 'colored')
    os.makedirs(pred_masks_dir, exist_ok=True)
    os.makedirs(pred_colored_dir, exist_ok=True)
    
    # Run inference
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80)
    print(f"Test-Time Augmentation (TTA): {'Enabled' if config.USE_TTA else 'Disabled'}")
    
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (image, img_name) in enumerate(tqdm(test_loader, desc='Testing')):
            image = image.to(device)
            
            # Predict (returns dense labels)
            if config.USE_TTA:
                prediction = tta_predict(model, image, device, config)
            else:
                prediction = simple_predict(model, image, device, config)
            
            # Convert to numpy (dense labels)
            prediction_dense = prediction.cpu().numpy().astype(np.uint8)
            
            # Convert dense labels back to sparse labels for output
            prediction_sparse = dense_to_sparse_labels(prediction_dense, config)
            all_predictions.append(prediction_dense)  # Store dense for metrics
            
            # Save prediction mask (sparse labels for submission)
            mask_filename = img_name[0].replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(pred_masks_dir, mask_filename)
            cv2.imwrite(mask_path, prediction_sparse)
            
            # Save colored visualization (using dense labels for consistent colors)
            colored_mask = colorize_mask(prediction_dense, config.NUM_CLASSES)
            colored_path = os.path.join(pred_colored_dir, mask_filename)
            cv2.imwrite(colored_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    
    print(f"\nPredictions saved to:")
    print(f"  Masks: {pred_masks_dir}")
    print(f"  Colored: {pred_colored_dir}")
    
    # If ground truth exists, compute metrics
    print("\n" + "="*80)
    print("CHECKING FOR GROUND TRUTH")
    print("="*80)
    
    # Check if test masks directory exists (for evaluation)
    test_mask_dir = config.TEST_MASK_DIR
    
    if os.path.exists(test_mask_dir):
        print(f"Ground truth found at: {test_mask_dir}")
        print("Computing evaluation metrics...")
        
        # Load ground truth and compute metrics
        metrics = SegmentationMetrics(config.NUM_CLASSES, config.CLASSES)
        
        test_files = sorted([f for f in os.listdir(config.TEST_IMG_DIR) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for idx, img_name in enumerate(tqdm(test_files, desc='Computing metrics')):
            # Load ground truth (sparse labels)
            mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(test_mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                gt_mask_sparse = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                # Convert sparse GT to dense for fair comparison
                gt_mask_dense = np.zeros_like(gt_mask_sparse, dtype=np.uint8)
                for sparse_label, dense_label in config.SPARSE_TO_DENSE_MAPPING.items():
                    if sparse_label != 255:
                        gt_mask_dense[gt_mask_sparse == sparse_label] = dense_label
                
                pred_mask_dense = all_predictions[idx]
                
                # Update metrics
                metrics.update(
                    pred_mask_dense.reshape(1, -1),
                    gt_mask_dense.reshape(1, -1)
                )
        
        # Print results
        metrics.print_results()
        
        # Save confusion matrix
        cm_path = os.path.join(pred_dir, 'test_confusion_matrix.png')
        metrics.plot_confusion_matrix(save_path=cm_path, normalize=True)
        
        # Save metrics to file
        results = metrics.get_results()
        metrics_path = os.path.join(pred_dir, 'test_metrics.txt')
        
        with open(metrics_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TEST SET EVALUATION METRICS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Mean IoU: {results['mean_iou']:.4f}\n\n")
            f.write("Per-class metrics:\n")
            f.write(f"{'Class':<20} {'IoU':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
            f.write("-"*80 + "\n")
            
            for i in range(config.NUM_CLASSES):
                class_name = config.CLASSES[i] if i < len(config.CLASSES) else f"Class {i}"
                f.write(f"{class_name:<20} "
                       f"{results['iou_per_class'][i]:<10.4f} "
                       f"{results['precision_per_class'][i]:<10.4f} "
                       f"{results['recall_per_class'][i]:<10.4f} "
                       f"{results['f1_per_class'][i]:<10.4f}\n")
        
        print(f"\nMetrics saved to: {metrics_path}")
    
    else:
        print("No ground truth found. Skipping metrics computation.")
        print("Predictions have been saved for submission.")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)


def test_with_validation_set(config, checkpoint_path=None):
    """
    Test on validation set (for debugging/verification)
    
    Args:
        config: Config object
        checkpoint_path: path to model checkpoint
    """
    from dataset import OffroadSegmentationDataset
    from torch.utils.data import DataLoader
    
    print("\n" + "="*80)
    print("TESTING ON VALIDATION SET")
    print("="*80)
    
    # Initialize device
    device = config.DEVICE
    
    # Create validation dataset
    val_transform = get_validation_augmentation(config.IMAGE_SIZE)
    val_dataset = OffroadSegmentationDataset(
        image_dir=config.VAL_IMG_DIR,
        mask_dir=config.VAL_MASK_DIR,
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Compute metrics
    metrics = SegmentationMetrics(config.NUM_CLASSES, config.CLASSES)
    
    model.eval()
    with torch.no_grad():
        for image, mask in tqdm(val_loader, desc='Validating'):
            image = image.to(device)
            
            if config.USE_TTA:
                prediction = tta_predict(model, image, device, config)
            else:
                prediction = simple_predict(model, image, device, config)
            
            # Update metrics
            metrics.update(
                prediction.cpu().numpy().reshape(1, -1),
                mask.numpy().reshape(1, -1)
            )
    
    # Print results
    metrics.print_results()


if __name__ == "__main__":
    config = Config()
    
    # Run test on test images
    test(config)
    
    # Optionally test on validation set for verification
    # test_with_validation_set(config)
