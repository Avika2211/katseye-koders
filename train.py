"""
Training script for Offroad Semantic Segmentation Hackathon
Complete training pipeline with all required features
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import json

from config import Config
from dataset import create_dataloaders
from model import create_model, ModelEMA
from loss import HybridLoss, compute_class_weights
from metrics import SegmentationMetrics, compute_batch_iou


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=20, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_metric):
        score = val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, config):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    running_ce_loss = 0.0
    running_dice_loss = 0.0
    running_iou = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.NUM_EPOCHS} [Train]')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config.USE_AMP:
            with autocast():
                outputs = model(images)
                loss, ce_loss, dice_loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, ce_loss, dice_loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            batch_iou = compute_batch_iou(predictions, masks, config.NUM_CLASSES)
        
        # Update running metrics
        running_loss += loss.item()
        running_ce_loss += ce_loss.item()
        running_dice_loss += dice_loss.item()
        running_iou += batch_iou
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{batch_iou:.4f}'
        })
    
    # Average metrics
    avg_loss = running_loss / len(train_loader)
    avg_ce_loss = running_ce_loss / len(train_loader)
    avg_dice_loss = running_dice_loss / len(train_loader)
    avg_iou = running_iou / len(train_loader)
    
    return avg_loss, avg_ce_loss, avg_dice_loss, avg_iou


@torch.no_grad()
def validate_epoch(model, val_loader, criterion, device, epoch, config):
    """Validate for one epoch"""
    model.eval()
    
    running_loss = 0.0
    running_ce_loss = 0.0
    running_dice_loss = 0.0
    
    # Initialize metrics
    metrics = SegmentationMetrics(config.NUM_CLASSES, config.CLASSES)
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{config.NUM_EPOCHS} [Val]')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        if config.USE_AMP:
            with autocast():
                outputs = model(images)
                loss, ce_loss, dice_loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss, ce_loss, dice_loss = criterion(outputs, masks)
        
        # Get predictions
        predictions = torch.argmax(outputs, dim=1)
        
        # Update metrics
        metrics.update(
            predictions.cpu().numpy(),
            masks.cpu().numpy()
        )
        
        # Update running losses
        running_loss += loss.item()
        running_ce_loss += ce_loss.item()
        running_dice_loss += dice_loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Average losses
    avg_loss = running_loss / len(val_loader)
    avg_ce_loss = running_ce_loss / len(val_loader)
    avg_dice_loss = running_dice_loss / len(val_loader)
    
    # Compute final metrics
    results = metrics.get_results()
    
    return avg_loss, avg_ce_loss, avg_dice_loss, results


def save_checkpoint(model, optimizer, scheduler, epoch, val_miou, checkpoint_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_miou': val_miou
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def train(config):
    """Main training function"""
    
    # Set random seed
    set_seed(config.SEED)
    
    # Create output directories
    config.create_dirs()
    
    # Initialize device
    device = config.DEVICE
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create dataloaders
    print("\n" + "="*80)
    print("CREATING DATALOADERS")
    print("="*80)
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    model = create_model(config)
    model = model.to(device)
    
    # Compute class weights
    print("\n" + "="*80)
    print("COMPUTING CLASS WEIGHTS")
    print("="*80)
    class_weights = compute_class_weights(train_loader, config.NUM_CLASSES, device)
    
    # Create loss function
    criterion = HybridLoss(
        class_weights=class_weights,
        ce_weight=config.CE_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
        num_classes=config.NUM_CLASSES,
        ignore_index=config.IGNORE_INDEX
    )
    
    # Create optimizer
    if config.OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    
    # Create learning rate scheduler
    if config.SCHEDULER == "cosine":
        # Cosine annealing with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_MAX - config.WARMUP_EPOCHS,
            eta_min=config.MIN_LR
        )
        
        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=config.WARMUP_EPOCHS
        )
        
        # Combine warmup + cosine
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[config.WARMUP_EPOCHS]
        )
    else:  # plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.FACTOR,
            patience=config.PATIENCE,
            verbose=True
        )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    # Training tracking
    best_val_miou = 0.0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': [],
        'learning_rates': []
    }
    
    # Start training
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Total epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_ce, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch, config
        )
        
        # Validate
        val_loss, val_ce, val_dice, val_results = validate_epoch(
            model, val_loader, criterion, device, epoch, config
        )
        
        val_miou = val_results['mean_iou']
        
        # Update learning rate
        if config.SCHEDULER == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_miou)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/train', train_iou, epoch)
        writer.add_scalar('IoU/val', val_miou, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Log per-class IoU
        for i, class_name in enumerate(config.CLASSES):
            writer.add_scalar(f'IoU_Class/{class_name}', val_results['iou_per_class'][i], epoch)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, Dice: {train_dice:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (CE: {val_ce:.4f}, Dice: {val_dice:.4f})")
        print(f"  Train IoU: {train_iou:.4f}")
        print(f"  Val mIoU: {val_miou:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        
        # Save training history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_miou'].append(val_miou)
        training_history['learning_rates'].append(current_lr)
        
        # Save best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_miou, checkpoint_path)
            print(f"  *** New best mIoU: {val_miou:.4f} ***")
        
        # Save latest model
        latest_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'latest_model.pth')
        save_checkpoint(model, optimizer, scheduler, epoch, val_miou, latest_checkpoint_path)
        
        # Early stopping check
        if early_stopping(val_miou):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
        
        print("-" * 80)
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Best validation mIoU: {best_val_miou:.4f}")
    
    # Save training history
    history_path = os.path.join(config.LOG_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=4)
    print(f"Training history saved to {history_path}")
    
    # Final validation with best model
    print("\n" + "="*80)
    print("FINAL VALIDATION WITH BEST MODEL")
    print("="*80)
    
    # Load best model
    best_checkpoint = torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Validate
    val_loss, val_ce, val_dice, val_results = validate_epoch(
        model, val_loader, criterion, device, config.NUM_EPOCHS, config
    )
    
    # Print detailed results
    metrics = SegmentationMetrics(config.NUM_CLASSES, config.CLASSES)
    metrics.confusion_matrix = val_results['confusion_matrix']
    metrics.print_results()
    
    # Plot confusion matrix
    cm_path = os.path.join(config.LOG_DIR, 'confusion_matrix.png')
    metrics.plot_confusion_matrix(save_path=cm_path, normalize=True)
    
    writer.close()
    
    print("\nAll done! Model ready for testing.")


if __name__ == "__main__":
    config = Config()
    train(config)
