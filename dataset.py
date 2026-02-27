"""
Dataset and data loading with strong augmentation pipeline
Optimized for domain shift generalization
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class OffroadSegmentationDataset(Dataset):
    """
    Dataset for offroad semantic segmentation
    Handles image and mask loading with augmentation
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir: directory containing images
            mask_dir: directory containing masks
            transform: albumentations transform pipeline
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get list of images
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = img_name  # Assuming same name
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Map sparse labels to dense (0, 1, 2, 3, 4, 5)
        from config import Config
        dense_mask = np.zeros_like(mask, dtype=np.uint8)
        for sparse_label, dense_label in Config.SPARSE_TO_DENSE_MAPPING.items():
            dense_mask[mask == sparse_label] = dense_label
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=dense_mask)
            image = transformed['image']
            dense_mask = transformed['mask']
        
        return image, dense_mask.long()


class TestDataset(Dataset):
    """
    Dataset for test images (no masks)
    """
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir: directory containing test images
            transform: albumentations transform pipeline
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Get list of images
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(self.image_files)} test images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, img_name


def get_training_augmentation(image_size, config):
    """
    AGGRESSIVE augmentation pipeline for MAXIMUM generalization
    Optimized for domain shift robustness and compatibility
    
    Augmentations:
    - Horizontal flip (desert scenes are often symmetric)
    - Rotation (±20°, handles camera tilt variations)
    - ShiftScaleRotate (combined geometric transform)
    - Color jitter (handles lighting variations)
    - Gaussian noise (improves robustness)
    - Blur/Sharpen (handles focus variations)
    - CLAHE (adaptive histogram equalization)
    - Geometric distortions (handles terrain variations)
    """
    augmentations = [
        # Geometric augmentations
        A.HorizontalFlip(p=config.AUG_PROB),
        
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=config.SCALE_LIMIT,
            rotate_limit=config.ROTATE_LIMIT,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=config.AUG_PROB
        ),
    ]
    
    # Add advanced augmentations if enabled
    if config.USE_ADVANCED_AUGS:
        augmentations.extend([
            # Geometric distortions for terrain robustness
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=config.GRID_DISTORTION_PROB
            ),
            
            A.OpticalDistortion(
                distort_limit=0.3,
                shift_limit=0.1,
                p=config.OPTICAL_DISTORTION_PROB
            ),
            
            # CLAHE for lighting normalization
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=0.5
            ) if config.USE_CLAHE else A.NoOp(),
        ])
    
    # Color augmentations (critical for domain shift)
    augmentations.extend([
        A.ColorJitter(
            brightness=config.BRIGHTNESS_LIMIT,
            contrast=config.CONTRAST_LIMIT,
            saturation=config.SATURATION_LIMIT,
            hue=config.HUE_LIMIT,
            p=config.AUG_PROB
        ),
        
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
        
        A.HueSaturationValue(
            hue_shift_limit=25,
            sat_shift_limit=40,
            val_shift_limit=25,
            p=0.5
        ),
    ])
    
    # Add blur/sharpen if enabled
    if config.USE_ADVANCED_AUGS:
        augmentations.extend([
            A.OneOf([
                A.MotionBlur(blur_limit=config.BLUR_LIMIT, p=1.0),
                A.GaussianBlur(blur_limit=config.BLUR_LIMIT, p=1.0),
                A.MedianBlur(blur_limit=config.BLUR_LIMIT, p=1.0),
            ], p=0.3),
            
            A.Sharpen(alpha=(0.1, config.SHARPEN_LIMIT), lightness=(0.5, 1.0), p=0.3),
        ])
    
    # Additional robustness augmentations
    augmentations.extend([
        A.GaussNoise(
            var_limit=config.NOISE_VAR_LIMIT,
            p=0.4
        ),
        
        # Resize to target size
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # Normalization and tensor conversion
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    return A.Compose(augmentations)


def get_validation_augmentation(image_size):
    """
    Validation/test augmentation (no random transforms)
    Only resize and normalize
    """
    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    return val_transform


def get_tta_augmentation(image_size):
    """
    Test-Time Augmentation transforms
    Returns list of transforms for TTA
    """
    # Base transform (no augmentation)
    base_transform = get_validation_augmentation(image_size)
    
    # Horizontal flip transform
    flip_transform = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    return [base_transform, flip_transform]


def create_dataloaders(config):
    """
    Create training and validation dataloaders
    
    Args:
        config: Config object
    
    Returns:
        train_loader, val_loader
    """
    # Training augmentation
    train_transform = get_training_augmentation(config.IMAGE_SIZE, config)
    
    # Validation augmentation (no random transforms)
    val_transform = get_validation_augmentation(config.IMAGE_SIZE)
    
    # Create datasets
    train_dataset = OffroadSegmentationDataset(
        image_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        transform=train_transform
    )
    
    val_dataset = OffroadSegmentationDataset(
        image_dir=config.VAL_IMG_DIR,
        mask_dir=config.VAL_MASK_DIR,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True  # Avoid batch size mismatch in last batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def create_test_dataloader(config):
    """
    Create test dataloader
    
    Args:
        config: Config object
    
    Returns:
        test_loader
    """
    # Validation augmentation (no random transforms)
    test_transform = get_validation_augmentation(config.IMAGE_SIZE)
    
    # Create dataset
    test_dataset = TestDataset(
        image_dir=config.TEST_IMG_DIR,
        transform=test_transform
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for TTA
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"\nTest Dataset Statistics:")
    print(f"  Test samples: {len(test_dataset)}")
    
    return test_loader
