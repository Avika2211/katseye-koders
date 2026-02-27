"""
Configuration file for Offroad Semantic Segmentation Hackathon
Author: Competition Solution
Model: SegFormer-B3 (chosen for superior generalization on domain shift tasks)
"""

import torch
import os

class Config:
    # Paths (updated for your data structure)
    DATA_ROOT = "data"
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train/images")
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train/segmentation")
    VAL_IMG_DIR = os.path.join(DATA_ROOT, "val/images")
    VAL_MASK_DIR = os.path.join(DATA_ROOT, "val/segmentation")
    TEST_IMG_DIR = os.path.join(DATA_ROOT, "test/images")
    TEST_MASK_DIR = os.path.join(DATA_ROOT, "test/segmentation")  # For evaluation if available
    
    # Output directories
    OUTPUT_DIR = "runs"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
    
    # Model Architecture
    # OPTIMIZED FOR WINNING: Using larger model + ensemble-ready settings
    # SegFormer chosen over DeepLabV3+ for:
    # 1. Better generalization on domain shift (hierarchical transformer)
    # 2. No positional encoding = better at different scales
    # 3. Lightweight MLP decoder = less overfitting
    # 4. State-of-the-art on outdoor scenes
    MODEL_NAME = "segformer"  # Options: "segformer", "deeplabv3plus"
    BACKBONE = "mit_b5"  # UPGRADED: mit_b5 (largest) for maximum capacity
    ENCODER_WEIGHTS = "imagenet"  # Pretrained weights
    
    # Label mapping (sparse labels to dense)
    # Your masks have values: [0, 1, 2, 3, 27, 39]
    # We need to map these to continuous indices [0, 1, 2, 3, 4, 5]
    SPARSE_TO_DENSE_MAPPING = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        27: 4,
        39: 5,
        255: 255  # Ignore label (if exists)
    }
    
    # Classes (based on your sparse labels)
    # You should update these names based on what each class represents
    CLASSES = [
        "Class_0",   # Sparse label: 0
        "Class_1",   # Sparse label: 1
        "Class_2",   # Sparse label: 2
        "Class_3",   # Sparse label: 3
        "Class_27",  # Sparse label: 27
        "Class_39",  # Sparse label: 39
    ]
    NUM_CLASSES = len(CLASSES)
    IGNORE_INDEX = 255  # Label to ignore in loss computation
    
    # Image specifications
    IMAGE_SIZE = (544, 960)  # (height, width) - matches your data
    IN_CHANNELS = 3
    
    # Training hyperparameters - OPTIMIZED FOR WINNING
    BATCH_SIZE = 2  # Reduced for larger model (mit_b5)
    NUM_EPOCHS = 200  # INCREASED: Train longer for better convergence
    LEARNING_RATE = 4e-4  # OPTIMIZED: Slightly lower for stability with larger model
    WEIGHT_DECAY = 0.02  # INCREASED: More regularization to prevent overfitting
    
    # Loss weights - OPTIMIZED
    CE_WEIGHT = 0.4  # Slightly reduced CE weight
    DICE_WEIGHT = 0.6  # INCREASED: Dice loss better for segmentation IoU
    
    # Optimizer
    OPTIMIZER = "adamw"  # AdamW better for transformers
    
    # Learning rate scheduler - OPTIMIZED
    SCHEDULER = "cosine"  # Options: "cosine", "plateau"
    MIN_LR = 1e-7  # Lower minimum for fine-tuning
    T_MAX = NUM_EPOCHS  # For cosine annealing
    WARMUP_EPOCHS = 5  # Warmup period for stable training
    PATIENCE = 15  # For ReduceLROnPlateau
    FACTOR = 0.5  # For ReduceLROnPlateau
    
    # Early stopping - OPTIMIZED
    EARLY_STOPPING_PATIENCE = 30  # INCREASED: More patience for longer training
    
    # Mixed precision training
    USE_AMP = True  # Automatic Mixed Precision (fp16)
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Test-Time Augmentation
    USE_TTA = True
    
    # Random seed for reproducibility
    SEED = 42
    
    # Checkpointing
    SAVE_BEST_ONLY = True
    MONITOR_METRIC = "val_miou"  # Metric to monitor for best model
    
    # Augmentation probabilities - MAXIMALLY AGGRESSIVE FOR GENERALIZATION
    AUG_PROB = 0.7  # INCREASED: More aggressive augmentation
    ROTATE_LIMIT = 20  # INCREASED: More rotation variance
    BRIGHTNESS_LIMIT = 0.3  # INCREASED: Handle extreme lighting
    CONTRAST_LIMIT = 0.3  # INCREASED: Handle contrast variations
    SATURATION_LIMIT = 0.3  # INCREASED: Handle color shifts
    HUE_LIMIT = 0.15  # INCREASED: More color variation
    SCALE_LIMIT = 0.15  # INCREASED: More scale variation
    NOISE_VAR_LIMIT = (10, 70)  # INCREASED: More noise robustness
    
    # Advanced augmentation settings
    USE_ADVANCED_AUGS = True  # Enable additional augmentations
    BLUR_LIMIT = 5  # Random blur for robustness
    SHARPEN_LIMIT = 0.3  # Random sharpening
    USE_CLAHE = True  # Adaptive histogram equalization
    GRID_DISTORTION_PROB = 0.3  # Elastic deformations
    OPTICAL_DISTORTION_PROB = 0.3  # Lens distortions
    
    @staticmethod
    def create_dirs():
        """Create necessary directories"""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
