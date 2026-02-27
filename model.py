"""
Model architecture: SegFormer
Chosen for superior generalization on domain shift tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def create_model(config):
    """
    Create segmentation model based on config
    
    SegFormer (default):
    - Hierarchical Transformer encoder (no positional encoding)
    - Lightweight MLP decoder
    - Better generalization on domain shift
    - State-of-the-art on outdoor scenes
    
    DeepLabV3+ (alternative):
    - ResNet backbone with atrous convolutions
    - ASPP for multi-scale context
    - Proven performance on segmentation
    
    Args:
        config: Config object
    
    Returns:
        model: PyTorch model
    """
    if config.MODEL_NAME == "segformer":
        print("Creating SegFormer model...")
        print("Rationale: SegFormer chosen for:")
        print("  1. Superior generalization on domain shift (no positional encoding)")
        print("  2. Hierarchical transformer captures both local and global context")
        print("  3. Lightweight decoder prevents overfitting")
        print("  4. State-of-the-art performance on outdoor scenes")
        
        model = smp.Segformer(
            encoder_name=config.BACKBONE,  # mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=config.IN_CHANNELS,
            classes=config.NUM_CLASSES
        )
        
    elif config.MODEL_NAME == "deeplabv3plus":
        print("Creating DeepLabV3+ model...")
        print("Rationale: DeepLabV3+ chosen for:")
        print("  1. ResNet backbone with strong feature extraction")
        print("  2. ASPP module for multi-scale context")
        print("  3. Proven performance on semantic segmentation")
        
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101",  # resnet50 or resnet101
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=config.IN_CHANNELS,
            classes=config.NUM_CLASSES
        )
    
    else:
        raise ValueError(f"Unknown model name: {config.MODEL_NAME}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


class ModelEMA:
    """
    Exponential Moving Average of model parameters
    Improves generalization by smoothing parameter updates
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
