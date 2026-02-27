"""
Setup verification script
Checks that all dependencies are installed and data is properly structured
"""

import sys
import os


def check_imports():
    """Check if all required packages are installed"""
    print("="*80)
    print("CHECKING IMPORTS")
    print("="*80)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'segmentation_models_pytorch': 'Segmentation Models PyTorch',
        'albumentations': 'Albumentations',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm',
        'tensorboard': 'TensorBoard',
        'PIL': 'Pillow'
    }
    
    all_good = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name:<30} [OK]")
        except ImportError:
            print(f"✗ {name:<30} [MISSING]")
            all_good = False
    
    if all_good:
        print("\n✓ All required packages are installed!")
    else:
        print("\n✗ Some packages are missing. Install with:")
        print("  pip install -r requirements.txt")
        return False
    
    # Check PyTorch CUDA
    import torch
    if torch.cuda.is_available():
        print(f"\n✓ CUDA is available!")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("\n⚠ CUDA is not available. Training will use CPU (much slower).")
    
    return True


def check_data_structure():
    """Check if data directories exist and have correct structure"""
    print("\n" + "="*80)
    print("CHECKING DATA STRUCTURE")
    print("="*80)
    
    from config import Config
    config = Config()
    
    required_dirs = {
        'Train Images': config.TRAIN_IMG_DIR,
        'Train Masks': config.TRAIN_MASK_DIR,
        'Val Images': config.VAL_IMG_DIR,
        'Val Masks': config.VAL_MASK_DIR,
        'Test Images': config.TEST_IMG_DIR
    }
    
    all_good = True
    
    for name, path in required_dirs.items():
        if os.path.exists(path):
            # Count files
            files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            print(f"✓ {name:<20} [OK] ({len(files)} files)")
        else:
            print(f"✗ {name:<20} [MISSING] {path}")
            all_good = False
    
    if all_good:
        print("\n✓ All data directories exist!")
    else:
        print("\n✗ Some data directories are missing.")
        print("\nExpected structure:")
        print("  Train/Images/")
        print("  Train/Masks/")
        print("  Val/Images/")
        print("  Val/Masks/")
        print("  testImages/")
        return False
    
    # Verify train/mask alignment
    print("\n" + "-"*80)
    print("Verifying train/mask alignment...")
    
    train_imgs = sorted([f for f in os.listdir(config.TRAIN_IMG_DIR) 
                        if f.endswith(('.png', '.jpg', '.jpeg'))])
    train_masks = sorted([f for f in os.listdir(config.TRAIN_MASK_DIR) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(train_imgs) == len(train_masks):
        print(f"✓ Train images and masks match ({len(train_imgs)} pairs)")
    else:
        print(f"⚠ Mismatch: {len(train_imgs)} images, {len(train_masks)} masks")
    
    val_imgs = sorted([f for f in os.listdir(config.VAL_IMG_DIR) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))])
    val_masks = sorted([f for f in os.listdir(config.VAL_MASK_DIR) 
                       if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(val_imgs) == len(val_masks):
        print(f"✓ Val images and masks match ({len(val_imgs)} pairs)")
    else:
        print(f"⚠ Mismatch: {len(val_imgs)} images, {len(val_masks)} masks")
    
    return True


def check_sample_data():
    """Load and display a sample from the dataset"""
    print("\n" + "="*80)
    print("CHECKING SAMPLE DATA")
    print("="*80)
    
    from config import Config
    import cv2
    import numpy as np
    
    config = Config()
    
    # Load one training image and mask
    train_imgs = [f for f in os.listdir(config.TRAIN_IMG_DIR) 
                 if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(train_imgs) == 0:
        print("✗ No training images found")
        return False
    
    img_name = train_imgs[0]
    img_path = os.path.join(config.TRAIN_IMG_DIR, img_name)
    mask_path = os.path.join(config.TRAIN_MASK_DIR, img_name)
    
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        print(f"✗ Could not load image: {img_path}")
        return False
    
    print(f"✓ Loaded sample image: {img_name}")
    print(f"  Shape: {image.shape}")
    print(f"  Dtype: {image.dtype}")
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"✗ Could not load mask: {mask_path}")
        return False
    
    print(f"✓ Loaded sample mask: {img_name}")
    print(f"  Shape: {mask.shape}")
    print(f"  Dtype: {mask.dtype}")
    
    # Check mask values
    unique_classes = np.unique(mask)
    print(f"  Unique classes: {unique_classes}")
    print(f"  Number of classes: {len(unique_classes)}")
    
    if len(unique_classes) > config.NUM_CLASSES:
        print(f"⚠ Warning: Found {len(unique_classes)} classes, but config has {config.NUM_CLASSES}")
        print(f"  You may need to update NUM_CLASSES in config.py")
    
    if np.max(unique_classes) >= config.NUM_CLASSES:
        print(f"⚠ Warning: Max class ID ({np.max(unique_classes)}) >= NUM_CLASSES ({config.NUM_CLASSES})")
        print(f"  This will cause errors during training!")
        return False
    
    return True


def check_model_creation():
    """Test model creation"""
    print("\n" + "="*80)
    print("CHECKING MODEL CREATION")
    print("="*80)
    
    try:
        from config import Config
        from model import create_model
        import torch
        
        config = Config()
        
        print("Creating model...")
        model = create_model(config)
        
        print("✓ Model created successfully!")
        
        # Test forward pass
        print("\nTesting forward pass...")
        device = torch.device('cpu')  # Use CPU for testing
        model = model.to(device)
        
        dummy_input = torch.randn(1, 3, *config.IMAGE_SIZE).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        if output.shape[1] != config.NUM_CLASSES:
            print(f"⚠ Warning: Output has {output.shape[1]} channels, expected {config.NUM_CLASSES}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset_loading():
    """Test dataset and dataloader creation"""
    print("\n" + "="*80)
    print("CHECKING DATASET LOADING")
    print("="*80)
    
    try:
        from config import Config
        from dataset import create_dataloaders
        
        config = Config()
        
        print("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(config)
        
        print("✓ Dataloaders created successfully!")
        
        # Load one batch
        print("\nLoading one batch from train_loader...")
        images, masks = next(iter(train_loader))
        
        print(f"✓ Batch loaded successfully!")
        print(f"  Image batch shape: {images.shape}")
        print(f"  Mask batch shape: {masks.shape}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Mask dtype: {masks.dtype}")
        print(f"  Image range: [{images.min():.2f}, {images.max():.2f}]")
        print(f"  Mask range: [{masks.min()}, {masks.max()}]")
        
        if masks.max() >= config.NUM_CLASSES:
            print(f"✗ Error: Mask has class ID {masks.max()}, but NUM_CLASSES is {config.NUM_CLASSES}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks"""
    print("\n" + "="*80)
    print("OFFROAD SEGMENTATION - SETUP VERIFICATION")
    print("="*80 + "\n")
    
    checks = [
        ("Import Check", check_imports),
        ("Data Structure Check", check_data_structure),
        ("Sample Data Check", check_sample_data),
        ("Model Creation Check", check_model_creation),
        ("Dataset Loading Check", check_dataset_loading)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n✗ {check_name} failed with exception:")
            print(f"  {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for check_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{check_name:<30} {status}")
        if not result:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n✓✓✓ ALL CHECKS PASSED! ✓✓✓")
        print("\nYou're ready to start training:")
        print("  python train.py")
        print("\nOr run inference:")
        print("  python test.py")
    else:
        print("\n✗✗✗ SOME CHECKS FAILED ✗✗✗")
        print("\nPlease fix the issues above before training.")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
