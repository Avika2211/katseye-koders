#!/usr/bin/env python
"""
Quick installation check - run this first to ensure everything is ready
"""

import sys

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} [OK]")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} [Need 3.8+]")
        return False


def check_packages():
    """Check if all required packages can be imported"""
    print("\nChecking packages...")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'segmentation_models_pytorch': 'Segmentation Models',
        'albumentations': 'Albumentations',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name:<25} [OK]")
        except ImportError:
            print(f"  ✗ {name:<25} [MISSING]")
            all_ok = False
    
    return all_ok


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"  ⚠ CUDA not available (will use CPU - slower)")
    except:
        print(f"  ✗ Cannot check CUDA")


def main():
    print("="*60)
    print("INSTALLATION CHECK")
    print("="*60 + "\n")
    
    checks = []
    
    # Check Python
    checks.append(check_python_version())
    
    # Check packages
    checks.append(check_packages())
    
    # Check CUDA
    check_cuda()
    
    # Summary
    print("\n" + "="*60)
    if all(checks):
        print("✓✓✓ ALL CHECKS PASSED! ✓✓✓")
        print("\nYou're ready to run:")
        print("  python run.py")
    else:
        print("✗✗✗ SOME CHECKS FAILED ✗✗✗")
        print("\nInstall missing packages:")
        print("  pip install -r requirements.txt")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
