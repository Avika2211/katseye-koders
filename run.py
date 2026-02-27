#!/usr/bin/env python
"""
Single command runner for Offroad Semantic Segmentation
Runs training and testing in sequence
"""

import os
import sys
import argparse


def check_data_structure():
    """Verify data structure exists"""
    from config import Config
    config = Config()
    
    required_dirs = [
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        config.TEST_IMG_DIR
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(dir_path)
    
    if missing:
        print("ERROR: Missing required directories:")
        for path in missing:
            print(f"  - {path}")
        print("\nExpected structure:")
        print("  data/")
        print("    train/")
        print("      images/")
        print("      segmentation/")
        print("    val/")
        print("      images/")
        print("      segmentation/")
        print("    test/")
        print("      images/")
        print("      segmentation/  (optional, for evaluation)")
        return False
    
    return True


def run_training():
    """Run training pipeline"""
    print("\n" + "="*80)
    print("STARTING TRAINING PHASE")
    print("="*80 + "\n")
    
    from train import train
    from config import Config
    
    config = Config()
    train(config)
    
    print("\n" + "="*80)
    print("TRAINING PHASE COMPLETE")
    print("="*80 + "\n")


def run_testing():
    """Run testing pipeline"""
    print("\n" + "="*80)
    print("STARTING TESTING PHASE")
    print("="*80 + "\n")
    
    from test import test
    from config import Config
    
    config = Config()
    test(config)
    
    print("\n" + "="*80)
    print("TESTING PHASE COMPLETE")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run Offroad Segmentation Pipeline')
    parser.add_argument('--skip-train', action='store_true', 
                       help='Skip training, only run testing')
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip testing, only run training')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify setup, do not run training or testing')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("OFFROAD SEMANTIC SEGMENTATION - AUTOMATED PIPELINE")
    print("="*80 + "\n")
    
    # Check data structure
    print("Verifying data structure...")
    if not check_data_structure():
        print("\nERROR: Data structure verification failed!")
        print("Please ensure your data is organized correctly.")
        sys.exit(1)
    
    print("✓ Data structure verified!\n")
    
    if args.verify_only:
        print("Verification complete. Exiting.")
        return
    
    # Run training
    if not args.skip_train:
        try:
            run_training()
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            response = input("Continue to testing? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                sys.exit(0)
        except Exception as e:
            print(f"\n\nERROR during training: {e}")
            import traceback
            traceback.print_exc()
            
            response = input("\nContinue to testing anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                sys.exit(1)
    else:
        print("Skipping training phase (--skip-train flag set)\n")
    
    # Run testing
    if not args.skip_test:
        try:
            run_testing()
        except KeyboardInterrupt:
            print("\n\nTesting interrupted by user!")
            sys.exit(0)
        except Exception as e:
            print(f"\n\nERROR during testing: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Skipping testing phase (--skip-test flag set)\n")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print("  Checkpoints: runs/checkpoints/")
    print("  Logs: runs/logs/")
    print("  Predictions: runs/predictions/")
    print("\nTo visualize training:")
    print("  tensorboard --logdir runs/logs")
    print("\nTo test with a different checkpoint:")
    print("  python test.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
