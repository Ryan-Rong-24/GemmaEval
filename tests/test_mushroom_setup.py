#!/usr/bin/env python3
"""
Test script to verify the mushroom dataset setup and dependencies.
"""

import os
import sys
from pathlib import Path

def test_dependencies():
    """Test if required dependencies are available."""
    print("🔍 Testing dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'HuggingFace Datasets'),
        ('unsloth', 'Unsloth'),
        ('trl', 'TRL'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow')
    ]
    
    all_good = True
    for package, name in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - Not installed")
            all_good = False
    
    return all_good

def test_gpu():
    """Test GPU availability."""
    print("\n🖥️  Testing GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available - {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("  ❌ CUDA not available")
            return False
    except ImportError:
        print("  ❌ PyTorch not installed")
        return False

def test_datasets():
    """Test dataset availability."""
    print("\n📁 Testing datasets...")
    
    train_path = "data/mushrooms/train/"
    test_path = "data/mushrooms/test/"
    
    train_exists = os.path.exists(train_path)
    test_exists = os.path.exists(test_path)
    
    print(f"  Training data: {'✅' if train_exists else '❌'} {train_path}")
    print(f"  Test data: {'✅' if test_exists else '❌'} {test_path}")
    
    if train_exists:
        # Count genera in training data
        try:
            train_genera = [d for d in os.listdir(train_path) 
                           if os.path.isdir(os.path.join(train_path, d))]
            print(f"    Training genera: {len(train_genera)} ({', '.join(train_genera[:3])}{'...' if len(train_genera) > 3 else ''})")
            
            # Count total images
            total_images = 0
            for genus in train_genera:
                genus_path = os.path.join(train_path, genus)
                images = [f for f in os.listdir(genus_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_images += len(images)
            print(f"    Total training images: {total_images}")
            
        except Exception as e:
            print(f"    ❌ Error reading training data: {e}")
    
    if test_exists:
        # Count genera in test data
        try:
            test_genera = [d for d in os.listdir(test_path) 
                          if os.path.isdir(os.path.join(test_path, d))]
            print(f"    Test genera: {len(test_genera)} ({', '.join(test_genera[:3])}{'...' if len(test_genera) > 3 else ''})")
            
            # Count total images
            total_images = 0
            for genus in test_genera:
                genus_path = os.path.join(test_path, genus)
                images = [f for f in os.listdir(genus_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_images += len(images)
            print(f"    Total test images: {total_images}")
            
        except Exception as e:
            print(f"    ❌ Error reading test data: {e}")
    
    return train_exists and test_exists

def test_script_imports():
    """Test if the main script can be imported."""
    print("\n📜 Testing script imports...")
    
    try:
        # Try to import the constants and classes from our script
        sys.path.append('.')
        from run_mushroom_lora_experiments import MUSHROOM_DESCRIPTIONS, IDENTIFICATION_PROMPTS, DatasetCache, Config
        
        print(f"  ✅ MUSHROOM_DESCRIPTIONS loaded ({len(MUSHROOM_DESCRIPTIONS)} genera)")
        print(f"  ✅ IDENTIFICATION_PROMPTS loaded ({len(IDENTIFICATION_PROMPTS)} prompts)")
        print(f"  ✅ DatasetCache class imported")
        print(f"  ✅ Config class imported")
        
        # Test config creation
        config = Config()
        print(f"  ✅ Config instance created")
        print(f"    Dataset path: {config.dataset_path}")
        print(f"    LoRA r values: {config.r_values}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error importing script components: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Mushroom LoRA Experiment Setup")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("GPU", test_gpu),
        ("Datasets", test_datasets),
        ("Script Imports", test_script_imports),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Ready to run LoRA experiments.")
        print("\nNext steps:")
        print("  1. Quick test: python run_mushroom_lora_experiments.py --r-values 4 8 --max-steps 10 --max-test-samples 20")
        print("  2. Full experiment: bash run_mushroom_experiments.sh")
    else:
        print("❌ Some tests failed. Please fix the issues before running experiments.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)