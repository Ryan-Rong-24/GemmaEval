#!/usr/bin/env python3
"""
Setup script for the Gemma Fire/Smoke Detection Evaluation System.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def create_directory_structure():
    """Create necessary directories."""
    print("📁 Creating directory structure...")
    
    directories = [
        "data",
        "results",
        "logs",
        "scripts",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}/")
    
    print("✅ Directory structure created")

def copy_env_template():
    """Copy environment template if .env doesn't exist."""
    if not Path(".env").exists():
        if Path("env.example").exists():
            shutil.copy("env.example", ".env")
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file and add your GEMINI_API_KEY")
        else:
            print("❌ env.example not found")
    else:
        print("✅ .env file already exists")

def install_dependencies():
    """Install Python dependencies."""
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        return False
    return True

def check_dataset():
    """Check if D-Fire dataset exists."""
    dataset_path = Path("data/D-Fire")
    if dataset_path.exists():
        print("✅ D-Fire dataset found")
        
        # Check structure
        required_paths = [
            dataset_path / "train" / "images",
            dataset_path / "train" / "labels",
            dataset_path / "test" / "images", 
            dataset_path / "test" / "labels"
        ]
        
        missing_paths = [p for p in required_paths if not p.exists()]
        
        if missing_paths:
            print("⚠️  Some dataset directories are missing:")
            for path in missing_paths:
                print(f"    Missing: {path}")
        else:
            print("✅ Dataset structure is complete")
    else:
        print("⚠️  D-Fire dataset not found at data/D-Fire")
        print("   Please download the dataset from:")
        print("   https://github.com/gaiasd/DFireDataset")

def check_api_key():
    """Check if API key is configured."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key != "your_gemini_api_key_here":
            print("✅ GEMINI_API_KEY is configured")
            return True
        else:
            print("⚠️  GEMINI_API_KEY not found or not set")
            print("   Please set your Gemini API key in the .env file")
            return False
    except ImportError:
        print("⚠️  Cannot check API key (dotenv not installed)")
        return False

def run_quick_test():
    """Run a quick test of the system."""
    print("🧪 Running quick system test...")
    
    if not check_api_key():
        print("⚠️  Skipping test due to missing API key")
        return
    
    test_command = "python main.py --quick-test --max-images 2"
    if run_command(test_command, "Quick system test"):
        print("🎉 System test passed!")
    else:
        print("❌ System test failed")

def main():
    """Main setup function."""
    print("=" * 60)
    print("🔥 Gemma Fire/Smoke Detection Evaluation System Setup")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    
    # Copy environment template
    copy_env_template()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed due to dependency installation issues")
        return 1
    
    # Check dataset
    check_dataset()
    
    # Check API key
    api_key_ok = check_api_key()
    
    print("\n" + "=" * 60)
    print("📋 Setup Summary")
    print("=" * 60)
    
    if api_key_ok:
        print("✅ System is ready to use!")
        print("\nTo run evaluations:")
        print("  python main.py --quick-test           # Quick test")
        print("  python main.py                        # Full evaluation")
        print("  python main.py --help                 # See all options")
    else:
        print("⚠️  Setup incomplete - please configure GEMINI_API_KEY")
        print("\nSteps to complete setup:")
        print("1. Edit .env file and add your Gemini API key")
        print("2. Run: python main.py --quick-test")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 