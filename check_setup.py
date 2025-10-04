#!/usr/bin/env python3
"""
Setup checker for Steel Defect Segmentation project
This script checks if all required components are available.
"""

import os
import sys
import torch

def check_model_file():
    """Check if the model file exists"""
    model_path = "outputs/checkpoints/best_model_dice.pth"
    print(f"🔍 Checking for model file: {model_path}")
    
    if os.path.exists(model_path):
        print("✅ Model file found!")
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.2f} MB")
        return True
    else:
        print("❌ Model file NOT found!")
        print("\n💡 Solutions:")
        print("1. Train the model using main.ipynb notebook")
        print("2. Download a pre-trained model (if available)")
        print("3. Use the demo mode (modify app.py to skip model loading)")
        return False

def check_pytorch():
    """Check PyTorch installation"""
    print(f"\n🔍 Checking PyTorch installation...")
    print(f"✅ PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available - using CPU")

def check_data():
    """Check if data files exist"""
    print(f"\n🔍 Checking data files...")
    
    data_files = ["data/train.csv", "data/sample_submission.csv"]
    for file in data_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} NOT found")

def main():
    print("🚀 Steel Defect Segmentation - Setup Checker")
    print("=" * 50)
    
    check_pytorch()
    check_data()
    has_model = check_model_file()
    
    print("\n" + "=" * 50)
    if has_model:
        print("🎉 All components ready! You can run:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Model file missing. Options:")
        print("1. Train model: Run cells in main.ipynb")
        print("2. Run demo mode: Modify app.py to use dummy model")
        print("3. Download model: If you have a pre-trained model")

if __name__ == "__main__":
    main()