#!/usr/bin/env python3
"""
Quick setup script for TasksAI Privacy Pipeline
"""
import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Virtual environment detected")
    else:
        print("⚠ No virtual environment detected. Consider using: python -m venv .venv")
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def check_tesseract():
    """Check if Tesseract is installed"""
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
        print("✓ Tesseract OCR found")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ Tesseract OCR not found. Please install:")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("  macOS: brew install tesseract")
        print("  Ubuntu: sudo apt install tesseract-ocr")
        return False

def main():
    print("🚀 Setting up TasksAI Privacy Pipeline...")
    
    # Install requirements
    install_requirements()
    
    # Check dependencies
    tesseract_ok = check_tesseract()
    
    if tesseract_ok:
        print("\n✅ Setup complete! Run with: python main.py")
    else:
        print("\n⚠ Setup incomplete. Please install missing dependencies.")

if __name__ == "__main__":
    main()