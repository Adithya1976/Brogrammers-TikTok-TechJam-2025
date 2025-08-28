#!/usr/bin/env python3
"""
Check current environment status for TasksAI Privacy Pipeline
"""
import sys
import os
import subprocess
import importlib.util

def check_virtual_env():
    """Check if virtual environment is active"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_path = os.environ.get('VIRTUAL_ENV', 'Unknown')
        print("✅ Virtual Environment: ACTIVE")
        print(f"   Path: {venv_path}")
        print(f"   Python: {sys.executable}")
        return True
    else:
        print("❌ Virtual Environment: NOT ACTIVE")
        print("   Run: .venv\\Scripts\\activate")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            # Try to import to check if it works
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package_name}: {version}")
            return True
        else:
            print(f"❌ {package_name}: NOT INSTALLED")
            return False
    except Exception as e:
        print(f"⚠️ {package_name}: ERROR - {e}")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            print(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("❌ GPU: CUDA not available")
            return False
    except ImportError:
        print("❌ GPU: PyTorch not installed")
        return False

def check_tesseract():
    """Check Tesseract OCR"""
    try:
        import pytesseract
        # Try to get version
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ Tesseract: {version_line}")
            return True
        else:
            print("❌ Tesseract: Command failed")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Tesseract: Command timeout")
        return False
    except FileNotFoundError:
        print("❌ Tesseract: Not found in PATH")
        print("   Download: https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    except ImportError:
        print("❌ PyTesseract: Not installed")
        return False

def main():
    print("🔍 TasksAI Privacy Pipeline - Environment Check")
    print("=" * 50)
    
    # Check virtual environment
    venv_ok = check_virtual_env()
    print()
    
    # Check core packages
    print("📦 Core Packages:")
    packages = [
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("OpenCV", "cv2"),
        ("Pillow", "PIL"),
        ("NumPy", "numpy"),
        ("Presidio Analyzer", "presidio_analyzer"),
        ("Presidio Anonymizer", "presidio_anonymizer"),
        ("PyTesseract", "pytesseract"),
        ("PSUtil", "psutil"),
        ("Aiofiles", "aiofiles")
    ]
    
    package_status = []
    for name, import_name in packages:
        status = check_package(name, import_name)
        package_status.append(status)
    
    print()
    
    # Check GPU
    print("🎮 GPU Status:")
    gpu_ok = check_gpu()
    print()
    
    # Check Tesseract
    print("🔤 OCR Status:")
    tesseract_ok = check_tesseract()
    print()
    
    # Summary
    print("📊 Summary:")
    print(f"   Virtual Environment: {'✅' if venv_ok else '❌'}")
    print(f"   Core Packages: {sum(package_status)}/{len(package_status)} installed")
    print(f"   GPU Acceleration: {'✅' if gpu_ok else '❌'}")
    print(f"   Tesseract OCR: {'✅' if tesseract_ok else '❌'}")
    
    if all([venv_ok, all(package_status), tesseract_ok]):
        print("\n🚀 Ready to start server!")
        print("   Run: python start_server.py")
    else:
        print("\n⚠️ Setup needed:")
        if not venv_ok:
            print("   - Activate virtual environment: .venv\\Scripts\\activate")
        if not all(package_status):
            print("   - Install packages: pip install -r requirements.txt")
        if not gpu_ok:
            print("   - Install PyTorch: pip install torch torchvision")
        if not tesseract_ok:
            print("   - Install Tesseract OCR")

if __name__ == "__main__":
    main()