# TasksAI Privacy Pipeline - Setup Guide

## Virtual Environment Status ✅

**Current Setup:**
- Virtual Environment: `.venv` (already activated)
- Python Version: 3.11.2
- Location: `C:\Users\dhavi\Documents\Dhanoosh\Projects\TikTikhack\Brogrammers-TikTok-TechJam-2025\.venv`

## Quick Setup (Your Environment)

Since you already have a virtual environment activated, just run:

```bash
# Install missing dependencies
pip install torch torchvision

# Install any remaining requirements
pip install -r requirements.txt

# Check if Tesseract is installed
python setup.py

# Start the optimized server
python start_server.py
```

## Virtual Environment Commands (For Reference)

```bash
# Activate virtual environment (if not already active)
.venv\Scripts\activate

# Deactivate virtual environment
deactivate

# Check if virtual environment is active
echo $env:VIRTUAL_ENV
```

## Dependencies Status

**Already Installed:**
- ✅ FastAPI (0.116.1)
- ✅ OpenCV (4.12.0.88)
- ✅ Pillow (11.3.0)
- ✅ Presidio Analyzer (2.2.359)
- ✅ Presidio Anonymizer (2.2.359)
- ✅ PyTesseract (0.3.13)
- ✅ NumPy (2.2.6)
- ✅ PSUtil (7.0.0)
- ✅ Uvicorn (0.35.0)

**Need to Install:**
- ⚠️ PyTorch (for GPU acceleration)
- ⚠️ TorchVision (for advanced image processing)

## GPU Setup for RTX 3060

```bash
# Install PyTorch with CUDA support for RTX 3060
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## External Dependencies

**Tesseract OCR** (Required):
- Download: https://github.com/UB-Mannheim/tesseract/wiki
- Install to default location: `C:\Program Files\Tesseract-OCR\`
- Add to PATH if not automatically added

## Verification

```bash
# Test GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Test Tesseract
python -c "import pytesseract; print('Tesseract OK')"

# Test server
python start_server.py
```

## Project Structure

```
Brogrammers-TikTok-TechJam-2025/
├── .venv/                 # Virtual environment (already set up)
├── core/                  # Core processing modules
│   ├── privacy_detector.py
│   ├── video_processor.py
│   ├── adversarial_noise.py
│   └── performance_monitor.py
├── data/                  # Processing results storage
├── main.py               # FastAPI server
├── start_server.py       # Optimized server startup
├── setup.py              # Setup script
├── requirements.txt      # Python dependencies
├── API_DOCS.md          # API documentation
└── README.md            # Project documentation
```

## Network Configuration for Mobile App

1. **Find your laptop's IP address:**
   ```bash
   ipconfig
   ```
   Look for "IPv4 Address" under your WiFi adapter

2. **Ensure Windows Firewall allows port 8000:**
   - Windows Security → Firewall & network protection
   - Allow an app through firewall
   - Add Python/uvicorn if needed

3. **Mobile app should connect to:**
   ```
   http://YOUR_LAPTOP_IP:8000/api/
   ```

## Troubleshooting

**If virtual environment is not activated:**
```bash
.venv\Scripts\activate
```

**If packages are missing:**
```bash
pip install -r requirements.txt
```

**If GPU is not detected:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**If Tesseract is not found:**
- Download and install from official source
- Add to PATH: `C:\Program Files\Tesseract-OCR\`
- Restart terminal

## Performance Optimization

Your RTX 3060 setup is optimized for:
- **Batch Size**: 8 images simultaneously
- **Max Image Size**: 2048x2048 pixels
- **Video Frame Limit**: 20 frames per video
- **Concurrent Requests**: 4 simultaneous API calls

## Ready for Hackathon! 🚀

Once setup is complete:
1. Run `python start_server.py`
2. Note the network IP address shown
3. Use that IP in your mobile app development
4. Reference `API_DOCS.md` for endpoint details