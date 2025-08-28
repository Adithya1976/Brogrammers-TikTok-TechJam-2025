# TasksAI Privacy Pipeline

A 72-hour hackathon project for detecting and protecting privacy-sensitive information in images and videos using OCR and AI.

## Features

- **OCR + Privacy Detection**: Extract text from images/videos and identify sensitive information (emails, names, phone numbers, etc.)
- **Adversarial Noise**: Add noise to images to protect against OCR while maintaining visual quality
- **Video Processing**: Analyze video frames for privacy concerns
- **Blurring**: Automatically blur sensitive regions
- **Web Interface**: Simple drag-and-drop interface for testing

## Quick Start

1. **Setup**:
   ```bash
   python setup.py
   ```

2. **Run**:
   ```bash
   python main.py
   ```

3. **Open**: http://localhost:8000

## API Endpoints

- `POST /process` - Analyze image/video for privacy concerns
- `POST /add-noise` - Add adversarial noise to image
- `POST /blur-sensitive` - Blur sensitive regions in image

## Tech Stack

- **Backend**: FastAPI, Python
- **OCR**: Tesseract
- **Privacy Detection**: Microsoft Presidio
- **Image Processing**: OpenCV, PIL
- **Video Processing**: OpenCV

## Hackathon Roadmap

### Day 1 (MVP)
- [x] Basic FastAPI setup
- [x] OCR integration
- [x] Presidio privacy detection
- [x] Simple web interface

### Day 2 (Core Features)
- [x] Video processing pipeline
- [x] Adversarial noise generation
- [x] Image blurring capabilities
- [ ] Database for storing results
- [ ] Improved UI/UX

### Day 3 (Polish & Extensions)
- [ ] Mobile-responsive design
- [ ] Batch processing
- [ ] Performance optimization
- [ ] Demo video creation
- [ ] Presentation preparation

## Privacy Entities Detected

- PERSON (names)
- EMAIL_ADDRESS
- PHONE_NUMBER
- CREDIT_CARD
- SSN (Social Security Numbers)
- And more via Presidio

## Installation Notes

**Tesseract OCR** is required:
- Windows: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt install tesseract-ocr`

## Demo Ideas

1. Upload ID card → Shows privacy score + blurred version
2. Upload screenshot with email → Detects email, adds noise
3. Upload video → Processes frames, flags sensitive content
4. Show before/after adversarial noise effectiveness