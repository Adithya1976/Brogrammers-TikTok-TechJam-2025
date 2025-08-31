# TasksAI Privacy Pipeline

[Youtube Video Link](https://www.youtube.com/watch?v=xHC2i59YnKc)

## Project Overview

A 72-hour hackathon project for developing a privacy-focussed photo gallery app with the following capabilities:


## Features

- **OCR + Privacy Detection**: Extract text from images/videos and identify sensitive information (emails, names, phone numbers, etc.)
- **Adversarial Noise**: Add noise to images to protect against OCR while maintaining visual quality
- **Video Processing**: Analyze video frames for privacy concerns
- **Blurring**: Automatically blur sensitive regions
- **Web Interface**: Simple drag-and-drop interface for testing

## Quick Start

1. **Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run**:
   First run the server:
   ```bash
   python main.py
   ```

   Wait until you see the log message "Uvicorn running at http://127.0.0.1:8000" in your terminal, indicating the server has started.
   
   Once this appears, all functionalities of the app will be available.

   Alternatively, use/modify the template at test_api.py.


## Tech Stack

- **Backend**: FastAPI, Python
- **OCR**: [Tesseract](https://github.com/tesseract-ocr/tesseract)
- **Privacy Detection**: [Microsoft Presidio](https://microsoft.github.io/presidio/)
- **Image Processing**: [GroundingSAM](https://huggingface.co/docs/transformers/v4.44.2/model_doc/grounding-dino#grounded-sam)
- **Audio Processing**: [OpenAI Whisper](https://github.com/openai/whisper)
- **Adversarial Noise Generator**: [GeoClip](https://arxiv.org/abs/2309.16020)


## Privacy Entities Detected

- PERSON
- EMAIL_ADDRESS
- PHONE_NUMBER
- CREDIT_CARD
- SSN (Social Security Numbers)
- LICENSE_PLATES
