from textwrap import indent
import traceback
import cv2
import numpy as np
from PIL import Image
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import List, Dict, Tuple
import io
import time
import json
import re

from sympy import im

# Use EasyOCR as primary OCR backend
try:
    import easyocr
    import torch
    OCR_BACKEND = 'easyocr'
    print("🔍 EasyOCR available")
except ImportError:
    OCR_BACKEND = 'demo'
    print("⚠️ EasyOCR not available, using demo mode")

class PrivacyDetector:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.ocr_reader = None
        
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available() or torch.mps.is_available()
        
        # Initialize EasyOCR
        self.ocr_backend = OCR_BACKEND
        if self.ocr_backend == 'easyocr':
            try:
                print(f"🚀 Initializing EasyOCR with {'GPU' if self.gpu_available else 'CPU'} support...")
                self.ocr_reader = easyocr.Reader(['en'], gpu=self.gpu_available, verbose=False)
                print(f"✅ EasyOCR initialized successfully on {'GPU' if self.gpu_available else 'CPU'}")
            except Exception as e:
                print(f"⚠️ EasyOCR initialization failed: {e}")
                self.ocr_backend = 'demo'
                print("⚠️ Falling back to demo mode")
        elif self.ocr_backend == "trocr":
            self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed', use_fast=True)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed').to('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
        else:
            print("⚠️ Using demo mode for OCR")
    
    def high_contrast_grayscale(self, image) -> np.ndarray:
        """
        Load an image, convert it to a light-background, dark-text grayscale
        suitable for OCR.
        """
        # # 1. Read and resize for faster processing (optional)
        # h, w = img.shape[:2]
        # max_dim = 1024
        # if max(h, w) > max_dim:
        #     scale = max_dim / max(h, w)
        #     img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img = np.array(image)

        # 2. Convert to LAB color space to isolate luminance
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 3. Apply CLAHE to L channel: boosts local contrast without over-amplifying noise
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # 4. Merge CLAHE L channel back and convert to BGR then grayscale
        lab_clahe = cv2.merge((l_clahe, a, b))
        bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(bgr_clahe, cv2.COLOR_BGR2GRAY)

        # 5. Normalize intensities to ensure text is dark and background light
        #    We scale pixel values so that the 5th percentile maps to ~220 (light),
        #    and the 95th percentile to ~30 (dark).
        p5, p95 = np.percentile(gray, (5, 95))
        gray_norm = np.clip((gray - p5) * (255.0 / (p95 - p5)), 0, 255).astype(np.uint8)
        gray_inv = cv2.bitwise_not(gray_norm)  # invert so text (dark originally) becomes black

        # 6. Optional: apply a light Gaussian blur to smooth noise
        gray_final = cv2.GaussianBlur(gray_inv, (3, 3), 0)

        return gray_final

    def unify_text_to_black(self, image) -> None:
        """
        Loads an image of a card with mixed-color text, and outputs
        a uniform light-gray background with all text rendered in black.
        """
        # 1. Read and downscale for speed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img = np.array(image)
        
        h, w = img.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

        # 2. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Estimate background by large closing (fills text regions)
        #    Kernel size should exceed typical text height
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # 4. Subtract background to emphasize text (both dark & light)
        #    This yields bright text where original was darker and vice versa
        diff = cv2.absdiff(background, gray)

        # 5. Normalize and threshold to obtain a binary mask of text strokes
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        _, text_mask = cv2.threshold(norm, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 6. Dilate the mask slightly to fill gaps in text strokes
        dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_mask = cv2.dilate(text_mask, dilate_k, iterations=1)

        # 7. Create a uniform light-gray background
        #    Pick the median of the estimated background for consistency
        bg_color = int(np.median(background))
        uniform_bg = np.full_like(gray, bg_color)

        # 8. Paint text pixels black onto the uniform background
        result = uniform_bg.copy()
        result[text_mask == 255] = 0  # text → black

        # 9. Save the preprocessed image
        return result

    def preprocess_mobile_photo_background_subtract(self, image_cv: np.ndarray) -> np.ndarray:
        # (same as before)
        if image_cv.ndim == 3:
            img = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        else:
            img = image_cv
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        bg = cv2.GaussianBlur(enhanced, (51,51), 0)
        normalized = cv2.divide(enhanced, bg, scale=255)
        mean_val = normalized.mean()
        invert = mean_val < 127
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(
            normalized, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type,
            blockSize=15, C=7
        )
        if invert:
            binary = cv2.bitwise_not(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        return cv2.dilate(binary, kernel, iterations=1)
        
    def extract_text_from_image(self, image_bytes: bytes, filename: str = None) -> str:
        """Extract text from image using EasyOCR"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            if self.ocr_backend == 'easyocr' and self.ocr_reader:
                print("🔍 Running EasyOCR..."   )
                # Preprocess image for better OCR results
                # image_np = self._preprocess_image_for_easyocr(image)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_np = np.array(image)
                
                # rotate image clockwise
                image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)

                # print shape of image
                print(f"Image shape after pre-process: {image_np.shape}")

                # store image for debugging
                cv2.imwrite("debug_image.png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

                # Run EasyOCR with optimized settings for cards/documents
                results = self.ocr_reader.readtext(
                    image_np
                )
                
                parts, boxes = [], []
                for bbox, text, confidence in results:
                    t = (text or "").strip()
                    if confidence > 0.4 and t:
                        parts.append(t)
                        boxes.append(bbox)

                span_to_bb = {}
                extracted_chunks = []
                pos = 0  # running char position in the final string

                for i, (t, bb) in enumerate(zip(parts, boxes)):
                    if i > 0:
                        # account for the single space inserted by `' '.join`
                        extracted_chunks.append(" ")
                        pos += 1
                    start = pos
                    extracted_chunks.append(t)
                    pos += len(t)
                    # map (start, end+1) -> bbox
                    span_to_bb[(start, pos)] = bb

                return ''.join(extracted_chunks), span_to_bb
            
        except Exception as e:
            print(f"OCR Error: {e}")
            traceback.print_exc()
            # Fallback to demo mode
    
    def _preprocess_image_for_easyocr(self, image: Image.Image) -> np.ndarray:
        """Preprocess image to improve EasyOCR accuracy for cards/documents"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Enhance contrast for better text recognition
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))  # Increased clipLimit, smaller tiles
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        image_np = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return image_np
    
    def analyze_privacy(self, text: str) -> Dict:
        """Analyze text for privacy-sensitive information"""
        if not text:
            return {"entities": [], "score": 0, "is_safe": True}
        
        # text = text.replace("/", " ")
        # Run Presidio analysis
        results = self.analyzer.analyze(text=text, language='en')
        
        # Define which entity types are actually privacy-sensitive
        HIGH_RISK_ENTITIES = {
            'EMAIL_ADDRESS', 'PHONE_NUMBER', 'PERSON', 'US_SSN', 'CREDIT_CARD',
            'US_DRIVER_LICENSE', 'US_PASSPORT', 'IBAN_CODE', 'IP_ADDRESS',
            'MEDICAL_LICENSE', 'US_BANK_NUMBER', 'CRYPTO', 'ID_NUMBER', 'FINANCIAL_CARD'
        }
        
        # Medium risk entities (less sensitive)
        MEDIUM_RISK_ENTITIES = {
            'LOCATION', 'URL', 'US_ITIN'
        }
        
        # Low risk entities (generally not privacy-sensitive)
        LOW_RISK_ENTITIES = {
            'DATE_TIME', 'NRP'  # NRP = Nationality/Religion/Political affiliation
        }
        
        entities = []
        privacy_score = 0
        
        # Process Presidio results
        for result in results:
            entity_type = result.entity_type
            confidence = result.score
            
            # Calculate weighted score based on entity type and confidence
            if entity_type in HIGH_RISK_ENTITIES:
                weight = 3.0
            elif entity_type in MEDIUM_RISK_ENTITIES:
                weight = 1.5
            elif entity_type in LOW_RISK_ENTITIES:
                weight = 0.5
            else:
                weight = 2.0  # Unknown entities get medium-high weight
            
            entity_score = confidence * weight
            privacy_score += entity_score
            
            entities.append({
                "type": entity_type,
                "start": result.start,
                "end": result.end,
                "text": text[result.start:result.end],
            })
            print(f"Detected entity: {entity_type} ({text[result.start:result.end]})")
    
        
        return entities
    
    def _detect_card_patterns(self, text: str) -> List[Dict]:
        """Detect specific card patterns that might be missed by Presidio"""
        patterns = []
        
        # Credit card patterns
        cc_pattern = r'\b(?:\d{4}[\s-]?){3}\d{4}\b'
        for match in re.finditer(cc_pattern, text):
            patterns.append({
                "type": "CREDIT_CARD",
                "confidence": 0.9,
                "start": match.start(),
                "end": match.end(),
                "text": match.group(),
                "risk_level": "high"
            })
        
        # ID number patterns (various formats)
        id_patterns = [
            r'\b[A-Z]\d{8,9}\b',  # Driver license format
            r'\b\d{2}/\d{2}/\d{4}\b',  # Date format (birth date, expiry)
            r'\bIssued on\s+\d{2}/\d{2}/\d{4}\b',  # Issue date
            r'\b[A-Z]{2,}\s+\d{6,}\b',  # ID with letters and numbers
        ]
        
        for pattern in id_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                patterns.append({
                    "type": "ID_NUMBER",
                    "confidence": 0.8,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "risk_level": "high"
                })
        
        # Card type indicators
        card_indicators = ['VISA', 'MASTERCARD', 'AMEX', 'DISCOVER', 'DBS', 'PLATINUM', 'MULTI-CURRENCY']
        for indicator in card_indicators:
            if indicator in text.upper():
                patterns.append({
                    "type": "FINANCIAL_CARD",
                    "confidence": 0.95,
                    "start": 0,
                    "end": len(indicator),
                    "text": indicator,
                    "risk_level": "high"
                })
        
        return patterns
    
    def process_image(self, image_bytes: bytes, filename: str = None) -> Dict:
        try:
            """Complete privacy analysis pipeline for images with performance tracking"""
            start_time = time.time()
            result = []
            
            # Extract text
            ocr_start = time.time()
            image = Image.open(io.BytesIO(image_bytes))
            shape = image.size
            ocr_text, span_to_bb = self.extract_text_from_image(image_bytes, filename)
            ocr_time = time.time() - ocr_start
            
            print(f"OCR Text: '{ocr_text}'")
            # Analyze privacy
            analysis_start = time.time()
            detection_list = self.analyze_privacy(ocr_text)
            analysis_time = time.time() - analysis_start

            mask_list = []

            # print span_to_bb keys and text for debugging
            print("Span to BB mapping:")
            for (s, e), bb in span_to_bb.items():
                print(f"  ({s}, {e}): ocr_text='{ocr_text[s:e]}'")


            # map bounding boxes to entities
            for detection in detection_list:
                start, end = detection['start'], detection['end']
                text = detection['text']
                target_entity = detection['type']
                # get all keys in span_to_bb that lie within (start, end)
                bb_list = [bb for (s, e), bb in span_to_bb.items() if s >= start and e <= end]

                for bb in bb_list:
                    # convert bb to image mask
                    mask = self.convert_bb_to_mask(bb, shape)
                    mask_list.append(mask)

                    result.append({
                        "text": text,
                        "bounding_box": bb,
                        "entity_type": target_entity,
                        "mask": mask.tolist()
                    })

            combined_mask = self.combined_mask(mask_list, shape)
            
            # mask the image out and save for debugging
            image_np = np.array(image)
            # rotate image clockwise
            image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
            print(image_np.shape)
            print(combined_mask.shape)
            masked_image = image_np.copy()
            masked_image[combined_mask == 1] = 0
            cv2.imwrite("debug_masked_image.png", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

            # print result's text and entity type
            for res in result:
                print(f"Text: {res['text']}, Entity Type: {res['entity_type']}")
            total_time = time.time() - start_time
        except Exception as e:
            print(f"Processing Error: {e}")
            traceback.print_exc()
            return {}

    def convert_bb_to_mask(self, bb: List[List[np.float64]], image_shape: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
        """Convert bounding box to binary mask"""
        mask = np.zeros(image_shape, dtype=np.uint8)
        pts = np.array(bb, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)

        # save mask for debugging
        cv2.imwrite("debug_mask.png", mask * 255)

        return mask

    def combined_mask(self, masks: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
        """Combine multiple binary masks into one"""
        combined = np.zeros(shape, dtype=np.uint8)
        for mask in masks:
            combined = np.maximum(combined, mask)
        # save combined mask for debugging
        cv2.imwrite("debug_combined_mask.png", combined * 255)
        return combined
        