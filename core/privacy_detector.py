from textwrap import indent
import traceback
from unittest import result
import cv2
from networkx import eccentricity
import numpy as np
from PIL import Image, ImageOps
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from regex import E
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import List, Dict, Sequence, Tuple, Type
import io
import time
import json
import re
import math

from sympy import im

# Use EasyOCR as primary OCR backend
try:
    import easyocr
    import torch
    OCR_BACKEND = 'easyocr'
    # print("🔍 EasyOCR available")
except ImportError:
    OCR_BACKEND = 'demo'
    print("⚠️ EasyOCR not available, using demo mode")

class PrivacyDetector:
    def __init__(self, debug: bool = False, use_gpu: bool = True):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.ocr_reader = None
        self.debug = debug
        
        # Check GPU availability
        self.gpu_available = use_gpu and (torch.cuda.is_available() or torch.backends.mps.is_available())
        
        # Initialize EasyOCR
        self.ocr_backend = OCR_BACKEND
        if self.ocr_backend == 'easyocr':
            try:
                if self.debug:
                    print(f"🚀 Initializing EasyOCR with {'GPU' if self.gpu_available else 'CPU'} support...")
                self.ocr_reader = easyocr.Reader(['en'], gpu=self.gpu_available, verbose=False)
                if self.debug:
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
        
    def extract_text_from_image(self, image_np, filename: str = None) -> str:
        """Extract text from image using EasyOCR"""
        try:
            
            if self.ocr_backend == 'easyocr' and self.ocr_reader:
                # Preprocess image for better OCR results
                # image_np = self._preprocess_image_for_easyocr(image)

                # print shape of image
                if self.debug:
                    print(f"Image shape after pre-process: {image_np.shape}")

                # store image for debugging
                cv2.imwrite("debug_image.png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

                # Run EasyOCR with optimized settings for cards/documents
                results = self.ocr_reader.readtext(
                    image_np
                )

                parts, boxes, confidences = [], [], []
                for bbox, text, confidence in results:
                    t = (text or "").strip()
                    if confidence > 0.4 and t:
                        parts.append(t)
                        boxes.append(bbox)
                        confidences.append(confidence)


                span_to_bb = {}
                extracted_chunks = []
                pos = 0  # running char position in the final string

                for i, (t, bb, conf) in enumerate(zip(parts, boxes, confidences)):
                    if i > 0:
                        # account for the single space inserted by `' '.join`
                        extracted_chunks.append(" ")
                        pos += 1
                    start = pos
                    extracted_chunks.append(t)
                    pos += len(t)
                    # map (start, end+1) -> bbox
                    span_to_bb[(start, pos)] = {
                        "box": bb,
                        "confidence": conf
                    }
                    

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
            return []
        
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

            
            entities.append({
                "type": entity_type,
                "start": result.start,
                "end": result.end,
                "text": text[result.start:result.end],
                "confidence": confidence,
                "span_len": int(result.end - result.start)
            })
            if self.debug:
                print(f"Detected entity: {entity_type} ({text[result.start:result.end]})")
    
        
        return entities
    
    def process_image(self, image: Image, filename: str = None) -> List[Dict]:
        try:
            """Multi-rotation OCR → Presidio → entity-wise dedupe by bigger-span + overlap-over-min."""
            start_time = time.time()

            # ---------------- helpers ----------------
            def rotate_image_keep_canvas(img_np: np.ndarray, angle_deg: float) -> np.ndarray:
                h, w = img_np.shape[:2]
                M = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle_deg, 1.0)
                return cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            def rotate_point(px: float, py: float, cx: float, cy: float, angle_deg: float) -> tuple[float, float]:
                ang = math.radians(angle_deg)
                ca, sa = math.cos(ang), math.sin(ang)
                x0, y0 = px - cx, py - cy
                return (x0*ca - y0*sa + cx, x0*sa + y0*ca + cy)

            def map_bbox_to_original(bb4: list[list[float]], w: int, h: int, angle_applied_deg: float) -> list[list[int]]:
                """bb4 is in rotated frame (angle_applied_deg applied). Map back by rotating -angle."""
                cx, cy = w/2.0, h/2.0
                inv = -angle_applied_deg
                mapped = [rotate_point(x, y, cx, cy, inv) for (x, y) in bb4]
                return [[int(round(x)), int(round(y))] for (x, y) in mapped]

            def union_mask(boxes: list[list[list[int]]], H: int, W: int) -> np.ndarray:
                m = np.zeros((H, W), dtype=np.uint8)
                for bb in boxes:
                    m = np.maximum(m, self.convert_bb_to_mask(bb, (H, W)))
                return m

            def mask_overlap_over_min(a: np.ndarray, b: np.ndarray) -> float:
                inter = np.logical_and(a, b).sum()
                sa, sb = a.sum(), b.sum()
                denom = min(sa, sb)
                return (inter / denom) if denom > 0 else 0.0

            # ---------------- image prep ----------------
            image = ImageOps.exif_transpose(image)
            image_np = np.array(image)
            H, W = image_np.shape[:2]

            angles = list(range(0, 360, 45))  # includes 0

            # Final kept entities (original frame)
            # each: {entity_type, text, start, end, span_len, boxes: [bb...]}
            best_entities: List[Dict] = []
            entity_masks: List[np.ndarray] = []  # cache union mask per entity

            def add_or_replace_entity(cand: Dict, entity_overlap_thr: float = 0.6):
                """Replace existing entities if cand is bigger-span AND overlap-over-min >= thr.
                Else, if tie & same (type+text), union boxes; otherwise keep existing."""
                # Prepare candidate mask
                cand_mask = union_mask(cand["boxes"], H, W)

                # Find overlaps (by overlap-over-min on masks)
                overlaps = []
                for idx, ex_mask in enumerate(entity_masks):
                    overlap = mask_overlap_over_min(ex_mask, cand_mask)
                    if overlap >= entity_overlap_thr:
                        overlaps.append((idx, overlap))

                if not overlaps:
                    # no strong overlap → new entity
                    best_entities.append(cand)
                    entity_masks.append(cand_mask)
                    return
                

                # There is at least one overlapping existing entity
                # Check if ANY overlapping existing has span_len >= candidate
                max_span = max(best_entities[i]["span_len"] for i, _ in overlaps)
                if cand["span_len"] > max_span:
                    # Candidate is bigger → remove all overlapped entities, then add candidate
                    to_remove = sorted([i for i, _ in overlaps], reverse=True)

                    # print the overlapped entity type and text that are to be removed
                    for i in to_remove:
                        best_entities.pop(i)
                        entity_masks.pop(i)
                    best_entities.append(cand)
                    entity_masks.append(cand_mask)
                    return

                # Tie or smaller: keep existing.
                # Optional: if tie AND clearly same semantic entity → union boxes into the best-overlap one
                # (same type + same text)
                if cand["span_len"] == max_span:
                    # choose the existing with max overlap
                    best_idx = max(overlaps, key=lambda t: t[1])[0]
                    ex = best_entities[best_idx]
                    if (ex["entity_type"] == cand["entity_type"]) and (ex["text"] == cand["text"]):
                        ex["boxes"].extend(cand["boxes"])
                        # refresh mask
                        entity_masks[best_idx] = np.maximum(entity_masks[best_idx], cand_mask)
                # If smaller span, we ignore cand entirely.

            # ---------------- per-rotation pass ----------------
            for angle in angles:
                t0 = time.time()
                rotated_np = rotate_image_keep_canvas(image_np, angle)

                if self.debug:
                    cv2.imwrite(f"debug_rotated_{angle}.png", cv2.cvtColor(rotated_np, cv2.COLOR_RGB2BGR))

                # OCR (you provide this)
                ocr_text, span_to_bb = self.extract_text_from_image(rotated_np, filename)

                if self.debug:
                    print(f"[{angle:3d}] OCR len={len(ocr_text)} span_count={len(span_to_bb)}")

                # Presidio on this rotation
                dets = self.analyze_privacy(ocr_text)

                # Build candidate entities for this rotation
                rotation_candidates: List[Dict] = []
                for d in dets:
                    start, end, text_span = d["start"], d["end"], d["text"]
                    ent_type, span_len = d["type"], d["span_len"]

                    boxes, text = [], []
                    for (s, e), bb_dict in span_to_bb.items():
                        # change it to partial match (within)
                        if not (s < end and e > start):
                            bb_rot = bb_dict.get("box")
                            bb_orig = map_bbox_to_original(bb_rot, W, H, -angle)
                            boxes.append(bb_orig)
                            text.append(ocr_text[s:e])

                    if boxes:
                        rotation_candidates.append({
                            "entity_type": ent_type,
                            "text": text_span,
                            "start": start,
                            "end": end,
                            "span_len": span_len,
                            "boxes": boxes,
                        })

                # Merge each candidate using bigger-span + overlap-over-min rule
                for cand in rotation_candidates:
                    add_or_replace_entity(cand, entity_overlap_thr=0.6)

                if getattr(self, "debug", False):
                    print(f"[{angle:3d}] kept_entities={len(best_entities)} took={time.time()-t0:.2f}s")
                
                # draw all current best_entities on a debug image
                if getattr(self, "debug", False):
                    debug_img = image_np.copy()
                    for ent in best_entities:
                        color = (0, 255, 0)
                        for bb in ent["boxes"]:
                            pts = np.array(bb, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(debug_img, [pts], isClosed=True, color=color, thickness=2)
                            # put entity type text near the first point
                            cv2.putText(debug_img, ent["entity_type"], tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.imwrite(f"debug_best_entities_after_{angle}.png", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

            # ---------------- finalize masks & output ----------------
            combined_mask = np.zeros((H, W), dtype=np.uint8)
            for ent_idx, ent in enumerate(best_entities):
                m = union_mask(ent["boxes"], H, W)
                ent["mask"] = m
                combined_mask = np.maximum(combined_mask, m)

            masked_image = image_np.copy()
            masked_image[combined_mask == 1] = 0
            if getattr(self, "debug", False):
                cv2.imwrite("debug_masked_image.png", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
                print("=== Final Entities ===")
                for e in best_entities:
                    print(f"  {e['entity_type']} span={e['span_len']} boxes={len(e['boxes'])}")

            # Return serialized masks (swap to PNG/RLE if desired)
            results_output = [
                {
                    "entity_name": e["entity_type"],
                    "mask": e["mask"],
                    "boxes": e["boxes"]
                }
                for e in best_entities
            ]
            return results_output

        except Exception as ex:
            traceback.print_exc()
            raise Exception(f"Processing error: {ex}")

    def convert_bb_to_mask(self, bb: List[List[np.float64]], image_shape: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
        """Convert bounding box to binary mask"""
        mask = np.zeros(image_shape, dtype=np.uint8)
        pts = np.array(bb, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)

        if self.debug:
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