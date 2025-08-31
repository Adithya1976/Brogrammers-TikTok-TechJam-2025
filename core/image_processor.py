import torch
from core.object_detector import GroundingDINO_SAMModel
from core.privacy_detector import PrivacyDetector
import io
from PIL import Image
import numpy as np
import cv2
import base64
from core.adversarial_noise import AdversarialNoiseGenerator
import logging
from typing import List, Dict, Any


class ImageProcessor:
    def __init__(self, privacy_detector: PrivacyDetector, grounding_sam: GroundingDINO_SAMModel):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.privacy_detector = privacy_detector
        self.object_detector = grounding_sam
        self.adversarial_noise_generator = AdversarialNoiseGenerator()
        self.image = None

    def to_png_b64(self, img: np.ndarray, is_mask=False) -> str:
        # For masks: convert bool -> uint8 (0/255).
        if is_mask:
            img = (img.astype(np.uint8) * 255)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError("PNG encode failed")
        return base64.b64encode(buf).decode("ascii")

    def process_image(self, image_bytes: bytes, max_dimension: int = 720) -> dict:
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        (original_width, original_height) = original_image.size

        # --- Calculate processing dimensions ---
        if original_width > original_height:
            processing_width = max_dimension
            processing_height = int(original_height * (max_dimension / original_width))
        else:
            processing_height = max_dimension
            processing_width = int(original_width * (max_dimension / original_height))
        
        logging.info(f"Original resolution: {original_width}x{original_height}")
        logging.info(f"Processing at downscaled resolution: {processing_width}x{processing_height}")

        # --- DOWNSAMPLE IMAGE for processing ---
        image = original_image.resize((processing_width, processing_height))

        # --- Step 1 & 2: Detect objects and PII on the downscaled image ---
        object_detection_results = self.object_detector.process_image(image)
        object_detection_entities = [{
            "entity_name": r["entity_name"],
            "mask": r["mask"]
        } for r in object_detection_results["entities"]]

        privacy_detection_results = self.privacy_detector.process_image(object_detection_results["processed_image"])
        privacy_detection_results = [{
            "entity_name": r["entity_name"],
            "mask": r["mask"] # mask is also a low-res numpy array
        } for r in privacy_detection_results]

        entities = object_detection_entities + privacy_detection_results
        
        # -----------------------------------------------------------------
        # --- NEW & CRUCIAL STEP: Upsample all masks to original size ---
        # -----------------------------------------------------------------
        logging.info(f"Upsampling {len(entities)} masks to {original_width}x{original_height}...")
        for entity in entities:
            low_res_mask = entity["mask"] # This is a low-resolution boolean numpy array

            # Convert boolean mask to uint8 (0 or 1) for cv2
            low_res_mask_uint8 = low_res_mask.astype(np.uint8)

            # Resize the mask. cv2.resize expects (width, height).
            # INTER_NEAREST is essential for masks to avoid creating blurry
            # intermediate values. It keeps the mask binary (0s and 1s).
            high_res_mask_uint8 = cv2.resize(
                low_res_mask_uint8,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Convert back to boolean and update the entity's mask
            entity["mask"] = high_res_mask_uint8 > 0

        # Now, encode the newly upscaled masks to Base64
        entities_encoded = [{
            "entity_name": e["entity_name"],
            "mask": self.to_png_b64(e["mask"], is_mask=True),
        } for e in entities]


        processed_image_pil = object_detection_results["processed_image"]
        image_upsampled = processed_image_pil.resize((original_width, original_height))

        # Step 3: Apply adversarial noise to the full-resolution image
        noise_induced_image = self.adversarial_noise_generator.apply_pgd_attack(
             image=image_upsampled,
             epsilon=0.007,
             iterations=20,
             target_coords=(85.8, -176.15)
        )

        result = {
            "processed_image": self.to_png_b64(np.array(noise_induced_image)),
            "shape": (original_width, original_height),
            "entities": entities_encoded
        }

        self.image = noise_induced_image

        return result

if __name__ == "__main__":
    input_file = "/Users/nipunsamudrala/workspace/coding projects/python projects/ImageClassification/resume.jpg"
    labels = ["card"]
    imageProcessor = ImageProcessor(PrivacyDetector(), GroundingDINO_SAMModel())
    with open(input_file, "rb") as f:
        image_bytes = f.read()
    result = imageProcessor.process_image(image_bytes)

    image = imageProcessor.image
    output_file = "/Users/nipunsamudrala/workspace/coding projects/python projects/ImageClassification/resume_blurred.jpg"
    image.save(output_file)
    # print(result)