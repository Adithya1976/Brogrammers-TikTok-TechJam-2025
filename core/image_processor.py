import torch
from core.object_detector import GroundingDINO_SAMModel
from core.privacy_detector import PrivacyDetector
import io
from PIL import Image
import numpy as np
import cv2
import base64
from core.adversarial_noise import AdversarialNoiseGenerator


class ImageProcessor:
    def __init__(self, privacy_detector: PrivacyDetector, grounding_sam: GroundingDINO_SAMModel):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.privacy_detector = privacy_detector
        self.object_detector = grounding_sam
        self.adversarial_noise_generator = AdversarialNoiseGenerator()
    
    def to_png_b64(self, img: np.ndarray) -> str:
        # For masks: convert bool -> uint8 (0/255). You can also try 1-bit PNG (see note below).
        img = (img.astype(np.uint8) * 255)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError("PNG encode failed")
        return base64.b64encode(buf).decode("ascii")

    def process_image(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        shape = image.size

        # step 1: Object Detection
        object_detection_results = self.object_detector.process_image(image)
        processed_image = object_detection_results["processed_image"]
        object_detection_entities = object_detection_results["entities"]
        # filter entity results to only entity_name and mask
        object_detection_entities = [{
            "entity_name": r["entity_name"],
            "mask": r["mask"]
        } for r in object_detection_entities]

        # step 2: PII detection
        privacy_detection_results = self.privacy_detector.process_image(processed_image)
        # filter PII detection results
        privacy_detection_results = [{
            "entity_name": r["entity_name"],
            "mask": r["mask"]
        } for r in privacy_detection_results]

        entities = object_detection_entities + privacy_detection_results
        entities = [{
            "entity_name": e["entity_name"],
            "mask": self.to_png_b64(e["mask"]),
        } for e in entities]

        # step 3: adversarial networks
        # placeholder
        # noise_induced_image = image
        noise_induced_image = self.adversarial_noise_generator.apply_pgd_attack(
             image=image,
             epsilon=0.007,
             iterations=20,
             target_coords=(85.8, -176.15) # Example: Target the Arctic Ocean
        )

        result = {
            "processed_image": self.to_png_b64(np.array(noise_induced_image)),
            "shape": shape,
            "entities": entities
        }
        
        return result