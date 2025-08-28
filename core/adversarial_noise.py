import numpy as np
import cv2
from PIL import Image
import io
from typing import Tuple

class AdversarialNoiseGenerator:
    def __init__(self):
        pass
    
    def add_gaussian_noise(self, image_bytes: bytes, intensity: float = 0.1) -> bytes:
        """Add Gaussian noise to obscure OCR while maintaining visual quality"""
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)
        
        # Generate noise
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        
        # Add noise and clip values
        noisy_image = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Convert back to bytes
        result_image = Image.fromarray(noisy_image)
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format='JPEG')
        
        return output_buffer.getvalue()
    
    def add_text_specific_noise(self, image_bytes: bytes, text_regions: list = None) -> bytes:
        """Add targeted noise to text regions"""
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)
        
        if text_regions:
            # Apply noise only to specified regions
            for region in text_regions:
                x, y, w, h = region
                noise = np.random.normal(0, 0.2 * 255, (h, w, img_array.shape[2]))
                img_array[y:y+h, x:x+w] = np.clip(
                    img_array[y:y+h, x:x+w] + noise, 0, 255
                )
        else:
            # Apply light noise globally
            noise = np.random.normal(0, 0.05 * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
        
        # Convert back to bytes
        result_image = Image.fromarray(img_array.astype(np.uint8))
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format='JPEG')
        
        return output_buffer.getvalue()
    
    def blur_sensitive_regions(self, image_bytes: bytes, regions: list) -> bytes:
        """Blur specific regions containing sensitive information"""
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)
        
        for region in regions:
            x, y, w, h = region
            # Apply Gaussian blur to the region
            roi = img_array[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
            img_array[y:y+h, x:x+w] = blurred_roi
        
        # Convert back to bytes
        result_image = Image.fromarray(img_array)
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format='JPEG')
        
        return output_buffer.getvalue()