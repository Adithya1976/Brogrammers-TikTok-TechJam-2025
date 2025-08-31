#!/usr/bin/env python3
import json
import base64
import io
import os
import numpy as np
from PIL import Image, ImageFilter

# -------------------------------------------------------------------
# EDITABLE INPUTS: Change these paths to match your files
# -------------------------------------------------------------------
# Path to the JSON file received from the API
JSON_FILE_PATH = "result_mercedes_number_plate.json"

# Path to the original image that was sent to the API
ORIGINAL_IMAGE_PATH = "/Users/nipunsamudrala/workspace/coding projects/python projects/ImageClassification/images/number plates/mercedes_number_plate.png"

# Directory where the output images will be saved
OUTPUT_DIR = "visualizations"

# The radius of the Gaussian blur effect for the mask overlay
BLUR_RADIUS = 25
# -------------------------------------------------------------------


def decode_b64_to_image(b64_string: str) -> Image.Image:
    """Decodes a base64 string into a Pillow Image object."""
    try:
        image_bytes = base64.b64decode(b64_string)
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream)
        return image
    except IOError as e:
        raise ValueError("Invalid image data") from e

def main():
    """
    Main function to visualize API results by overlaying blurred masks on the original image.
    Uses hardcoded paths for input files.
    """
    # --- 1. Setup and Validate Inputs ---
    for path in [JSON_FILE_PATH, ORIGINAL_IMAGE_PATH]:
        if not os.path.exists(path):
            print(f"!!! ERROR: Input file not found at '{path}'")
            print("Please update the file path variables at the top of the script.")
            return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Outputs will be saved to the '{OUTPUT_DIR}/' directory.")

    # --- 2. Load Data ---
    print(f"Loading data from {JSON_FILE_PATH}...")
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result_data = data.get("result")
    if not result_data:
        print("Error: 'result' key not found in JSON data.")
        return

    # --- 3. Output 1: Decode and Save the Main Processed Image ---
    print("\n--- 1. Decoding Processed Image ---")
    processed_img_b64 = result_data.get("processed_image")
    if processed_img_b64:
        pil_image = decode_b64_to_image(processed_img_b64)
        if pil_image:
            output_path = os.path.join(OUTPUT_DIR, "processed_image_decoded.png")
            pil_image.save(output_path)
            print(f"✅ Saved decoded processed image to: {output_path}")
    else:
        print("⚠️  No 'processed_image' found in JSON to decode.")

    # --- 4. Output 2: Create Original Image + Blurred Masks Overlay ---
    print("\n--- 2. Creating Blurred Mask Overlay ---")
    
    # Load original image
    print(f"Loading original image from {ORIGINAL_IMAGE_PATH}...")
    original_image = Image.open(ORIGINAL_IMAGE_PATH).convert("RGB")
    
    entities = result_data.get("entities", [])
    if not entities:
        print("No entities found in JSON. Cannot create overlay.")
        return
        
    # a. Create a combined mask for all detected objects
    print(f"Aggregating {len(entities)} masks...")
    combined_mask_np = np.zeros(np.array(original_image).shape[:2], dtype=bool)

    for entity in entities:
        mask_b64 = entity.get("mask")
        if not mask_b64:
            continue

        mask_pil = decode_b64_to_image(mask_b64)
        if mask_pil:
            bit_mask = np.array(mask_pil) > 0
            combined_mask_np = np.logical_or(combined_mask_np, bit_mask)

    # b. Create a fully blurred version of the original image
    print(f"Applying Gaussian blur with radius {BLUR_RADIUS}...")
    blurred_image = original_image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

    # c. Convert the combined boolean NumPy mask to a PIL Image for compositing
    combined_mask_pil = Image.fromarray(combined_mask_np.astype(np.uint8) * 255, mode='L')
    
    # d. Composite the blurred image onto the original using the combined mask
    print("Compositing blurred regions onto the original image...")
    final_image = Image.composite(blurred_image, original_image, combined_mask_pil)

    # e. Save the final result
    overlay_output_path = os.path.join(OUTPUT_DIR, "original_with_blurred_masks.png")
    final_image.save(overlay_output_path)
    print(f"✅ Saved final composite image to: {overlay_output_path}")
    
    print("\nAll tasks complete.")

if __name__ == "__main__":
    main()
