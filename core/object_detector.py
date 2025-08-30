from dataclasses import dataclass
from pprint import pprint
from transformers import AutoProcessor, AutoModelForMaskGeneration, pipeline
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import torch
from typing import List, Optional, Dict, Any
import numpy as np
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]
    
    def to_dict(self) -> Dict[str, float]:
        return {"xmin": self.xmin, "ymin": self.ymin, "xmax": self.xmax, "ymax": self.ymax}


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=float(detection_dict['box']['xmin']),
                                   ymin=float(detection_dict['box']['ymin']),
                                   xmax=float(detection_dict['box']['xmax']),
                                   ymax=float(detection_dict['box']['ymax'])))


class GroundingDINO_SAMModel:
    """
    A class for Grounding DINO with SAM (Segment Anything Model) integration.
    Models are loaded once during initialization for efficient video processing.
    """
    def __init__(self, device: torch.device | None = None) -> None:
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = device

        logging.info(f"Initializing GroundingSAM on device: {self.device}")

        # --- Use fast model names ---
        dino_model_name = "IDEA-Research/grounding-dino-tiny"
        sam_model_name = "facebook/sam-vit-base"

        # --- Load models and processors once ---
        self.object_detector = pipeline(model=dino_model_name, task="zero-shot-object-detection", device=self.device)
        self.sam_processor = AutoProcessor.from_pretrained(sam_model_name)
        self.sam_model = AutoModelForMaskGeneration.from_pretrained(sam_model_name).to(self.device)

        # Set default labels for detection (if none are provided)
        self.default_labels = ["person", "license plate", "card", "sign"]

    def detect(self, image: Image.Image, labels: List[str] = [], score_threshold: float = 0.25) -> List[DetectionResult]:
        """
        Runs Grounding DINO on the given image.

        Args:
            image (Image.Image): The image to process.
            labels (List[str]): The text labels for detection.
            score_threshold (float): The score threshold for filtering detections.

        Returns:
            A list of DetectionResult objects.
        """
        if not labels:
            labels = self.default_labels

        # Ensure labels end with a dot for better performance
        processed_labels = [label if label.endswith(".") else label + "." for label in labels]
        
        results = self.object_detector(image, candidate_labels=processed_labels, threshold=score_threshold)
        detections = [DetectionResult.from_dict(result) for result in results]

        # Store these detections for the segment method to use
        self.detections = detections
        return self.detections

    def segment(self, image: Image.Image, detections: List[DetectionResult]) -> List[DetectionResult] | None:
        """
        Segments the detected objects (from the last `detect` call) in the image using SAM.
        
        Args:
            image (Image.Image): The image that was used for the last detection.

        Returns:
            The detection list with masks updated.
        """
        if not detections:
            logging.warning("Warning: segment() called without any prior detections.")
            return None
        
        input_boxes = [[box.xyxy for box in self.__get_boxes(detections)]]
        feats = self.sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt")

        original_sizes = feats.original_sizes
        reshaped_input_sizes = feats.reshaped_input_sizes

        inputs = {}
        for k, v in feats.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
                v = v.to(dtype=torch.float32)
            inputs[k] = v.to(self.device)

        # Get the segmentation masks
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        masks_tensor = self.sam_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes
        )[0]
        
        if masks_tensor.ndim == 4 and masks_tensor.shape[1] == 1:
            masks_tensor = masks_tensor.squeeze(1)
        else:
            masks_tensor = masks_tensor.squeeze()

        for i, detection in enumerate(detections):
            # Get the i-th mask and ensure it's a 2D numpy binary mask (H, W)
            mask_tensor = masks_tensor[i].cpu().squeeze()
            mask_np = mask_tensor.numpy()
            # If the mask has an unexpected channel dimension, collapse it to (H, W)
            if mask_np.ndim == 3:
                h, w = image.height, image.width
                if mask_np.shape == (h, w, mask_np.shape[2]):
                    mask_np = mask_np.max(axis=-1)
                elif mask_np.shape == (mask_np.shape[0], h, w):
                    mask_np = mask_np.max(axis=0)
                else:
                    # fallback: reduce the first axis
                    mask_np = mask_np.max(axis=0)
            # store as binary 0/1 uint8 (other code expects this)
            detection.mask = (mask_np > 0).astype(np.uint8)

        return detections

    def __get_boxes(self, detections: List[DetectionResult]) -> List[BoundingBox]:
        """
        Returns the bounding boxes of the detected objects.
        """
        if detections is None:
            return []
        return [result.box for result in detections]

    def draw_on_image(self, image: Image.Image, detections: List[DetectionResult]) -> Image.Image:
        """
        Draws the bounding boxes and masks on the image.
        """
        if not detections:
            return image

        image = image.copy() # Work on a copy of the image

        draw = ImageDraw.Draw(image)

        for detection in detections:
            # Draw bounding box
            draw.rectangle(detection.box.xyxy, outline="red", width=2)

            # Draw mask
            if detection.mask is not None:
                mask_image = Image.fromarray((detection.mask * 255).astype(np.uint8))
                image.paste(mask_image, (0, 0), mask_image)

        return image

    def blur_objects_in_image(self, image: Image.Image, detections: List[DetectionResult], radius: int = 25) -> Image.Image:
        """
        Blurs the objects in the image based on the detected masks.

        Args:
            radius (int): The radius of the blur effect.

        Returns:
            The final image with blurred objects.
        """
        # 1. Create a fully blurred version of the original image
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # 2. Create a combined mask for all detected objects
        combined_mask_np = np.zeros(np.array(image).shape[:2], dtype=np.uint8)

        if not detections:
            return image

        for detection in detections:
            if detection.mask is None:
                continue

            # Combine the current object's mask with the master mask.
            combined_mask_np = np.maximum(combined_mask_np, detection.mask)

        # 3. Convert the combined NumPy mask back to a PIL Image
        combined_mask_pil = Image.fromarray((combined_mask_np * 255).astype(np.uint8), mode='L')

        # 4. Composite the blurred image onto the original using the combined mask
        final_image = Image.composite(blurred_image, image, combined_mask_pil)

        return final_image

    def process_image(self, image: Image.Image, labels: List[str] = [], score_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Runs the full pipeline on a single image and returns a list of all
        individual detected entities with their own masks.

        Args:
            image (Image.Image): The input image to process.
            labels (List[str]): The list of entity names to detect.
            score_threshold (float): The confidence threshold for object detection.

        Returns:
            A dictionary containing the image shape and a list of individual entities.
        """
        width, height = image.size

        # 1. Detect all object instances
        detections = self.detect(image, labels, score_threshold)
        if not detections:
            logging.info("No objects detected.")
            return {"shape": (width, height), "entities": []}

        # 2. Segment each instance to get its individual mask
        segmented_detections = self.segment(image, detections)
        if not segmented_detections:
            return {"shape": (width, height), "entities": []}

        # 3. Format the final output, preserving each entity
        output_entities = []
        for i, detection in enumerate(segmented_detections):
            # Ensure the detection has a mask
            if detection.mask is None:
                continue

            output_entities.append({
                "entity_id": i,  # A simple unique ID for this image
                "entity_name": detection.label,
                "score": round(detection.score, 2),
                "box": detection.box.to_dict(), # Use the new helper method
                "mask": detection.mask # The individual (height, width) numpy array
            })

        return {
            "processed_image": self.blur_objects_in_image(image, segmented_detections),
            "shape": (width, height),
            "entities": output_entities
        }


# Example Workflow
if __name__ == "__main__":
    # Ensure your dataclasses and detect/draw_on_image functions are defined
    
    image_path = '/Users/nipunsamudrala/workspace/coding projects/python projects/ImageClassification/images/number plates/mercedes_number_plate.png'

    try:
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
    except FileNotFoundError:
        logging.error(f"Error: Image file not found at {image_path}. Please update the path.")
        exit()

    # --- Define labels to search for ---
    input_labels = ["person", "license plate", "chair"]

    # --- Initialize the pipeline ONCE ---
    print("Initializing GroundingDINO_SAMModel pipeline...")
    g_sam = GroundingDINO_SAMModel()

    # --- Process the image ---
    print(f"\nProcessing image for labels: {input_labels}")
    processed_data = g_sam.process_image(image=image, labels=input_labels)

    # --- Print the structured output ---
    print("\n--- Structured Output ---")
    # Using pprint for readable dictionary printing
    pprint(processed_data)

    # --- (Optional) Visualize the aggregated masks ---
    if processed_data['entities']:
        print("\nVisualizing aggregated masks...")
        # Create a copy of the image to draw on
        visual_image = image.copy()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] # Red, Green, Blue, Yellow

        for i, entity in enumerate(processed_data['entities']):
            mask_np = entity['mask']
            color = colors[i % len(colors)]
            
            # Create a colored mask image: shape (H, W, 3)
            colored_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
            colored_mask[mask_np == 1] = color
            
            # Convert to PIL Image for blending
            mask_image = Image.fromarray(colored_mask, 'RGB')
            
            # Blend the mask with the original image
            visual_image = Image.blend(visual_image, mask_image, alpha=0.1)

        visual_image.show(title="Aggregated Masks Visualization")

    ## Another Example

    # input_labels = ["person", "license plate", "sign"]

    # image = Image.open(image_path).convert("RGB")
    # image = ImageOps.exif_transpose(image)  # handle exif orientation

    # groundingSAM = GroundingSAM()

    # # 1. Detect objects with GroundingDINO
    # print("Step 1: Detecting objects...")
    # detections = groundingSAM.detect(image=image, labels=input_labels, score_threshold=0.3)

    # if not detections:
    #     print("No objects detected. Exiting.")
    # else:
    #     for detection in detections:
    #         print(detection)

    #     # Draw original bounding boxes for comparison
    #     bbox_image = groundingSAM.draw_on_image(image, detections)
    #     print("Showing image with original bounding boxes...")
    #     bbox_image.show()

    #     # 2. Segment the detected objects with SAM
    #     print("\nStep 2: Segmenting detected objects...")
    #     segmented_detections = groundingSAM.segment(image=image, detections=detections)

    #     # 3. Apply the blur effect using the generated masks
    #     print("\nStep 3: Applying blur effect...")
    #     blurred_image = blur_objects_in_image(image, segmented_detections)

    #     print("Showing final image with blurred objects...")
    #     blurred_image.show()