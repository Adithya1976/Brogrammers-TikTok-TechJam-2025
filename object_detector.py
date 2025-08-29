from dataclasses import dataclass
from transformers import AutoProcessor, AutoModelForMaskGeneration, pipeline
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import torch
from typing import List, Optional, Dict
import numpy as np


@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


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


class GroundingSAM:
    """
    A class for Grounding DINO with SAM (Segment Anything Model) integration.
    """
    def __init__(self, image: Image.Image, labels: List[str], device: torch.device | None = None) -> None:
        self.image = image
        self.labels = labels # list of text labels for grounding dino

        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = device

        self.detections = None

        # --- Use fast model names ---
        dino_model_name = "IDEA-Research/grounding-dino-tiny"
        sam_model_name = "facebook/sam-vit-base"

        # --- Load models and processors ---
        self.object_detector = pipeline(model=dino_model_name, task="zero-shot-object-detection", device=self.device, use_fast=True)

        self.sam_processor = AutoProcessor.from_pretrained(sam_model_name)
        self.sam_model = AutoModelForMaskGeneration.from_pretrained(sam_model_name).to(self.device)

    def detect(self, score_threshold: float = 0.25) -> List[DetectionResult]:
        """
        Runs a Grounding DINO object-detection model on the image using the provided labels.

        Args:
            score_threshold (float): The score threshold for filtering detections.

        Returns:
            Updates and returns the `self.detections` attribute.
        """
        # model_name = "IDEA-Research/grounding-dino-tiny"
        # object_detector = pipeline(model=model_name, task="zero-shot-object-detection", device=self.device, use_fast=True)

        labels = [label if label.endswith(".") else label+"." for label in self.labels]

        results = self.object_detector(self.image,  candidate_labels=labels, threshold=score_threshold)
        results = [DetectionResult.from_dict(result) for result in results]

        self.detections = results
        return self.detections

    def __get_boxes(self) -> List[BoundingBox]:
        """
        Returns the bounding boxes of the detected objects.
        """
        if self.detections is None:
            return []
        return [result.box for result in self.detections]

    def __get_masks(self) -> List[np.ndarray]:
        """
        Returns the masks of the detected objects.
        """
        if self.detections is None:
            return []
        return [result.mask for result in self.detections if result.mask is not None]

    def __get_labels(self) -> List[str]:
        """
        Returns the labels of the detected objects.
        """
        if self.detections is None:
            return []
        return [result.label for result in self.detections]

    def __get_scores(self) -> List[float]:
        """
        Returns the scores of the detected objects.
        """
        if self.detections is None:
            return []
        return [result.score for result in self.detections]

    def segment(self) -> List[DetectionResult] | None:
        """
        Segments the detected objects in the image using SAM.

        Returns:
            Updates the masks for each object in `self.detections` and returns the updated detections.
        """
        # model_name = "facebook/sam-vit-base"
        # segmentator = AutoModelForMaskGeneration.from_pretrained(model_name).to(self.device)
        # processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        # Prepare inputs for SAM
        # SAM's processor can take a list of bounding box lists. We have one image, so it's a list containing one list of boxes.
        input_boxes = [[box.xyxy for box in self.__get_boxes()]]
        feats = self.sam_processor(images=self.image, input_boxes=input_boxes, return_tensors="pt")
        # keep original/reshaped sizes for post-processing
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
        
        # Post-process to get masks in the original image size.
        masks_tensor = self.sam_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes
        )[0]

        # Normalize masks_tensor to shape (num_masks, H, W)
        # Common outputs: (num, 1, H, W) or (num, H, W). Remove singleton channel at axis=1 if present.
        if masks_tensor.ndim == 4 and masks_tensor.shape[1] == 1:
            masks_tensor = masks_tensor.squeeze(1)  # -> (num, H, W)
        else:
            masks_tensor = masks_tensor.squeeze()  # remove any extra singleton dims
        # now masks_tensor should be (num_masks, H, W)

        # Check if the number of masks matches the number of detections
        if not self.detections:
            return self.detections

        # if len(masks_tensor) != len(self.detections):
        #     print(f"Warning: Mismatch between number of detections ({len(self.detections)}) and masks ({len(masks_tensor)}).")
        #     return self.detections

        # Attach each mask to its corresponding detection object
        for i, detection in enumerate(self.detections):
            # Ensure mask_tensor is 2D (H, W)
            mask_tensor = masks_tensor[i].cpu().squeeze() # type: ignore
            # If a channel dimension remains (e.g., (H, W, C) or (C, H, W)), collapse it.
            if mask_tensor.ndim == 3:
                # if channels-first (C, H, W) -> collapse channels by max
                # if channels-last (H, W, C) converting to numpy below will still work but we collapse anyway
                try:
                    # channels-first: collapse dim 0
                    mask_tensor = mask_tensor.max(dim=0)[0]
                except Exception:
                    # fallback: convert to numpy then collapse last axis
                    m = mask_tensor.numpy()
                    mask_tensor = torch.from_numpy(m.max(axis=-1))

            # Binarize and convert to uint8 0/255
            mask_np = (mask_tensor.numpy() > 0).astype(np.uint8) * 255

            detection.mask = mask_np

        return self.detections

    def draw_on_image(self) -> Image.Image:
        """
        Draws the bounding boxes and masks on the image.
        """
        if self.detections is None:
            return self.image

        image = self.image.copy() # Work on a copy of the image

        draw = ImageDraw.Draw(image)

        for detection in self.detections:
            # Draw bounding box
            draw.rectangle(detection.box.xyxy, outline="red", width=2)

            # Draw mask
            if detection.mask is not None:
                mask_image = Image.fromarray((detection.mask * 255).astype(np.uint8))
                image.paste(mask_image, (0, 0), mask_image)

        return image

    # TODO: Check if this is needed
    def blur_objects_in_image(self, radius: int = 25) -> Image.Image:
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

        if not self.detections:
            return self.image

        for detection in self.detections:
            if detection.mask is None:
                continue

            # Combine the current object's mask with the master mask.
            combined_mask_np = np.maximum(combined_mask_np, detection.mask)

        # 3. Convert the combined NumPy mask back to a PIL Image
        combined_mask_pil = Image.fromarray(combined_mask_np, mode='L')

        # 4. Composite the blurred image onto the original using the combined mask
        final_image = Image.composite(blurred_image, image, combined_mask_pil)

        return final_image


def blur_objects_in_image(image: Image.Image, detections: List[DetectionResult] | None, radius: int = 25) -> Image.Image:
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
    combined_mask_pil = Image.fromarray(combined_mask_np, mode='L')

    # 4. Composite the blurred image onto the original using the combined mask
    final_image = Image.composite(blurred_image, image, combined_mask_pil)

    return final_image

# Example Workflow
if __name__ == "__main__":
    # Ensure your dataclasses and detect/draw_on_image functions are defined
    
    image_path = '/Users/nipunsamudrala/workspace/coding projects/python projects/ImageClassification/images/number plates/mercedes_number_plate.png'
    input_labels = ["person", "license plate", "sign"]

    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)  # handle exif orientation

    groundingSAM = GroundingSAM(image=image, labels=input_labels)

    # 1. Detect objects with GroundingDINO
    print("Step 1: Detecting objects...")
    detections = groundingSAM.detect(score_threshold=0.3)

    if not detections:
        print("No objects detected. Exiting.")
    else:
        for detection in detections:
            print(detection)

        # Draw original bounding boxes for comparison
        bbox_image = groundingSAM.draw_on_image()
        print("Showing image with original bounding boxes...")
        bbox_image.show()

        # 2. Segment the detected objects with SAM
        print("\nStep 2: Segmenting detected objects...")
        segmented_detections = groundingSAM.segment()

        # 3. Apply the blur effect using the generated masks
        print("\nStep 3: Applying blur effect...")
        blurred_image = blur_objects_in_image(image, segmented_detections)

        print("Showing final image with blurred objects...")
        blurred_image.show()
