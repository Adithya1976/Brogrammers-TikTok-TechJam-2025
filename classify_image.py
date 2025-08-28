from dataclasses import dataclass
from platform import processor
from sympy import Li
from transformers import AutoProcessor, GroundingDinoForObjectDetection, AutoModelForMaskGeneration, pipeline
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import torch
from typing import Any, List, Optional, Dict, Tuple
import numpy as np
import cv2


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
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=float(detection_dict['box']['xmin']),
                                   ymin=float(detection_dict['box']['ymin']),
                                   xmax=float(detection_dict['box']['xmax']),
                                   ymax=float(detection_dict['box']['ymax'])))


def detect(image_path, input_labels: List[str], device: torch.device | None = None, score_threshold: float = 0.25) -> List[DetectionResult]:
    """
    Runs a Grounding DINO object-detection model on image_path using the provided prompt.
    - prompt: the text prompt used by grounding dino (use "" for now)
    - device: torch.device or None. If None, picks mps on mac, else cuda if available, else cpu.
    - score_threshold: detection score threshold returned by post-processing
    Returns: list of detections: dicts with keys 'score', 'label', 'box' (xyxy in image coordinates)
    """
    # choose device (mac will pick mps when available)
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)  # handle exif orientation

    '''# model_name = "IDEA-Research/grounding-dino-tiny"
    # processor = AutoProcessor.from_pretrained(model_name)
    # model = GroundingDinoForObjectDetection.from_pretrained(model_name).to(device)

    # # prepare inputs (include text prompt)
    # prompt = [label if label.endswith(".") else label+"." for label in input_labels]
    # inputs = processor(images=image, text=prompt, return_tensors="pt")
    # inputs = {k: v.to(device) for k, v in inputs.items()}

    # # forward
    # outputs = model(**inputs)

    # # post-process to get boxes in pixel coords (height, width expected)
    # target_sizes = torch.tensor([image.size[::-1]]).to(device)  # image.size -> (width, height); target_sizes expects (height, width)
    # detections = processor.post_process_grounded_object_detection(outputs, target_sizes=target_sizes, threshold=score_threshold)[0]

    # # Get readable labels robustly:
    # # - prefer "text_labels" (string names)
    # # - else try mapping integer ids via model.config.id2label
    # # - else fallback to str() of the value
    # raw_text_labels = detections.get("text_labels", None)
    # if raw_text_labels is not None:
    #     labels = list(raw_text_labels)
    # else:
    #     raw_labels = detections.get("labels", [])
    #     labels = []
    #     for rl in raw_labels:
    #         try:
    #             # rl might be a tensor/int id
    #             labels.append(model.config.id2label[int(rl)])
    #         except Exception:
    #             labels.append(str(rl))

    # def safe_score_to_float(s):
    #     try:
    #         return float(s.detach().cpu().item())
    #     except Exception:
    #         try:
    #             return float(s)
    #         except Exception:
    #             return None

    # results = []
    # for score, label, box in zip(detections["scores"], labels, detections["boxes"]):
    #     results.append({
    #         "score": safe_score_to_float(score),
    #         "label": label,
    #         "box": [float(x) for x in box],  # [xmin, ymin, xmax, ymax]
    #     })

    # return results'''

    model_name = "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=model_name, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in input_labels]

    results = object_detector(image,  candidate_labels=labels, threshold=score_threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results


def get_boxes(detections: List[DetectionResult]) -> List[BoundingBox]:
    return [d.box for d in detections]


def blur_objects_in_image(image: Image.Image, detections: List[DetectionResult], radius: int = 25) -> Image.Image:
    """
    Applies a Gaussian blur to the areas defined by the masks in the detection results.
    This version is robust to input masks having an extra dimension.

    Args:
        image (Image.Image): The original PIL image.
        detections (List[DetectionResult]): A list of detections, each must contain a mask.
        radius (int): The radius for the Gaussian blur.

    Returns:
        Image.Image: A new image with the detected objects blurred.
    """
    # 1. Create a fully blurred version of the original image
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    # 2. Create a combined mask for all detected objects
    combined_mask_np = np.zeros(np.array(image).shape[:2], dtype=np.uint8)

    for detection in detections:
        if detection.mask is None:
            continue

        # Defensive handling: ensure mask_tensor is 2D (H, W)
        mask_tensor = detection.mask.cpu().squeeze()
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

        # Combine the current object's mask with the master mask.
        combined_mask_np = np.maximum(combined_mask_np, mask_np)

    # 3. Convert the combined NumPy mask back to a PIL Image
    combined_mask_pil = Image.fromarray(combined_mask_np, mode='L')

    # 4. Composite the blurred image onto the original using the combined mask
    final_image = Image.composite(blurred_image, image, combined_mask_pil)

    return final_image


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


def draw_on_image(image_path, detections: List[DetectionResult]):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)  # handle exif orientation
    draw = ImageDraw.Draw(image)

    # Get Boxes
    boxes = get_boxes(detections)
    for box in boxes:
        draw.rectangle(box.xyxy, outline="red", width=3)

    return image


def segment(image_path: str, detections: List[DetectionResult], device: torch.device | None = None) -> List[DetectionResult]:
    """
    Generates segmentation masks for given detections using SAM.

    Args:
        image_path (str): Path to the image file.
        detections (List[DetectionResult]): Detections from GroundingDINO.
        device (torch.device, optional): The device to run the model on.

    Returns:
        List[DetectionResult]: The input detections, updated with a `mask` attribute.
    """
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)  # handle exif orientation

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load the segmentation model and processor
    model_name = "facebook/sam-vit-base"
    segmentator = AutoModelForMaskGeneration.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    # Prepare inputs for SAM
    # SAM's processor can take a list of bounding box lists. We have one image, so it's a list containing one list of boxes.
    input_boxes = [[box.xyxy for box in get_boxes(detections)]]
    feats = processor(images=image, input_boxes=input_boxes, return_tensors="pt")
    # keep original/reshaped sizes for post-processing
    original_sizes = feats.original_sizes
    reshaped_input_sizes = feats.reshaped_input_sizes

    # Build a device-ready dict: convert float64 tensors -> float32 (MPS does not support float64)
    inputs = {}
    for k, v in feats.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
            v = v.to(dtype=torch.float32)
        inputs[k] = v.to(device)
 
    # Get the segmentation masks
    with torch.no_grad():
        outputs = segmentator(**inputs)
    
    # Post-process to get masks in the original image size.
    masks_tensor = processor.post_process_masks(
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
    if len(masks_tensor) != len(detections):
        print(f"Warning: Mismatch between number of detections ({len(detections)}) and masks ({len(masks_tensor)}).")
        return detections

    # Attach each mask to its corresponding detection object
    for i, detection in enumerate(detections):
        detection.mask = masks_tensor[i]

    return detections

if __name__ == "__main__":
    # Ensure your dataclasses and detect/draw_on_image functions are defined
    
    image_path = '/Users/nipunsamudrala/workspace/coding projects/python projects/ImageClassification/images/people/windows-p74ndnYWRY4-unsplash.jpg'
    input_labels = ["face", "laptop"]
    
    # 1. Detect objects with GroundingDINO
    print("Step 1: Detecting objects...")
    detections = detect(image_path, input_labels=input_labels, device=None)
    
    if not detections:
        print("No objects detected. Exiting.")
    else:
        for detection in detections:
            print(detection)

        # Draw original bounding boxes for comparison
        bbox_image = draw_on_image(image_path, detections)
        print("Showing image with original bounding boxes...")
        bbox_image.show()

        # 2. Segment the detected objects with SAM
        print("\nStep 2: Segmenting detected objects...")
        detections_with_masks = segment(image_path, detections)

        # 3. Apply the blur effect using the generated masks
        print("\nStep 3: Applying blur effect...")
        original_image = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
        blurred_image = blur_objects_in_image(original_image, detections_with_masks, radius=25)
        
        print("Showing final image with blurred objects...")
        blurred_image.show()
