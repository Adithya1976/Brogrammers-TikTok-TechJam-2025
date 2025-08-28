from transformers import AutoProcessor, AutoModelForObjectDetection
from PIL import Image
import torch

def detect_with_grounding_dino(image_path, prompt: str = "", device: torch.device | None = None, score_threshold: float = 0.25):
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

    # NOTE: replace this with a published Grounding DINO checkpoint on Hugging Face if needed
    model_name = "ShilongLiu/groundingdino"  # <-- replace with valid HF grounding dino checkpoint id
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name).to(device)

    # prepare inputs (include text prompt)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # forward
    outputs = model(**inputs)

    # post-process to get boxes in pixel coords (height, width expected)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)  # image.size -> (width, height); target_sizes expects (height, width)
    detections = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=score_threshold)[0]

    results = []
    for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
        results.append({
            "score": float(score),
            "label": model.config.id2label[int(label)],
            "box": [float(x) for x in box],  # [xmin, ymin, xmax, ymax]
        })

    return results

if __name__ == "__main__":
    image_path = '/Users/nipunsamudrala/workspace/coding projects/python projects/ImageClassification/photo_6159037021939680355_y.jpg'
    detections = detect_with_grounding_dino(image_path, prompt="", device=None)
    print("Detections:", detections)
