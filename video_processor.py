import cv2
import imagehash
import numpy as np
from PIL import Image, ImageFilter
from typing import List
from object_detector import GroundingSAM, DetectionResult, BoundingBox, blur_objects_in_image


def get_keyframes(video_path: str, threshold: int = 5) -> List[int]:
    """
    Extracts keyframes from a video based on perceptual hash similarity.

    A frame is considered a keyframe if its content has changed significantly
    compared to the last selected keyframe.

    Args:
        video_path (str): The path to the video file.
        threshold (int): The Hamming distance threshold. A higher value means
                         less sensitivity to change (fewer keyframes). A good
                         starting value is 5.

    Returns:
        List[int]: A list of frame numbers that are keyframes.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    keyframes = []
    last_hash = None
    frame_number = 0

    while True:
        success, frame = video.read()
        if not success:
            break  # End of video

        # The first frame is always a keyframe
        if frame_number == 0:
            # Convert frame (NumPy array from OpenCV) to a PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            last_hash = imagehash.phash(pil_image)
            keyframes.append(frame_number)
            print(f"Frame {frame_number} selected (first frame)")

        else:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_hash = imagehash.phash(pil_image)

            # Calculate the Hamming distance
            hash_diff = current_hash - last_hash # type: ignore

            if hash_diff > threshold:
                keyframes.append(frame_number)
                last_hash = current_hash
                print(f"Frame {frame_number} selected (hash diff: {hash_diff})")

        frame_number += 1

    video.release()
    return keyframes

# Classical Method - GroundingDino + CSRT Tracking
# Faster, lesser memory but blurs out entire bounding box
def blur_region_in_image(image: Image.Image, box: tuple, method='pixelate', **kwargs) -> Image.Image:
    """A helper to blur a single rectangular region in an image."""
    x1, y1, w, h = [int(v) for v in box]
    x2, y2 = x1 + w, y1 + h
    
    # Crop the region of interest
    roi = image.crop((x1, y1, x2, y2))
    
    if method == 'pixelate':
        pixel_size = kwargs.get('pixel_size', 30)
        small_roi = roi.resize((w // pixel_size, h // pixel_size), resample=Image.Resampling.NEAREST)
        effect_roi = small_roi.resize(roi.size, resample=Image.Resampling.NEAREST)
    else: # Default to gaussian
        radius = kwargs.get('radius', 25)
        effect_roi = roi.filter(ImageFilter.GaussianBlur(radius=radius))
        
    # Paste the modified ROI back into the image
    image.paste(effect_roi, (x1, y1))
    return image


def process_video_with_csrt(video_path: str, output_path: str, labels: List[str]):
    """
    Processes a video with a robust "Track-then-Segment" strategy,
    including a "last known position" fallback to prevent flickering.
    """
    print("Step 1: Finding keyframes for re-detection...")
    keyframe_indices = get_keyframes(video_path, threshold=7)
    print(f"\nFound {len(keyframe_indices)} keyframes for re-detection.")

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_number = 0
    trackers = []
    
    # --- FIX 1: Add a variable to store the last good state ---
    last_known_detections = []

    # # Initialize model once for efficiency
    # grounding_sam_pipeline = GroundingSAM(labels=labels)

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        grounding_sam_pipeline = GroundingSAM(image=current_pil_image, labels=labels)

        if frame_number in keyframe_indices:
            print(f"\n--- Keyframe {frame_number}: Running Full Detection & Segmentation ---")
            trackers = [] # Reset trackers
            detections = grounding_sam_pipeline.detect()

            if detections:
                segmented_detections = grounding_sam_pipeline.segment()
                processed_pil_image = blur_objects_in_image(current_pil_image, segmented_detections)
                
                # Update our last known state with these high-quality detections
                last_known_detections = segmented_detections

                for d in detections:
                    box = d.box.xyxy
                    tracker_box = (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))
                    tracker = cv2.TrackerCSRT_create() # type: ignore
                    tracker.init(frame, tracker_box)
                    trackers.append(tracker)
            else:
                # No detections, so clear the last known state and trackers
                last_known_detections = []
                trackers = []
                processed_pil_image = current_pil_image
        
        else: # This is an intermediate frame
            tracked_detections = []
            if trackers:
                for tracker in trackers:
                    success, box = tracker.update(frame)
                    if success:
                        x1, y1, w, h = box
                        tracked_box = BoundingBox(xmin=x1, ymin=y1, xmax=x1 + w, ymax=y1 + h)
                        tracked_detections.append(DetectionResult(score=0.99, label="tracked", box=tracked_box))

            # --- FIX 2: The Core Logic Change ---
            # If we have successfully tracked objects, use them.
            if tracked_detections:
                grounding_sam_pipeline.detections = tracked_detections
                segmented_detections = grounding_sam_pipeline.segment()
                processed_pil_image = blur_objects_in_image(current_pil_image, segmented_detections)
                # Update our last known state with the new tracked positions
                last_known_detections = segmented_detections
            # ELSE, if tracking failed but we have a "last known good position", USE IT!
            elif last_known_detections:
                print(f"Frame {frame_number}: Tracker failed. Using last known position.")
                # We don't need to re-segment, just apply the last known masks to the new frame.
                # This assumes your blur function can take masks from a previous frame.
                processed_pil_image = blur_objects_in_image(current_pil_image, last_known_detections)
            # Otherwise, there's nothing to blur.
            else:
                processed_pil_image = current_pil_image

        # Write the final frame to the output video
        output_frame_bgr = cv2.cvtColor(np.array(processed_pil_image), cv2.COLOR_RGB2BGR)
        writer.write(output_frame_bgr)

        frame_number += 1
        if frame_number % fps == 0:
            print(f"Processed {frame_number} frames...")

    print("Finished processing. Releasing video resources.")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_file_path = '/Users/nipunsamudrala/workspace/coding projects/python projects/ImageClassification/videos/bedroom.mp4'
    output_video_path = '/Users/nipunsamudrala/workspace/coding projects/python projects/ImageClassification/videos/bedroom_anonymized.mp4'
    labels_to_blur = ["face"]

    process_video_with_csrt(video_file_path, output_video_path, labels_to_blur)
