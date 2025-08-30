import base64
from collections import defaultdict
import cv2
import imagehash
import numpy as np
import logging
from PIL import Image
from typing import List, Dict, Any, Tuple
from core.privacy_detector import PrivacyDetector
from core.object_detector import GroundingDINO_SAMModel, DetectionResult, BoundingBox


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
            logging.info(f"Frame {frame_number} selected (first frame)")

        else:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_hash = imagehash.phash(pil_image)

            # Calculate the Hamming distance
            hash_diff = current_hash - last_hash # type: ignore

            if hash_diff > threshold:
                keyframes.append(frame_number)
                last_hash = current_hash
                logging.info(f"Frame {frame_number} selected (hash diff: {hash_diff})")

        frame_number += 1

    video.release()
    return keyframes


def process_video_with_csrt(video_path: str, output_path: str, labels: List[str]):
    """
    Processes a video with a robust "Track-then-Segment" strategy,
    including a "last known position" fallback to prevent flickering.
    """
    logging.info("Step 1: Finding keyframes for re-detection...")
    keyframe_indices = get_keyframes(video_path, threshold=7)
    logging.info(f"\nFound {len(keyframe_indices)} keyframes for re-detection.")

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

    # Initialize model once for efficiency
    grounding_sam_pipeline = GroundingDINO_SAMModel()

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if frame_number in keyframe_indices:
            logging.info(f"\n--- Keyframe {frame_number}: Running Full Detection & Segmentation ---")
            trackers = [] # Reset trackers
            detections = grounding_sam_pipeline.detect(image=current_pil_image, labels=labels)

            if detections:
                segmented_detections = grounding_sam_pipeline.segment(image=current_pil_image, detections=detections)
                processed_pil_image = grounding_sam_pipeline.blur_objects_in_image(current_pil_image, segmented_detections) # type: ignore
                
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
                segmented_detections = grounding_sam_pipeline.segment(image=current_pil_image, detections=tracked_detections)
                processed_pil_image = grounding_sam_pipeline.blur_objects_in_image(current_pil_image, segmented_detections) # type: ignore
                # Update our last known state with the new tracked positions
                last_known_detections = segmented_detections
            # ELSE, if tracking failed but we have a "last known good position", USE IT!
            elif last_known_detections:
                logging.info(f"Frame {frame_number}: Tracker failed. Using last known position.")
                # We don't need to re-segment, just apply the last known masks to the new frame.
                # This assumes your blur function can take masks from a previous frame.
                processed_pil_image = grounding_sam_pipeline.blur_objects_in_image(current_pil_image, last_known_detections)
            # Otherwise, there's nothing to blur.
            else:
                processed_pil_image = current_pil_image

        # Write the final frame to the output video
        output_frame_bgr = cv2.cvtColor(np.array(processed_pil_image), cv2.COLOR_RGB2BGR) # type: ignore
        writer.write(output_frame_bgr)

        frame_number += 1
        if frame_number % fps == 0:
            logging.info(f"Processed {frame_number} frames...")

    logging.info("Finished processing. Releasing video resources.")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


def calculate_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA.xmin, boxB.xmin)
    yA = max(boxA.ymin, boxB.ymin)
    xB = min(boxA.xmax, boxB.xmax)
    yB = min(boxA.ymax, boxB.ymax)

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    # Compute the area of both bounding boxes
    boxAArea = (boxA.xmax - boxA.xmin) * (boxA.ymax - boxA.ymin)
    boxBArea = (boxB.xmax - boxB.xmin) * (boxB.ymax - boxB.ymin)

    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def create_debug_video(
    original_video_path: str, 
    entity_data: Dict[str, Any], 
    output_path: str, 
    alpha: float = 0.4
):
    """
    Creates a debug video by superimposing entity masks onto the original video.

    This is a post-processing step and does not run any AI models.

    Args:
        original_video_path (str): Path to the source video file.
        entity_data (Dict[str, Any]): The output from VideoProcessor.process_video.
        output_path (str): Path to save the annotated output video.
        alpha (float): The transparency level for the mask overlay.
    """
    logging.info(f"Starting debug video creation for {original_video_path}")
    
    cap = cv2.VideoCapture(original_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Define a list of distinct colors (BGR format for OpenCV)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 0, 255), (0, 128, 255)
    ]

    # --- Pre-process entity data for fast lookup ---
    # Create a map from frame_number -> list of (entity, mask_index)
    frame_to_entities_map = defaultdict(list)
    for entity in entity_data['entities']:
        for i, frame_num in enumerate(range(entity['start_frame'], entity['end_frame'] + 1)):
            frame_to_entities_map[frame_num].append((entity, i)) # Store entity and its mask index

    frame_number = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Create a copy for the overlay to keep the original clean
        overlay = frame.copy()

        # Check if there are any active entities for this frame
        if frame_number in frame_to_entities_map:
            for entity, mask_index in frame_to_entities_map[frame_number]:
                entity_id = entity['entity_id']
                mask = entity['mask_video'][mask_index]
                color = colors[entity_id % len(colors)]

                # Apply the colored mask to the overlay
                # Note: mask is (H, W), we need to make it (H, W, 3) for coloring
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask == 1] = color
                overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
                
                # --- Add bounding box and label ---
                # Find contours to get the bounding box of the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Get the largest contour in case of fragmented masks
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Draw rectangle on the final frame (not the overlay)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Prepare and draw text label
                    label_text = f"ID: {entity_id} ({entity['label']})"
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
                    cv2.putText(frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Blend the original frame with the overlay
        final_frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        # Re-draw the bounding box and text on top of the blended image
        # (This is a style choice, makes the text clearer)
        if frame_number in frame_to_entities_map:
            for entity, _ in frame_to_entities_map[frame_number]:
                 # Code to re-draw text/box can be duplicated here if needed
                 # for now, the previous draw is on the 'frame' which is blended, which is fine.
                 pass
                 
        writer.write(final_frame)

        if frame_number % int(fps) == 0:
            logging.info(f"Written {frame_number} frames to debug video...")
        frame_number += 1
    
    logging.info(f"Debug video successfully saved to {output_path}")
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


def upsample_entity_masks(entity_data: Dict[str, Any], original_width: int, original_height: int) -> Dict[str, Any]:
    """
    Upsamples the mask_video for each entity to the original video resolution.

    Args:
        entity_data (Dict[str, Any]): The data object from VideoProcessor.
        original_width (int): The original width of the video.
        original_height (int): The original height of the video.

    Returns:
        The entity_data dictionary with masks scaled up.
    """
    if not entity_data or not entity_data['entities']:
        return entity_data

    logging.info(f"Upsampling masks to original resolution: {original_width}x{original_height}")
    
    for entity in entity_data['entities']:
        low_res_masks = entity['mask_video']
        num_masks, low_h, low_w = low_res_masks.shape
        
        # Skip if already at original resolution (or if something is wrong)
        if low_h == original_height and low_w == original_width:
            continue

        # Create a new array to hold the upsampled masks
        high_res_masks = np.zeros((num_masks, original_height, original_width), dtype=np.uint8)
        
        for i in range(num_masks):
            # Use INTER_NEAREST for binary masks to avoid creating blurry grey values
            high_res_masks[i] = cv2.resize(
                low_res_masks[i],
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST
            )
        
        entity['mask_video'] = high_res_masks
        
    return entity_data


class VideoProcessor:
    """
    Processes a video to detect, track, and generate masks for both
    visual (GroundingSAM) and text-based (OCR) entities.
    """
    def __init__(self, grounding_sam: GroundingDINO_SAMModel, privacy_detector: PrivacyDetector, labels: List[str] = []):
        self.visual_labels = labels
        self.grounding_sam = grounding_sam

        self.privacy_detector = privacy_detector
        self.video_info = {}

    def process_video(self, video_path: str, max_dimension: int = 720, keyframe_threshold: int = 7, iou_threshold: float = 0.5) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_info = {"width": original_width, "height": original_height, "fps": fps, "frame_count": frame_count}

        # --- Calculate processing dimensions ---
        if original_width > original_height:
            # Landscape or square
            processing_width = max_dimension
            processing_height = int(original_height * (max_dimension / original_width))
        else:
            # Portrait
            processing_height = max_dimension
            processing_width = int(original_width * (max_dimension / original_height))
        
        logging.info(f"Original resolution: {original_width}x{original_height}")
        logging.info(f"Processing at downscaled resolution: {processing_width}x{processing_height}")

        keyframe_indices = get_keyframes(video_path, threshold=keyframe_threshold)

        tracked_entities: Dict[int, Dict[str, Any]] = {}
        next_entity_id = 0
        
        frame_number = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            # --- DOWNSAMPLE FRAME ---
            resized_frame = cv2.resize(frame, (processing_width, processing_height), interpolation=cv2.INTER_AREA)
            current_pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

            if frame_number in keyframe_indices:
                logging.info(f"--- Keyframe {frame_number}: Detecting and Re-identifying entities ---")

                # Get detections from both pipelines
                visual_detections = self.grounding_sam.detect(image=current_pil_image, labels=self.visual_labels)
                text_detections_dicts = self.privacy_detector.process_image(current_pil_image)
                all_processed_detections: List[DetectionResult] = []

                # Process and segment visual detections
                if visual_detections:
                    segmented_visuals = self.grounding_sam.segment(current_pil_image, visual_detections)
                    if segmented_visuals:
                        for det in segmented_visuals:
                            setattr(det, 'type', 'visual') # Tag entity type
                            all_processed_detections.append(det)
                
                # Process and create box masks for text detections
                for det_dict in text_detections_dicts:
                    for box in det_dict['boxes']:
                        text_dr = DetectionResult(
                            score=det_dict.get('score', 0.95),
                            label=det_dict['entity_name'],
                            box=box,
                            mask=self._create_box_mask(box, processing_height, processing_width) # Generate box mask
                        )
                        setattr(text_dr, 'type', 'text') # Tag entity type
                        all_processed_detections.append(text_dr)

                # --- Re-Identification Logic ---
                if not all_processed_detections:
                    for entity in tracked_entities.values(): entity['active'] = False
                else:
                    active_entity_ids = [eid for eid, e in tracked_entities.items() if e['active']]
                    matched_detection_indices = set()

                    for entity_id in active_entity_ids:
                        entity = tracked_entities[entity_id]
                        best_match_iou, best_match_idx = -1, -1
                        for i, det in enumerate(all_processed_detections):
                            iou = calculate_iou(entity['box'], det.box)
                            if iou > iou_threshold and iou > best_match_iou:
                                best_match_iou, best_match_idx = iou, i
                        
                        if best_match_idx != -1:
                            detection = all_processed_detections[best_match_idx]
                            # Update entity tracker and data
                            tracker = cv2.TrackerCSRT_create() # type: ignore
                            tracker_box = (int(detection.box.xmin), int(detection.box.ymin), int(detection.box.xmax - detection.box.xmin), int(detection.box.ymax - detection.box.ymin))
                            tracker.init(resized_frame, tracker_box)
                            entity.update({'tracker': tracker, 'box': detection.box})
                            entity['masks'].append((frame_number, detection.mask))
                            matched_detection_indices.add(best_match_idx)
                        else:
                            entity['active'] = False
                    
                    for i, det in enumerate(all_processed_detections):
                        if i not in matched_detection_indices:
                            # This is a new entity
                            tracker = cv2.TrackerCSRT_create() # type: ignore
                            tracker_box = (int(det.box.xmin), int(det.box.ymin), int(det.box.xmax - det.box.xmin), int(det.box.ymax - det.box.ymin))
                            tracker.init(resized_frame, tracker_box)
                            
                            # Store the entity type
                            tracked_entities[next_entity_id] = {
                                'label': det.label,
                                'type': getattr(det, 'type'), # Get the type we tagged earlier
                                'tracker': tracker,
                                'box': det.box,
                                'masks': [(frame_number, det.mask)],
                                'active': True
                            }
                            logging.info(f"New entity {next_entity_id} ({det.label}, type: {getattr(det, 'type')}) detected.")
                            next_entity_id += 1
            else:
                # --- Intermediate Frame: Update trackers and generate masks ---
                active_entities = [e for e in tracked_entities.values() if e['active']]
                if active_entities:
                    boxes_to_process, entity_map = [], []
                    for entity in active_entities:
                        success, box = entity['tracker'].update(resized_frame)
                        if success:
                            x1, y1, w, h = [int(v) for v in box]
                            entity['box'] = BoundingBox(xmin=x1, ymin=y1, xmax=x1 + w, ymax=y1 + h)
                            boxes_to_process.append(DetectionResult(score=0.99, label=entity['label'], box=entity['box']))
                            entity_map.append(entity)
                        else:
                            entity['active'] = False

                    ## INTEGRATION: Split processing based on entity type
                    visual_entities_to_segment = []
                    visual_entity_map = []

                    for i, det_result in enumerate(boxes_to_process):
                        entity = entity_map[i]
                        if entity['type'] == 'visual':
                            visual_entities_to_segment.append(det_result)
                            visual_entity_map.append(entity)
                        elif entity['type'] == 'text':
                            # For text, just generate the box mask directly
                            box_mask = self._create_box_mask(det_result.box, processing_height, processing_width)
                            entity['masks'].append((frame_number, box_mask))
                    
                    # Run batch segmentation ONLY on visual entities
                    if visual_entities_to_segment:
                        segmented_results = self.grounding_sam.segment(current_pil_image, visual_entities_to_segment)
                        if segmented_results:
                            for i, seg_det in enumerate(segmented_results):
                                visual_entity_map[i]['masks'].append((frame_number, seg_det.mask))
            
            if frame_number % fps == 0: logging.info(f"Processed {frame_number}/{frame_count} frames...")
            frame_number += 1
            
        cap.release()
        return self._finalize_output(tracked_entities)

    def _finalize_output(self, tracked_entities: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upscale (if needed) and structure the final output data.
        Args:
            tracked_entities (Dict[int, Dict[str, Any]]): The tracked entities data.
        Returns:
            Dict[str, Any]: The structured output data.
        """
        output_entities = []
        upsampled_entities = upsample_entity_masks(entity_data=tracked_entities, original_width=self.video_info['width'], original_height=self.video_info['height'])
        for entity_id, entity_data in upsampled_entities.items():
            if not entity_data['masks']: continue
            frame_nums, masks = zip(*entity_data['masks'])
            mask_video = np.stack(masks, axis=0)

            # Add entity type to the final output
            output_entities.append({
                "entity_name": entity_data['label'],
                "start_frame": min(frame_nums),
                "end_frame": max(frame_nums),
                "mask_video": base64.b64encode(mask_video.tobytes()).decode("ascii")
            })
        
        return {
            "shape": (self.video_info['width'], self.video_info['height'], self.video_info['frame_count']),
            "entities": output_entities
        }

    def _ocr_to_bounding_box(self, ocr_box: List[Tuple[int, int]]) -> BoundingBox:
        """Converts an OCR box object to a BoundingBox object. The OCR box object
        is a list of 4 coordinates ([x1, y1), (x2, y2), (x3, y3), (x4, y4)]."""
        return BoundingBox(
            xmin=min(box[0] for box in ocr_box),
            ymin=min(box[1] for box in ocr_box),
            xmax=max(box[0] for box in ocr_box),
            ymax=max(box[1] for box in ocr_box)
        )

    def _create_box_mask(self, box: BoundingBox, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        xmin, ymin, xmax, ymax = [int(v) for v in box.xyxy]
        mask[ymin:ymax, xmin:xmax] = 1
        return mask


if __name__ == '__main__':
    video_file_path = '/Users/adithyasamudrala/Brogrammers-TikTok-TechJam-2025/bedroom.mp4'
    labels_to_process = []

    # --- STAGE 1: EXPENSIVE PROCESSING ---
    processor = VideoProcessor(labels=labels_to_process)
    individual_entity_data = processor.process_video(video_file_path)

    # --- STAGE 2: INSPECT THE OUTPUT ---
    print("\n--- Final Structured Output for Individual Entities ---")
    
    # Print overall structure
    print(f"Video Shape: {individual_entity_data['shape']}")
    print(f"Total entities found: {len(individual_entity_data['entities'])}")
    
    # Print details for each entity
    for entity in individual_entity_data['entities']:
        print("-" * 20)
        print(f"  Entity ID: {entity['entity_id']}")
        print(f"  Label: {entity['label']}")
        print(f"  Present from frame {entity['start_frame']} to {entity['end_frame']}")
        print(f"  Mask Video Shape: {entity['mask_video'].shape}") # Should be (end-start+1, H, W) or similar
        print("-" * 20)

    output_video_path = 'bedroom_processed.mp4'
    create_debug_video(original_video_path=video_file_path, entity_data=individual_entity_data, output_path=output_video_path)
