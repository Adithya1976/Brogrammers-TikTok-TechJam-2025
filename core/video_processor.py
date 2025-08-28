import cv2
import numpy as np
import tempfile
import os
from typing import List, Dict
from .privacy_detector import PrivacyDetector

class VideoProcessor:
    def __init__(self):
        self.privacy_detector = PrivacyDetector()
        
    def extract_frames(self, video_bytes: bytes, max_frames: int = 10) -> List[np.ndarray]:
        """Extract key frames from video for analysis"""
        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name
        
        try:
            cap = cv2.VideoCapture(temp_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract frames at regular intervals
            interval = max(1, total_frames // max_frames)
            
            frame_count = 0
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % interval == 0:
                    frames.append(frame)
                    
                frame_count += 1
            
            cap.release()
            return frames
            
        finally:
            os.unlink(temp_path)
    
    def process_video(self, video_bytes: bytes) -> Dict:
        """Process video for privacy concerns"""
        frames = self.extract_frames(video_bytes)
        
        all_entities = []
        all_text = []
        max_privacy_score = 0
        
        for i, frame in enumerate(frames):
            # Convert frame to bytes for processing
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Process frame
            result = self.privacy_detector.process_image(frame_bytes)
            
            if result["ocr_text"]:
                all_text.append(f"Frame {i}: {result['ocr_text']}")
            
            all_entities.extend(result["entities"])
            max_privacy_score = max(max_privacy_score, result["privacy_score"])
        
        # Remove duplicates
        unique_entities = list(set(all_entities))
        
        return {
            "frames_processed": len(frames),
            "ocr_text": " | ".join(all_text),
            "privacy_score": max_privacy_score,
            "is_safe": max_privacy_score < 5,
            "entities": unique_entities
        }