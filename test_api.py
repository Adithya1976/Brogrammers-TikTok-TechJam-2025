#!/usr/bin/env python3
"""
Test the TasksAI Privacy Pipeline API (matches 202 + job polling behavior).

Usage:
  # Process a single file with a 20-minute timeout
  python test_api.py --wait 1200 my_video.MOV

  # Process multiple files as a single batch job
  python test_api.py --batch file1.jpg file2.MOV
"""
import argparse
import base64
import json
import mimetypes
import os
import time
from typing import Dict, Optional

import cv2
import numpy as np
import requests


def join_url(base: str, path: str) -> str:
    """Construct a full URL from a base and a path."""
    base = base.rstrip("/")
    path = path if path.startswith("/") else f"/{path}"
    return f"{base}{path}"


def get_health(base_url: str, timeout: float = 5.0) -> Optional[Dict]:
    """Check the health of the API."""
    try:
        r = requests.get(join_url(base_url, "/api/health"), timeout=timeout)
        if r.status_code == 200:
            return r.json()
        print(f"   ❌ /api/health failed: HTTP {r.status_code}")
        try:
            print(f"   Body: {r.text}")
        except Exception:
            pass
    except requests.exceptions.ConnectionError:
        print("   ❌ Server not running. Start it (e.g.,: python start_server.py)")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    return None


def detect_mime(path: str) -> str:
    """Detect the MIME type of a file."""
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"


def poll_job(base_url: str, job_id: str, max_wait: float = 180.0, tick: float = 1.0) -> Dict:
    """Poll a job until it's done or an error occurs."""
    url = join_url(base_url, f"/jobs/{job_id}")
    start = time.time()
    last_status = None
    while True:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"Job GET failed: HTTP {r.status_code} {r.text}")
        payload = r.json()
        status = payload.get("status")
        if status != last_status:
            print(f"   ⏱️  Job status: {status}")
            last_status = status
        if status in ("done", "error"):
            return payload
        if time.time() - start > max_wait:
            raise TimeoutError(f"Timed out waiting for job {job_id}")
        time.sleep(tick)


def overlay_video_masks(video_path: str, result_data: dict) -> None:
    """
    Overlay masks from the API result onto the original video and save a new file.
    """
    print(f"\n🎬 Processing video masks for {video_path}...")
    entities = result_data.get("entities")
    shape = result_data.get("shape")

    if not shape or len(shape) not in [2, 3]:
        print(f"   ❌ Invalid 'shape' in result for video processing. Expected 2 or 3 elements, but got: {shape}")
        return

    original_width, original_height = shape[:2]

    if not entities:
        print("   ⚠️ No entities with masks found in the result.")
        return

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"   ❌ Error opening video file: {video_path}")
        return

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    output_filename = f"{os.path.splitext(video_path)[0]}_processed.mp4"
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (original_width, original_height))

    try:
        # Create a blank composite mask that will be applied to all frames.
        # This approach simplifies the demo by creating one static mask for the whole video.
        combined_mask_for_video = np.zeros((original_height, original_width), dtype=np.uint8)
        for entity in entities:
            mask_b64 = entity.get("mask_video")
            if mask_b64:
                mask_bytes = base64.b64decode(mask_b64)
                # Reshape the decoded bytes into the multi-frame mask array
                mask_array = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(-1, original_height, original_width)
                # Combine all masks for this entity across all frames into one static mask
                static_entity_mask = np.any(mask_array, axis=0).astype(np.uint8)
                # Add this entity's static mask to the main composite mask
                np.bitwise_or(combined_mask_for_video, static_entity_mask, out=combined_mask_for_video)

        combined_mask_bool = combined_mask_for_video > 0

    except Exception as e:
        print(f"   ❌ Failed to decode or process masks: {e}")
        video_capture.release()
        video_writer.release()
        import traceback
        traceback.print_exc()
        return

    print(f"   ✅ Decoded masks. Now applying to video frames...")
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Apply the combined static mask to every frame
        frame[combined_mask_bool] = [0, 0, 0]  # Set masked area to black

        video_writer.write(frame)
        frame_count += 1

    print(f"   ✅ Processed {frame_count} frames.")
    print(f"   💾 Saved processed video to: {output_filename}")

    video_capture.release()
    video_writer.release()


def test_single_file(base_url: str, path: str, timeout: float = 10.0, max_wait: float = 180.0) -> None:
    """Test processing for a single file."""
    if not os.path.exists(path):
        print(f"   ⚠️  {path} not found")
        return

    print(f"\n▶️  Testing single file: {path}")
    ctype = detect_mime(path)
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, ctype)}
        r = requests.post(join_url(base_url, "/process"), files=files, timeout=timeout)

    if r.status_code == 202:
        try:
            data = r.json()
        except Exception:
            print(f"   ❌ Invalid JSON in 202 response. Body: {r.text}")
            return

        job_id = data.get("job_id")
        print(f"   ✅ Enqueued (job_id={job_id})")

        if not job_id:
            print("   ❌ Missing job_id in response; cannot poll.")
            return

        result = poll_job(base_url, job_id, max_wait=max_wait)
        print("   🎯 Final job payload:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # --- VIDEO POST-PROCESSING HOOK ---
        if result.get("status") == "done" and ctype.startswith("video/"):
            overlay_video_masks(path, result.get("result", {}))
        return

    if r.status_code == 200:
        print(f"   ⚠️ Received 200 OK (immediate result). Body: {r.text}")
        return

    print(f"   ❌ Upload failed: HTTP {r.status_code}. Body: {r.text}")


def test_batch(base_url: str, paths: list[str], timeout: float = 30.0, max_wait: float = 600.0) -> None:
    """Test processing for a batch of files."""
    print(f"\n▶️  Testing batch upload: {len(paths)} files")
    files = []
    to_close = []
    try:
        for p in paths:
            if not os.path.exists(p):
                print(f"   ⚠️  Skipping missing file: {p}")
                continue
            f = open(p, "rb")
            to_close.append(f)
            files.append(("files", (os.path.basename(p), f, detect_mime(p))))

        if not files:
            print("   ❌ No valid files for batch.")
            return

        r = requests.post(join_url(base_url, "/api/process-batch"), files=files, timeout=timeout)

        if r.status_code != 202:
            print(f"   ❌ Batch enqueue failed: HTTP {r.status_code}. Body: {r.text}")
            return

        data = r.json()
        job_id = data.get("job_id")
        print(f"   ✅ Batch enqueued (job_id={job_id})")
        if not job_id:
            print("   ❌ Missing job_id in batch response; cannot poll.")
            return

        result = poll_job(base_url, job_id, max_wait=max_wait)
        print("   🎯 Final batch job payload:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # --- VIDEO POST-PROCESSING HOOK FOR BATCH ---
        if result.get("status") == "done":
            video_files_in_batch = [p for p in paths if detect_mime(p).startswith("video/")]
            if video_files_in_batch:
                print(f"\n🎬 Batch job complete. Applying results to video files...")
                # This assumes the result payload can be applied to each video.
                # A more complex API might return results per file.
                for video_path in video_files_in_batch:
                    overlay_video_masks(video_path, result.get("result", {}))

    finally:
        for f in to_close:
            try:
                f.close()
            except Exception:
                pass


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test TasksAI Privacy Pipeline API")
    parser.add_argument("--base", dest="base_url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--batch", action="store_true", help="Send all provided files as a single batch job")
    parser.add_argument("--wait", type=float, default=180.0, help="Max seconds to wait for a job")
    parser.add_argument("files", nargs="+", help="Image/video files to upload")
    args = parser.parse_args()

    print("🧪 Testing TasksAI Privacy Pipeline API")
    print("=" * 40)

    print("1. Testing health check...")
    if not get_health(args.base_url):
        return

    if args.batch:
        # For batch, we'll use a longer default wait time if the user doesn't specify one
        batch_wait_time = args.wait if args.wait != 180.0 else 600.0
        test_batch(args.base_url, args.files, max_wait=batch_wait_time)
    else:
        for idx, path in enumerate(args.files, start=1):
            print("-" * 20)
            print(f"File {idx} of {len(args.files)}")
            test_single_file(args.base_url, path, max_wait=args.wait)

    print("\n🎯 API Testing Complete!")


if __name__ == "__main__":
    main()