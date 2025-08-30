#!/usr/bin/env python3
"""
Test the TasksAI Privacy Pipeline API (matches 202 + job polling behavior).

Usage:
  python test_api.py --base http://localhost:8000 IMG_8414.JPG
  python test_api.py --base http://localhost:8000 --batch sample1.jpg sample2.jpg

Notes:
- /process returns 202 + {"job_id", "status_url"} and we poll /jobs/{job_id}
- /api/health is used to verify server status
- Batch mode hits /api/process-batch and polls the same /jobs/{job_id}
"""
import argparse
import json
import mimetypes
import os
import time
from typing import Dict, Optional

import requests


def join_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    path = path if path.startswith("/") else f"/{path}"
    return f"{base}{path}"


def get_health(base_url: str, timeout: float = 5.0) -> Optional[Dict]:
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
    mt, _ = mimetypes.guess_type(path)
    # default to image/jpeg if unknown; FastAPI will still use filename
    return mt or "application/octet-stream"


def poll_job(base_url: str, job_id: str, max_wait: float = 180.0, tick: float = 1.0) -> Dict:
    """Poll /jobs/{job_id} until status in {'done','error'} or timeout."""
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


def test_single_file(base_url: str, path: str, timeout: float = 10.0) -> None:
    if not os.path.exists(path):
        print(f"   ⚠️  {path} not found")
        return

    print(f"\n▶️  Testing single file: {path}")
    ctype = detect_mime(path)
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, ctype)}
        r = requests.post(join_url(base_url, "/process"), files=files, timeout=timeout)

    # Expect 202 with job id
    if r.status_code == 202:
        try:
            data = r.json()
        except Exception:
            print("   ❌ Invalid JSON in 202 response")
            print(f"   Body: {r.text}")
            return

        job_id = data.get("job_id")
        status_url = data.get("status_url")
        loc_header = r.headers.get("Location")
        print(f"   ✅ Enqueued (job_id={job_id})")
        if status_url:
            print(f"   📍 Status URL (body): {status_url}")
        if loc_header:
            print(f"   📍 Status URL (Location header): {loc_header}")

        if not job_id:
            print("   ❌ Missing job_id in response; cannot poll.")
            return

        # Poll
        result = poll_job(base_url, job_id)
        print("   🎯 Final job payload:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # Back-compat: some servers might return 200 with immediate result
    if r.status_code == 200:
        print("   ⚠️ Received 200 OK (immediate result). Body:")
        try:
            print(json.dumps(r.json(), indent=2, ensure_ascii=False))
        except Exception:
            print(r.text)
        return

    print(f"   ❌ Upload failed: HTTP {r.status_code}")
    try:
        print(f"   Body: {r.text}")
    except Exception:
        pass


def test_batch(base_url: str, paths: list[str], timeout: float = 30.0) -> None:
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
            print(f"   ❌ Batch enqueue failed: HTTP {r.status_code}")
            print(f"   Body: {r.text}")
            return

        data = r.json()
        job_id = data.get("job_id")
        print(f"   ✅ Batch enqueued (job_id={job_id})")
        if not job_id:
            print("   ❌ Missing job_id in batch response; cannot poll.")
            return

        result = poll_job(base_url, job_id, max_wait=600.0)  # batches can take longer
        print("   🎯 Final batch job payload:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    finally:
        for f in to_close:
            try:
                f.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Test TasksAI Privacy Pipeline API")
    parser.add_argument(
        "--base", dest="base_url", default="http://localhost:8000", help="Base URL of the API"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Send all provided files as a single batch job"
    )
    parser.add_argument(
        "--wait", type=float, default=180.0, help="Max seconds to wait for single job"
    )
    parser.add_argument("files", nargs="*", help="Image/video files to upload")
    args = parser.parse_args()

    print("🧪 Testing TasksAI Privacy Pipeline API")
    print("=" * 40)

    # 1) Health
    print("1. Testing health check...")
    health = get_health(args.base_url)
    if not health:
        return
    try:
        print(f"   ✅ Server: {health.get('server', 'unknown')} (v{health.get('version','?')})")
        gpu_name = health.get("gpu_name") or health.get("gpu") or health.get("gpu_model")
        if gpu_name:
            print(f"   GPU: {gpu_name}")
    except Exception:
        pass

    # 2) Upload(s)
    if args.batch and len(args.files) >= 1:
        test_batch(args.base_url, args.files)
    else:
        targets = args.files or ["IMG.png"]  # default demo file
        for idx, path in enumerate(targets, start=2):
            print(f"\n{idx}. Upload test")
            test_single_file(args.base_url, path)
            # respect user-configured wait per job (poller uses its own default unless you tweak it)


    print("\n🎯 API Testing Complete!")
    print("📱 Mobile app can connect to: http://YOUR_LAPTOP_IP:8000/api/")


if __name__ == "__main__":
    main()
