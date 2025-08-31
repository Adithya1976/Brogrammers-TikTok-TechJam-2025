import asyncio
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import io
import time
import json
import uuid
import tempfile
from datetime import datetime, timezone
from typing import List, Dict, Any

from core.image_processor import ImageProcessor
from core.object_detector import GroundingDINO_SAMModel
from core.privacy_detector import PrivacyDetector
from core.video_processor import VideoProcessor
from core.adversarial_noise import AdversarialNoiseGenerator
from core.performance_monitor import performance_monitor
from models import ProcessingResult
from config import config

# -------------------------
# App & Middleware
# -------------------------
app = FastAPI(
    title="TasksAI Privacy Pipeline",
    version="1.0.0",
    description="GPU-accelerated privacy detection for mobile apps"
)

# -------------------------
# Globals
# -------------------------
grounding_sam = GroundingDINO_SAMModel()
privacy_detector = PrivacyDetector()
image_processor = ImageProcessor(grounding_sam=grounding_sam, privacy_detector=privacy_detector)
video_processor = VideoProcessor(grounding_sam=grounding_sam)
storage = ProcessingResult()

JOBS_DIR = "./jobs"
UPLOADS_DIR = "./uploads"

def _ensure_dirs():
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

_ensure_dirs()

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _job_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"{job_id}.json")

def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".job_", dir=JOBS_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

def _init_job_record(job_id: str, job_type: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    rec = {
        "job_id": job_id,
        "status": "pending",            # pending | running | done | error
        "type": job_type,               # "image" | "video" | "batch"
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
        "result": None,
        "error": None,
    }
    if extra:
        rec.update(extra)
    _atomic_write_json(_job_path(job_id), rec)
    return rec

def _update_job(job_id: str, **fields):
    path = _job_path(job_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"job {job_id} not found")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.update(fields)
    data["updated_at"] = _utcnow_iso()
    _atomic_write_json(path, data)
    return data

def _save_upload_to_disk(job_id: str, file: UploadFile) -> str:
    # store by job to avoid name clashes
    basename = f"{job_id}__{os.path.basename(file.filename)}"
    dest = os.path.join(UPLOADS_DIR, basename)
    # UploadFile exposes .read() as async; caller reads and provides bytes when needed.
    return dest

async def _write_bytes(path: str, content: bytes):
    # write file atomically
    tmp_fd, tmp_path = tempfile.mkstemp(dir=UPLOADS_DIR, prefix=".up_")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

# -------------------------
# Background workers
# -------------------------
async def _process_single_file_job(job_id: str, stored_path: str, content_type: str, original_name: str):
    t0 = time.time()
    try:
        _update_job(job_id, status="running")

        if content_type.startswith("image/"):
            with open(stored_path, "rb") as f:
                content = f.read()
            result = await asyncio.to_thread(image_processor.process_image, content)
        elif content_type.startswith("video/"):
            result = await asyncio.to_thread(video_processor.process_video, stored_path)
        else:
            raise ValueError(f"Unsupported file type: {content_type}")

        processing_time = time.time() - t0
        performance_monitor.record_processing_time(processing_time)

        _update_job(job_id, status="done", result=result, error=None)
    except Exception as e:
        performance_monitor.record_error()
        traceback.print_exc()
        _update_job(job_id, status="error", error=str(e), result=None)

async def _process_batch_job(job_id: str, entries: List[Dict[str, str]]):
    """
    entries: list of {"path": ..., "content_type": ..., "filename": ...}
    """
    t0 = time.time()
    aggregated: List[Dict[str, Any]] = []
    try:
        # Match single-file job: mark as running (and include initial progress)
        _update_job(job_id, status="running", result={"processed": 0, "results": []})

        for idx, ent in enumerate(entries, start=1):
            per_file: Dict[str, Any] = {
                "filename": ent["filename"],
                "file_type": "image" if ent["content_type"].startswith("image/") else "video"
            }

            try:
                if ent["content_type"].startswith("image/"):
                    # Images: read bytes and offload to thread (same as single-file)
                    with open(ent["path"], "rb") as f:
                        content = f.read()
                    res = await asyncio.to_thread(image_processor.process_image, content)

                elif ent["content_type"].startswith("video/"):
                    # Videos: pass the PATH to the processor (same as single-file)
                    res = await asyncio.to_thread(video_processor.process_video, ent["path"])

                else:
                    raise ValueError(f"Unsupported file type: {ent['content_type']}")

                per_file.update({"success": True, "result": res})

            except Exception as e:
                performance_monitor.record_error()
                traceback.print_exc()
                per_file.update({"success": False, "error": str(e)})

            aggregated.append(per_file)

            # Stream progress after each file
            _update_job(job_id, result={"processed": idx, "results": aggregated})

        processing_time = time.time() - t0
        performance_monitor.record_processing_time(processing_time)

        _update_job(
            job_id,
            status="done",
            error=None,
            result={"processed": len(aggregated), "results": aggregated},
        )

    except Exception as e:
        performance_monitor.record_error()
        traceback.print_exc()
        _update_job(job_id, status="error", error=str(e))
# -------------------------
# Endpoints
# -------------------------
@app.post("/process", status_code=202)
async def process_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        job_id = str(uuid.uuid4())
        stored_path = _save_upload_to_disk(job_id, file)
        content = await file.read()
        await _write_bytes(stored_path, content)

        _init_job_record(job_id, job_type=("image" if file.content_type.startswith("image/") else "video"), extra={
            "original_filename": file.filename,
            "content_type": file.content_type,
            "input_path": stored_path,
        })

        # fire: schedule background processing
        asyncio.create_task(_process_single_file_job(job_id, stored_path, file.content_type, file.filename))

        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "pending",
                "status_url": f"/jobs/{job_id}"
            },
            headers={"Location": f"/jobs/{job_id}"}
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {str(e)}")

@app.post("/api/process-batch", status_code=202)
async def process_batch(bg: BackgroundTasks, files: List[UploadFile] = File(...)):
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        job_id = str(uuid.uuid4())
        entries = []

        for f in files:
            content = await f.read()
            path = _save_upload_to_disk(job_id, f)
            await _write_bytes(path, content)
            entries.append({"path": path, "content_type": f.content_type, "filename": f.filename})

        _init_job_record(job_id, job_type="batch", extra={
            "files": [{"filename": e["filename"], "content_type": e["content_type"], "path": e["path"]} for e in entries]
        })

        # fire
        asyncio.create_task(_process_batch_job(job_id, entries))

        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "pending",
                "status_url": f"/jobs/{job_id}"
            },
            headers={"Location": f"/jobs/{job_id}"}
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to enqueue batch: {str(e)}")

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    path = _job_path(job_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Job not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "server": "TasksAI Privacy Pipeline",
        "version": "1.0.0",
        **config.get_system_info()
    }

# (Optional) keep your static files, UI, etc. here if needed
# app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0", # Prefer localhost over 0.0.0.0
        port=8000,
        workers=1,  # single worker keeps GPU libs simple; scale with a real queue later
    )
