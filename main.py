import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import io
import time
from typing import List
from core.privacy_detector import PrivacyDetector
from core.video_processor import VideoProcessor
from core.adversarial_noise import AdversarialNoiseGenerator
from core.performance_monitor import performance_monitor
from models import ProcessingResult
from config import config

app = FastAPI(
    title="TasksAI Privacy Pipeline", 
    version="1.0.0",
    description="GPU-accelerated privacy detection for mobile apps"
)

# Add CORS middleware for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

privacy_detector = PrivacyDetector()
video_processor = VideoProcessor()
noise_generator = AdversarialNoiseGenerator()
storage = ProcessingResult()

# Mount static files
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TasksAI Privacy Pipeline</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f8f9fa; }
            .header { text-align: center; margin-bottom: 30px; }
            .upload-area { border: 2px dashed #007bff; padding: 40px; text-align: center; margin: 20px 0; background: white; border-radius: 10px; }
            .upload-area:hover { border-color: #0056b3; background: #f8f9ff; }
            .result { margin: 20px 0; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .privacy-flag { color: #dc3545; font-weight: bold; }
            .safe { color: #28a745; font-weight: bold; }
            .btn { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; }
            .btn-primary { background: #007bff; color: white; }
            .btn-secondary { background: #6c757d; color: white; }
            .btn:hover { opacity: 0.8; }
            .actions { margin-top: 15px; }
            .score-bar { width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin: 10px 0; }
            .score-fill { height: 100%; transition: width 0.3s ease; }
            .loading { display: none; text-align: center; margin: 20px 0; }
            .tabs { display: flex; margin-bottom: 20px; }
            .tab { padding: 10px 20px; background: #e9ecef; border: none; cursor: pointer; }
            .tab.active { background: #007bff; color: white; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔒 TasksAI Privacy Pipeline</h1>
            <p>Detect and protect privacy-sensitive information in images and videos</p>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('upload')">Upload & Analyze</button>
            <button class="tab" onclick="showTab('results')">Recent Results</button>
        </div>
        
        <div id="upload-tab" class="tab-content active">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*,video/*" style="display: none;" onchange="fileSelected()" />
                <h3>📁 Drop files here or click to upload</h3>
                <p>Supports images (JPG, PNG) and videos (MP4, AVI)</p>
                <div id="fileName"></div>
            </div>
            
            <div class="loading" id="loading">
                <p>🔄 Processing file...</p>
            </div>
            
            <div id="results"></div>
        </div>
        
        <div id="results-tab" class="tab-content">
            <h3>Recent Processing Results</h3>
            <div id="recentResults">Loading...</div>
        </div>
        
        <script>
            let currentFile = null;
            
            function showTab(tabName) {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                
                document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
                document.getElementById(`${tabName}-tab`).classList.add('active');
                
                if (tabName === 'results') {
                    loadRecentResults();
                }
            }
            
            function fileSelected() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (file) {
                    currentFile = file;
                    document.getElementById('fileName').innerHTML = `
                        <p><strong>Selected:</strong> ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)</p>
                        <button class="btn btn-primary" onclick="uploadFile()">🔍 Analyze File</button>
                    `;
                }
            }
            
            async function uploadFile() {
                if (!currentFile) return;
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').innerHTML = '';
                
                const formData = new FormData();
                formData.append('file', currentFile);
                
                try {
                    const response = await fetch('/process', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('results').innerHTML = '<div class="result"><p style="color: red;">Error processing file</p></div>';
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }
            
            function displayResults(result) {
                const scoreColor = result.privacy_score > 7 ? '#dc3545' : result.privacy_score > 4 ? '#ffc107' : '#28a745';
                
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `
                    <div class="result">
                        <h3>📊 Analysis Results</h3>
                        <p><strong>File:</strong> ${result.filename} (${result.file_type})</p>
                        
                        <div>
                            <strong>Privacy Risk Score: ${result.privacy_score}/10</strong>
                            <div class="score-bar">
                                <div class="score-fill" style="width: ${result.privacy_score * 10}%; background: ${scoreColor};"></div>
                            </div>
                        </div>
                        
                        <p><strong>Status:</strong> <span class="${result.is_safe ? 'safe' : 'privacy-flag'}">
                            ${result.is_safe ? '✅ Safe' : '⚠️ Privacy Concerns Detected'}
                        </span></p>
                        
                        <p><strong>Detected Entities:</strong> ${result.entities.length ? result.entities.join(', ') : 'None'}</p>
                        
                        ${result.ocr_text ? `<p><strong>Extracted Text:</strong><br><em>"${result.ocr_text.substring(0, 200)}${result.ocr_text.length > 200 ? '...' : ''}"</em></p>` : ''}
                        
                        ${result.frames_processed ? `<p><strong>Video Frames Processed:</strong> ${result.frames_processed}</p>` : ''}
                        
                        <div class="actions">
                            ${!result.is_safe && result.file_type === 'image' ? `
                                <button class="btn btn-secondary" onclick="addNoise('${result.filename}')">🔒 Add Privacy Noise</button>
                                <button class="btn btn-secondary" onclick="blurSensitive('${result.filename}')">🌫️ Blur Sensitive Areas</button>
                            ` : ''}
                        </div>
                    </div>
                `;
            }
            
            async function addNoise(filename) {
                if (!currentFile) return;
                
                const formData = new FormData();
                formData.append('file', currentFile);
                
                try {
                    const response = await fetch('/add-noise?intensity=0.15', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `noisy_${filename}`;
                        a.click();
                    }
                } catch (error) {
                    console.error('Error adding noise:', error);
                }
            }
            
            async function blurSensitive(filename) {
                if (!currentFile) return;
                
                const formData = new FormData();
                formData.append('file', currentFile);
                
                try {
                    const response = await fetch('/blur-sensitive', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `blurred_${filename}`;
                        a.click();
                    }
                } catch (error) {
                    console.error('Error blurring:', error);
                }
            }
            
            async function loadRecentResults() {
                try {
                    const response = await fetch('/results');
                    const results = await response.json();
                    
                    const resultsDiv = document.getElementById('recentResults');
                    if (results.length === 0) {
                        resultsDiv.innerHTML = '<p>No results yet. Upload some files to get started!</p>';
                        return;
                    }
                    
                    resultsDiv.innerHTML = results.map(r => `
                        <div class="result">
                            <h4>${r.result.filename}</h4>
                            <p><strong>Processed:</strong> ${new Date(r.timestamp).toLocaleString()}</p>
                            <p><strong>Privacy Score:</strong> ${r.result.privacy_score}/10</p>
                            <p><strong>Status:</strong> <span class="${r.result.is_safe ? 'safe' : 'privacy-flag'}">
                                ${r.result.is_safe ? 'Safe' : 'Privacy Concerns'}
                            </span></p>
                        </div>
                    `).join('');
                } catch (error) {
                    console.error('Error loading results:', error);
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    start_time = time.time()
    
    try:
        # Read file content
        content = await file.read()
        
        # Process based on file type
        if file.content_type.startswith('image/'):
            result = privacy_detector.process_image(content, file.filename)
        elif file.content_type.startswith('video/'):
            result = video_processor.process_video(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Record performance
        processing_time = time.time() - start_time
        performance_monitor.record_processing_time(processing_time)
        
        return result
            
    except Exception as e:
        performance_monitor.record_error()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/add-noise")
async def add_noise(file: UploadFile = File(...), intensity: float = 0.1):
    """Add adversarial noise to image to protect privacy"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only images supported")
    
    try:
        content = await file.read()
        noisy_image = noise_generator.add_gaussian_noise(content, intensity)
        
        return StreamingResponse(
            io.BytesIO(noisy_image),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename=noisy_{file.filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Noise generation error: {str(e)}")

@app.post("/blur-sensitive")
async def blur_sensitive(file: UploadFile = File(...)):
    """Blur sensitive regions in image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only images supported")
    
    try:
        content = await file.read()
        
        # First detect privacy issues to get regions
        analysis = privacy_detector.process_image(content)
        
        # For demo, blur entire image if privacy concerns found
        if not analysis["is_safe"]:
            # Mock regions - in real implementation, you'd detect text bounding boxes
            regions = [(50, 50, 200, 100)]  # x, y, width, height
            blurred_image = noise_generator.blur_sensitive_regions(content, regions)
        else:
            blurred_image = content
        
        return StreamingResponse(
            io.BytesIO(blurred_image),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename=blurred_{file.filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blurring error: {str(e)}")

@app.get("/results")
async def get_results():
    """Get recent processing results"""
    return storage.list_results()

@app.get("/results/{result_id}")
async def get_result(result_id: str):
    """Get specific processing result"""
    result = storage.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result

@app.get("/api/health")
async def health_check():
    """Health check endpoint for mobile app"""
    return {
        "status": "healthy",
        "server": "TasksAI Privacy Pipeline",
        "version": "1.0.0",
        **config.get_system_info()
    }

@app.get("/api/system-info")
async def get_system_info():
    """Get detailed system information for mobile app optimization"""
    return {
        "system": config.get_system_info(),
        "processing": config.get_processing_config(),
        "endpoints": {
            "process_single": "/process",
            "process_batch": "/api/process-batch",
            "add_noise": "/add-noise",
            "blur_sensitive": "/blur-sensitive",
            "gallery": "/api/gallery",
            "results": "/results"
        }
    }

@app.get("/api/gallery")
async def get_gallery():
    """Mobile-optimized gallery endpoint with thumbnails"""
    results = storage.list_results(limit=100)
    
    gallery_items = []
    for result in results:
        gallery_items.append({
            "id": result["id"],
            "filename": result["result"]["filename"],
            "privacy_score": result["result"]["privacy_score"],
            "is_safe": result["result"]["is_safe"],
            "entities": result["result"]["entities"],
            "file_type": result["result"]["file_type"],
            "timestamp": result["timestamp"],
            "processing_time": result["result"].get("processing_time", 0)
        })
    
    return {
        "items": gallery_items,
        "total": len(gallery_items)
    }

@app.post("/api/process-batch")
async def process_batch(files: List[UploadFile] = File(...)):
    """Process multiple files in batch for mobile app"""
    results = []
    
    for file in files:
        try:
            content = await file.read()
            
            if file.content_type.startswith('image/'):
                result = privacy_detector.process_image(content, file.filename)
                result["filename"] = file.filename
                result["file_type"] = "image"
            elif file.content_type.startswith('video/'):
                result = video_processor.process_video(content)
                result["filename"] = file.filename
                result["file_type"] = "video"
            else:
                continue
            
            # Save result
            result_id = storage.save_result(file.filename, result)
            result["result_id"] = result_id
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return {
        "processed": len(results),
        "results": results
    }

@app.get("/api/stats")
async def get_performance_stats():
    """Get server performance statistics for mobile app"""
    return performance_monitor.get_stats()

if __name__ == "__main__":
    # Optimized for RTX 3060 laptop server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,  # Single worker to maximize GPU usage
        loop="uvloop" if os.name != 'nt' else "asyncio" # Use uvloop on Linux/Mac
    )