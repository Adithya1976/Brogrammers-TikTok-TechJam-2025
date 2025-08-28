#!/usr/bin/env python3
"""
Optimized server startup for RTX 3060 laptop
"""
import os
import sys
import subprocess
import socket
import time
from config import config

def check_port(port: int) -> bool:
    """Check if port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def get_local_ip():
    """Get local IP address for mobile app connection"""
    try:
        # Connect to a remote address to get local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "localhost"

def start_server():
    """Start the optimized server"""
    print("🚀 Starting TasksAI Privacy Pipeline Server")
    print("=" * 50)
    
    # System info
    system_info = config.get_system_info()
    print(f"GPU: {system_info['gpu_name']}")
    print(f"GPU Memory: {system_info['gpu_memory_gb']} GB")
    print(f"CPU Cores: {system_info['cpu_cores']}")
    print(f"RAM: {system_info['ram_gb']} GB")
    print()
    
    # Check port availability
    port = 8000
    if not check_port(port):
        print(f"❌ Port {port} is already in use")
        return
    
    # Get network info
    local_ip = get_local_ip()
    print(f"🌐 Server will be available at:")
    print(f"   Local: http://localhost:{port}")
    print(f"   Network: http://{local_ip}:{port}")
    print(f"   Mobile API: http://{local_ip}:{port}/api/")
    print()
    
    # Server configuration
    processing_config = config.get_processing_config()
    print("⚙️ Server Configuration:")
    print(f"   GPU Acceleration: {processing_config['use_gpu_acceleration']}")
    print(f"   Batch Size: {processing_config['batch_size']}")
    print(f"   Max Image Size: {processing_config['max_image_size']}")
    print(f"   Concurrent Requests: {processing_config['concurrent_requests']}")
    print()
    
    print("📱 Mobile App Endpoints:")
    print(f"   Health Check: http://{local_ip}:{port}/api/health")
    print(f"   System Info: http://{local_ip}:{port}/api/system-info")
    print(f"   Process File: http://{local_ip}:{port}/process")
    print(f"   Gallery: http://{local_ip}:{port}/api/gallery")
    print(f"   Performance: http://{local_ip}:{port}/api/stats")
    print()
    
    print("🔥 Starting server with RTX 3060 optimization...")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # Start the server
    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            workers=1,  # Single worker for GPU optimization
            reload=False,  # Disable reload for production
            access_log=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    start_server()