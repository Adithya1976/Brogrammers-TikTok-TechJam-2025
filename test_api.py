#!/usr/bin/env python3
"""
Test the TasksAI Privacy Pipeline API
"""
import requests
import time
import os

def test_server():
    """Test the server with sample images"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing TasksAI Privacy Pipeline API")
    print("=" * 40)
    
    # Test health check
    try:
        print("1. Testing health check...")
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Server healthy: {health['server']}")
            print(f"   GPU: {health['gpu_name']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("   ❌ Server not running. Start with: python start_server.py")
        return
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return
    
    # Test system info
    try:
        print("\n2. Testing system info...")
        response = requests.get(f"{base_url}/api/system-info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"   ✅ System info retrieved")
            print(f"   CPU Cores: {info['system']['cpu_cores']}")
            print(f"   RAM: {info['system']['ram_gb']} GB")
        else:
            print(f"   ❌ System info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ System info error: {e}")
    
    # Test file processing
    test_files = ['sample_safe.jpg', 'sample_private.jpg', 'sample_mixed.jpg']
    test_files = ['IMG_8412.JPG']
    
    for i, filename in enumerate(test_files, 3):
        if os.path.exists(filename):
            print(f"\n{i}. Testing {filename}...")
            try:
                with open(filename, 'rb') as f:
                    files = {'file': (filename, f, 'image/jpeg')}
                    response = requests.post(f"{base_url}/process", files=files, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    print(result)
                else:
                    print(f"   ❌ Processing failed: {response.status_code}")
                    print(f"   Error: {response.text}")
            except Exception as e:
                print(f"   ❌ Processing error: {e}")
        else:
            print(f"\n{i}. ⚠️ {filename} not found (run: python demo.py)")
    
    # Test gallery
    # try:
    #     print(f"\n{len(test_files) + 3}. Testing gallery...")
    #     response = requests.get(f"{base_url}/api/gallery", timeout=5)
    #     if response.status_code == 200:
    #         gallery = response.json()
    #         print(f"   ✅ Gallery retrieved: {gallery['total']} items")
    #     else:
    #         print(f"   ❌ Gallery failed: {response.status_code}")
    # except Exception as e:
    #     print(f"   ❌ Gallery error: {e}")
    
    # Test performance stats
    try:
        print(f"\n{len(test_files) + 4}. Testing performance stats...")
        response = requests.get(f"{base_url}/api/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✅ Stats retrieved")
            print(f"   Uptime: {stats.get('uptime_seconds', 0)}s")
            print(f"   Total Requests: {stats.get('total_requests', 0)}")
            if 'processing_time' in stats:
                print(f"   Avg Processing Time: {stats['processing_time'].get('avg', 'N/A')}s")
        else:
            print(f"   ❌ Stats failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Stats error: {e}")
    
    print("\n🎯 API Testing Complete!")
    print(f"📱 Mobile app can connect to: http://YOUR_LAPTOP_IP:8000/api/")

if __name__ == "__main__":
    test_server()