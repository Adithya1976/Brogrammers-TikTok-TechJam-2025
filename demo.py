#!/usr/bin/env python3
"""
Demo script for TasksAI Privacy Pipeline
Creates sample images with text for testing
"""
from PIL import Image, ImageDraw, ImageFont
import io
import requests
import os

def create_sample_images():
    """Create sample images with different privacy levels"""
    
    # Sample 1: Safe image
    img1 = Image.new('RGB', (400, 200), color='white')
    draw1 = ImageDraw.Draw(img1)
    draw1.text((20, 50), "Welcome to our store!\nOpen 9AM - 5PM", fill='black')
    img1.save('sample_safe.jpg')
    
    # Sample 2: Privacy concerns
    img2 = Image.new('RGB', (400, 200), color='white')
    draw2 = ImageDraw.Draw(img2)
    draw2.text((20, 30), "Contact: john.doe@email.com\nPhone: (555) 123-4567\nSSN: 123-45-6789", fill='black')
    img2.save('sample_private.jpg')
    
    # Sample 3: Mixed content
    img3 = Image.new('RGB', (400, 200), color='white')
    draw3 = ImageDraw.Draw(img3)
    draw3.text((20, 30), "Restaurant Menu\nCall us: (555) 999-8888\nEmail: info@restaurant.com", fill='black')
    img3.save('sample_mixed.jpg')
    
    print("✅ Created sample images:")
    print("  - sample_safe.jpg (no privacy concerns)")
    print("  - sample_private.jpg (high privacy risk)")
    print("  - sample_mixed.jpg (moderate privacy risk)")

def test_api():
    """Test the API with sample images"""
    base_url = "http://localhost:8000"
    
    # Test if server is running
    try:
        response = requests.get(base_url)
        print("✅ Server is running")
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: python main.py")
        return
    
    # Test processing
    test_files = ['sample_safe.jpg', 'sample_private.jpg', 'sample_mixed.jpg']
    
    for filename in test_files:
        if os.path.exists(filename):
            print(f"\n🔍 Testing {filename}...")
            
            with open(filename, 'rb') as f:
                files = {'file': (filename, f, 'image/jpeg')}
                response = requests.post(f"{base_url}/process", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"  Privacy Score: {result['privacy_score']}/10")
                    print(f"  Safe: {result['is_safe']}")
                    print(f"  Entities: {result['entities']}")
                else:
                    print(f"  Error: {response.status_code}")

def main():
    print("🚀 TasksAI Privacy Pipeline Demo")
    print("=" * 40)
    
    # Create sample images
    create_sample_images()
    
    # Test API
    print("\n📡 Testing API...")
    test_api()
    
    print("\n🎯 Demo complete!")
    print("Open http://localhost:8000 to try the web interface")

if __name__ == "__main__":
    main()