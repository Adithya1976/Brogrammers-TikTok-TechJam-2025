#!/usr/bin/env python3
"""
Test with a real card image to verify OCR and privacy detection
"""
import os
from core.privacy_detector import PrivacyDetector

def test_real_card():
    """Test with a real card image"""
    print("🔍 Testing Real Card Privacy Detection")
    print("=" * 40)
    
    detector = PrivacyDetector()
    
    # Look for any uploaded image files
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("❌ No image files found. Please upload an image file.")
        return
    
    for filename in image_files:
        if 'photo_' in filename or 'card' in filename.lower() or 'id' in filename.lower():
            print(f"\n📸 Testing {filename}...")
            
            try:
                with open(filename, 'rb') as f:
                    image_bytes = f.read()
                
                result = detector.process_image(image_bytes, filename)
                
                print(f"   OCR Text: {result['ocr_text']}")
                print(f"   Privacy Score: {result['privacy_score']}/10")
                print(f"   Is Safe: {result['is_safe']}")
                print(f"   Entities Found: {result['entities']}")
                print(f"   Processing Time: {result['processing_time']}s")
                
                # Show detailed entities
                if result['detailed_entities']:
                    print("   Detailed Entities:")
                    for entity in result['detailed_entities']:
                        print(f"     - {entity['type']}: '{entity['text']}' (confidence: {entity['confidence']:.2f})")
                        
            except Exception as e:
                print(f"   ❌ Error processing {filename}: {e}")
    
    print("\n✅ Real Card Test Complete!")

if __name__ == "__main__":
    test_real_card()