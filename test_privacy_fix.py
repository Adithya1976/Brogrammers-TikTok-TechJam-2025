#!/usr/bin/env python3
"""
Test the improved privacy detector to verify it properly detects privacy issues
"""
import os
from core.privacy_detector import PrivacyDetector

def test_privacy_detection():
    """Test privacy detection with sample images"""
    print("🔍 Testing Privacy Detection Fix")
    print("=" * 40)
    
    detector = PrivacyDetector()
    
    # Test with sample images
    test_files = ['sample_safe.jpg', 'sample_private.jpg', 'sample_mixed.jpg']
    
    for filename in test_files:
        if os.path.exists(filename):
            print(f"\n📸 Testing {filename}...")
            
            with open(filename, 'rb') as f:
                image_bytes = f.read()
            
            result = detector.process_image(image_bytes, filename)
            
            print(f"   OCR Text: {result['ocr_text'][:100]}...")
            print(f"   Privacy Score: {result['privacy_score']}/10")
            print(f"   Is Safe: {result['is_safe']}")
            print(f"   Entities Found: {result['entities']}")
            print(f"   Processing Time: {result['processing_time']}s")
            
            # Show detailed entities for debugging
            if result['detailed_entities']:
                print("   Detailed Entities:")
                for entity in result['detailed_entities']:
                    print(f"     - {entity['type']}: '{entity['text']}' (confidence: {entity['confidence']:.2f})")
        else:
            print(f"\n⚠️ {filename} not found")
    
    print("\n✅ Privacy Detection Test Complete!")

if __name__ == "__main__":
    test_privacy_detection()