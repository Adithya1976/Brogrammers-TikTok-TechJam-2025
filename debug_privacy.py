#!/usr/bin/env python3
"""
Debug privacy detection issues
"""
from core.privacy_detector import PrivacyDetector
import traceback

def test_privacy_detector():
    print("🔍 Debugging Privacy Detector")
    print("=" * 30)
    
    try:
        detector = PrivacyDetector()
        print("✅ PrivacyDetector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize PrivacyDetector: {e}")
        traceback.print_exc()
        return
    
    # Test text analysis
    test_text = "Contact john.doe@email.com or call (555) 123-4567"
    print(f"\n📝 Testing text analysis with: '{test_text}'")
    
    try:
        result = detector.analyze_privacy(test_text)
        print(f"✅ Analysis result: {result}")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        traceback.print_exc()
        return
    
    # Test with empty text
    print(f"\n📝 Testing with empty text...")
    try:
        result = detector.analyze_privacy("")
        print(f"✅ Empty text result: {result}")
    except Exception as e:
        print(f"❌ Empty text failed: {e}")
        traceback.print_exc()
    
    # Test image processing with different files
    test_files = ["sample_safe.jpg", "sample_private.jpg", "sample_mixed.jpg"]
    
    for filename in test_files:
        print(f"\n🖼️ Testing {filename}...")
        try:
            with open(filename, "rb") as f:
                image_bytes = f.read()
            
            result = detector.process_image(image_bytes, filename)
            print(f"✅ Result: Privacy Score {result['privacy_score']}/10, Safe: {result['is_safe']}")
            print(f"   OCR Text: '{result['ocr_text']}'")
            print(f"   Entities: {result['entities']}")
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    test_privacy_detector()