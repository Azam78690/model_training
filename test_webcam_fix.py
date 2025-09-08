#!/usr/bin/env python3
"""
Test script to verify the webcam fix works
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_webcam_fix():
    """Test the fixed webcam functionality"""
    print("🎥 Testing Webcam Fix")
    print("=" * 30)
    
    try:
        from data_collection.collect_sign_data import SignLanguageDataCollector
        
        # Create collector
        collector = SignLanguageDataCollector()
        print("✅ Sign language data collector created")
        
        # Test camera initialization
        collector.start_camera()
        print("✅ Camera initialized")
        
        # Test camera reading
        ret, frame = collector.cap.read()
        if ret:
            print("✅ Camera reading works")
        else:
            print("❌ Camera reading failed")
            return False
        
        # Test camera stopping
        collector.stop_camera()
        print("✅ Camera stopped successfully")
        
        # Test multiple start/stop cycles
        for i in range(3):
            print(f"Testing cycle {i+1}/3...")
            collector.start_camera()
            ret, frame = collector.cap.read()
            if ret:
                print(f"  ✅ Cycle {i+1} - Camera working")
            else:
                print(f"  ❌ Cycle {i+1} - Camera failed")
                return False
            collector.stop_camera()
            print(f"  ✅ Cycle {i+1} - Camera stopped")
        
        print("\n🎉 Webcam fix test PASSED!")
        print("The camera should now work properly for multiple recordings.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🧪 WEBCAM FIX TEST")
    print("=" * 40)
    
    print("This test verifies that the webcam fix resolves the freezing issue.")
    print("The fix includes:")
    print("- Proper camera release and reinitialization")
    print("- Better error handling")
    print("- Window cleanup after each recording")
    print("- Multiple recording capability")
    
    success = test_webcam_fix()
    
    if success:
        print("\n✅ FIX VERIFIED - Webcam should work properly now!")
        print("\n🚀 Try recording again:")
        print("   python data_collection/collect_sign_data.py")
    else:
        print("\n❌ Fix verification failed. Please check the error messages.")

if __name__ == "__main__":
    main()
