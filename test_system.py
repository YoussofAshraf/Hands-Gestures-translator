"""
Test Script
Quick test to verify all components are working
"""

import sys
import cv2


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe
        print("✓ Mediapipe imported successfully")
    except ImportError as e:
        print(f"✗ Mediapipe import failed: {e}")
        return False
    
    try:
        import numpy
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import pyttsx3
        print("✓ pyttsx3 imported successfully")
    except ImportError as e:
        print(f"✗ pyttsx3 import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    return True


def test_camera():
    """Test if camera is accessible"""
    print("\nTesting camera...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Camera failed to open")
        print("  Possible solutions:")
        print("  - Check if camera is connected")
        print("  - Check camera permissions")
        print("  - Try different camera index (1, 2, etc.)")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("✗ Failed to read frame from camera")
        return False
    
    print(f"✓ Camera working - Frame shape: {frame.shape}")
    return True


def test_modules():
    """Test if custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from hand_detector import HandDetector
        print("✓ hand_detector module imported successfully")
    except ImportError as e:
        print(f"✗ hand_detector import failed: {e}")
        return False
    
    try:
        from gesture_classifier import GestureClassifier, SimpleGestureRecognizer
        print("✓ gesture_classifier module imported successfully")
    except ImportError as e:
        print(f"✗ gesture_classifier import failed: {e}")
        return False
    
    return True


def test_hand_detection():
    """Test hand detection functionality"""
    print("\nTesting hand detection...")
    
    try:
        from hand_detector import HandDetector
        
        detector = HandDetector()
        print("✓ HandDetector initialized successfully")
        
        # Test with a dummy image
        import numpy as np
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.find_hands(dummy_image, draw=False)
        print("✓ Hand detection method works")
        
        return True
    except Exception as e:
        print(f"✗ Hand detection test failed: {e}")
        return False


def test_gesture_classifier():
    """Test gesture classifier"""
    print("\nTesting gesture classifier...")
    
    try:
        from gesture_classifier import GestureClassifier, SimpleGestureRecognizer
        
        classifier = GestureClassifier()
        print("✓ GestureClassifier initialized successfully")
        
        simple = SimpleGestureRecognizer()
        print("✓ SimpleGestureRecognizer initialized successfully")
        
        # Test simple recognition
        fingers = [1, 1, 0, 0, 0]
        gesture = simple.recognize_finger_count(fingers)
        print(f"✓ Simple recognition works - Example: {fingers} -> {gesture}")
        
        return True
    except Exception as e:
        print(f"✗ Gesture classifier test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Hand Gesture Recognition System - Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test camera
    if not test_camera():
        all_passed = False
    
    # Test custom modules
    if not test_modules():
        all_passed = False
    
    # Test hand detection
    if not test_hand_detection():
        all_passed = False
    
    # Test gesture classifier
    if not test_gesture_classifier():
        all_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour system is ready to use!")
        print("\nNext steps:")
        print("1. Run the GUI application: python gui_app.py")
        print("2. Or run the CLI application: python main.py")
        print("3. To collect training data: python data_collector.py")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before running the application.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
