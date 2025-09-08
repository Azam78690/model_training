#!/usr/bin/env python3
"""
Test script to verify Model Manager installation
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print("✅ TorchVision imported successfully")
        print(f"   Version: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        print(f"   Version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
        print(f"   Version: {mp.__version__}")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    
    try:
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 imported successfully")
    except ImportError as e:
        print(f"❌ PyQt6 import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        print(f"   Version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import pyttsx3
        print("✅ pyttsx3 imported successfully")
    except ImportError as e:
        print(f"❌ pyttsx3 import failed: {e}")
        return False
    
    try:
        from gtts import gTTS
        print("✅ gTTS imported successfully")
    except ImportError as e:
        print(f"❌ gTTS import failed: {e}")
        return False
    
    return True

def test_project_structure():
    """Test project structure"""
    print("\n📁 Testing project structure...")
    
    required_dirs = [
        "datasets",
        "models", 
        "gui",
        "training",
        "prediction",
        "data_collection",
        "versioning",
        "utils"
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory {dir_name} exists")
        else:
            print(f"❌ Directory {dir_name} missing")
            return False
    
    required_files = [
        "requirements.txt",
        "README.md",
        "interactive_manager.py",
        "data_collection_menu.py",
        "training/sign_model.py",
        "training/braille_model.py",
        "prediction/predict_sign.py",
        "prediction/predict_braille.py",
        "utils/mediapipe_utils.py",
        "utils/braille_mapping.py",
        "utils/tts_utils.py",
        "versioning/registry_manager.py",
        "versioning/model_registry.json"
    ]
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ File {file_name} exists")
        else:
            print(f"❌ File {file_name} missing")
            return False
    
    return True

def test_model_imports():
    """Test model imports"""
    print("\n🧠 Testing model imports...")
    
    try:
        from training.sign_model import SignGRUClassifier
        print("✅ SignGRUClassifier imported successfully")
        
        # Test model creation
        model = SignGRUClassifier()
        print("✅ SignGRUClassifier created successfully")
        print(f"   Model info: {model.get_model_info()}")
    except Exception as e:
        print(f"❌ SignGRUClassifier test failed: {e}")
        return False
    
    try:
        from training.braille_model import BrailleCNNClassifier
        print("✅ BrailleCNNClassifier imported successfully")
        
        # Test model creation
        model = BrailleCNNClassifier()
        print("✅ BrailleCNNClassifier created successfully")
        print(f"   Model info: {model.get_model_info()}")
    except Exception as e:
        print(f"❌ BrailleCNNClassifier test failed: {e}")
        return False
    
    return True

def test_utilities():
    """Test utility modules"""
    print("\n🔧 Testing utility modules...")
    
    try:
        from utils.mediapipe_utils import HandLandmarkExtractor
        print("✅ HandLandmarkExtractor imported successfully")
        
        # Test extractor creation
        extractor = HandLandmarkExtractor()
        print("✅ HandLandmarkExtractor created successfully")
    except Exception as e:
        print(f"❌ HandLandmarkExtractor test failed: {e}")
        return False
    
    try:
        from utils.braille_mapping import braille_to_char, char_to_braille
        print("✅ Braille mapping functions imported successfully")
        
        # Test braille mapping
        char = braille_to_char([1,0,0,0,0,0])
        print(f"✅ Braille mapping test: {char}")
    except Exception as e:
        print(f"❌ Braille mapping test failed: {e}")
        return False
    
    try:
        from utils.tts_utils import TTS_System
        print("✅ TTS_System imported successfully")
        
        # Test TTS creation
        tts = TTS_System()
        print("✅ TTS_System created successfully")
    except Exception as e:
        print(f"❌ TTS_System test failed: {e}")
        return False
    
    return True

def test_registry():
    """Test model registry"""
    print("\n📋 Testing model registry...")
    
    try:
        from versioning.registry_manager import ModelRegistry
        print("✅ ModelRegistry imported successfully")
        
        # Test registry creation
        registry = ModelRegistry()
        print("✅ ModelRegistry created successfully")
        
        # Test registry operations
        stats = registry.get_model_statistics()
        print(f"✅ Registry statistics: {stats}")
    except Exception as e:
        print(f"❌ ModelRegistry test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧠 MODEL MANAGER - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Project Structure", test_project_structure),
        ("Model Imports", test_model_imports),
        ("Utility Modules", test_utilities),
        ("Model Registry", test_registry)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Model Manager is ready to use.")
        print("\n🚀 Quick Start:")
        print("   python interactive_manager.py          # CLI mode")
        print("   python interactive_manager.py --gui    # GUI mode")
        print("   python data_collection_menu.py         # Data collection")
    else:
        print("❌ Some tests failed. Please check the installation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
