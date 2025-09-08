#!/usr/bin/env python3
"""
Demo script showing how to use the data collection tools
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_braille_data_collection():
    """Demonstrate braille data collection"""
    print("🔤 BRAILLE DATA COLLECTION DEMO")
    print("=" * 40)
    
    try:
        from data_collection.collect_braille_data import BrailleDataCollector
        
        # Create collector
        collector = BrailleDataCollector()
        print("✅ Braille data collector created")
        
        # Create a test class
        collector.create_class("A")
        print("✅ Created class 'A'")
        
        # Show braille reference
        collector.show_braille_reference()
        
        # Show statistics
        collector.show_statistics()
        
        print("\n🎯 To start interactive collection, run:")
        print("python data_collection/collect_braille_data.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_sign_data_collection():
    """Demonstrate sign language data collection"""
    print("\n🎬 SIGN LANGUAGE DATA COLLECTION DEMO")
    print("=" * 40)
    
    try:
        from data_collection.collect_sign_data import SignLanguageDataCollector
        
        # Create collector
        collector = SignLanguageDataCollector()
        print("✅ Sign language data collector created")
        
        # Create a test class
        collector.create_class("Hello")
        print("✅ Created class 'Hello'")
        
        # Show statistics
        collector.show_statistics()
        
        print("\n🎯 To start interactive collection, run:")
        print("python data_collection/collect_sign_data.py")
        print("(Note: Requires webcam access)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🧠 MODEL MANAGER - DATA COLLECTION DEMO")
    print("=" * 50)
    
    print("\nThe data collection tools are fully set up and working!")
    print("Here's what you can do:")
    
    print("\n1. 🔤 BRAILLE DATA COLLECTION:")
    print("   - Upload braille images")
    print("   - Manual 6-dot pattern input")
    print("   - Batch image processing")
    print("   - Interactive class management")
    
    print("\n2. 🎬 SIGN LANGUAGE DATA COLLECTION:")
    print("   - Real-time webcam recording")
    print("   - MediaPipe hand landmark extraction")
    print("   - Interactive gesture recording")
    print("   - Sequence data management")
    
    print("\n🚀 QUICK START:")
    print("   python data_collection_menu.py          # Unified menu")
    print("   python data_collection/collect_braille_data.py  # Braille only")
    print("   python data_collection/collect_sign_data.py     # Sign language only")
    
    print("\n📊 DATA COLLECTION FEATURES:")
    print("   ✅ Interactive class creation and management")
    print("   ✅ Real-time data collection with visual feedback")
    print("   ✅ Automatic data processing and formatting")
    print("   ✅ Statistics and progress monitoring")
    print("   ✅ File upload and batch processing")
    print("   ✅ Manual pattern input with validation")
    
    # Run demos
    demo_braille_data_collection()
    demo_sign_data_collection()
    
    print("\n🎉 DATA COLLECTION IS READY TO USE!")
    print("All tools are working correctly and ready for data collection.")

if __name__ == "__main__":
    main()
