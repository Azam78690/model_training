"""
Unified Data Collection Menu
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection.collect_sign_data import SignLanguageDataCollector
from data_collection.collect_braille_data import BrailleDataCollector

def main():
    print("📊 DATA COLLECTION TOOL")
    print("=" * 50)
    
    while True:
        print("\nChoose data collection type:")
        print("1. 🎬 Sign Language Data (Webcam + MediaPipe)")
        print("2. 🔤 Braille Data (Images + Manual Input)")
        print("0. 🚪 Exit")
        
        choice = input("\nEnter choice (0-2): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            print("\n🎬 Starting Sign Language Data Collection...")
            try:
                collector = SignLanguageDataCollector()
                collector.interactive_collection()
            except Exception as e:
                print(f"❌ Error: {e}")
        elif choice == '2':
            print("\n🔤 Starting Braille Data Collection...")
            try:
                collector = BrailleDataCollector()
                collector.interactive_collection()
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
