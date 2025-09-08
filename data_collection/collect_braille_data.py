"""
Braille Data Collection Tool
Collects braille data via image upload or manual 6-dot pattern input.
"""

import os
import sys
import shutil
from datetime import datetime
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.braille_mapping import braille_to_char, char_to_braille
except ImportError:
    print("❌ Could not import braille mapping utils. Make sure you're running from the project root.")
    sys.exit(1)

class BrailleDataCollector:
    def __init__(self, output_dir="datasets/braille"):
        self.output_dir = output_dir
        self.current_class = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def list_classes(self):
        """List existing classes"""
        if not os.path.exists(self.output_dir):
            return []
        
        classes = [d for d in os.listdir(self.output_dir) 
                  if os.path.isdir(os.path.join(self.output_dir, d))]
        return sorted(classes)
    
    def create_class(self, class_name):
        """Create a new class directory"""
        class_dir = os.path.join(self.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        self.current_class = class_name
        print(f"✅ Created class: {class_name}")
        return class_dir
    
    def count_samples(self, class_name):
        """Count samples in a class"""
        class_dir = os.path.join(self.output_dir, class_name)
        if not os.path.exists(class_dir):
            return 0
        
        files = [f for f in os.listdir(class_dir) 
                if f.endswith(('.png', '.jpg', '.jpeg', '.npy'))]
        return len(files)
    
    def upload_image(self, image_path, class_name=None):
        """Upload and process a braille image"""
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return False
        
        if not class_name:
            class_name = self.current_class
        
        if not class_name:
            print("❌ No class selected. Please create or select a class first.")
            return False
        
        # Create class directory if it doesn't exist
        class_dir = os.path.join(self.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{class_name}_{timestamp}.jpg"
        dest_path = os.path.join(class_dir, filename)
        
        try:
            # Load and process image
            image = Image.open(image_path)
            
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize to standard size (64x64)
            image = image.resize((64, 64), Image.Resampling.LANCZOS)
            
            # Save processed image
            image.save(dest_path, 'JPEG', quality=95)
            
            print(f"✅ Image uploaded: {filename}")
            print(f"   Class: {class_name}")
            print(f"   Size: {image.size}")
            print(f"   Samples in class: {self.count_samples(class_name)}")
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    def manual_input(self, class_name=None):
        """Manual braille pattern input"""
        if not class_name:
            class_name = self.current_class
        
        if not class_name:
            print("❌ No class selected. Please create or select a class first.")
            return False
        
        print(f"\n🔤 Manual Braille Input for class: {class_name}")
        print("Enter 6-dot braille pattern (1 for raised dot, 0 for flat dot)")
        print("Example: 1,0,0,0,0,0 for 'A'")
        print("Press Enter with empty input to cancel")
        
        while True:
            pattern_input = input("\nEnter pattern (6 numbers, comma-separated): ").strip()
            
            if not pattern_input:
                print("❌ Input cancelled")
                return False
            
            try:
                # Parse pattern
                pattern = [int(x.strip()) for x in pattern_input.split(',')]
                
                if len(pattern) != 6:
                    print("❌ Pattern must have exactly 6 dots")
                    continue
                
                if not all(x in [0, 1] for x in pattern):
                    print("❌ Pattern must contain only 0s and 1s")
                    continue
                
                # Get character
                char = braille_to_char(pattern)
                print(f"✅ Pattern {pattern} represents: '{char}'")
                
                # Confirm
                confirm = input(f"Save this pattern for class '{class_name}'? (y/n): ").strip().lower()
                if confirm == 'y':
                    return self.save_pattern(pattern, class_name, char)
                else:
                    print("❌ Pattern not saved")
                    return False
                    
            except ValueError:
                print("❌ Invalid input. Please enter 6 numbers separated by commas.")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def save_pattern(self, pattern, class_name, char):
        """Save braille pattern as data"""
        # Create class directory if it doesn't exist
        class_dir = os.path.join(self.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{class_name}_{timestamp}.npy"
        filepath = os.path.join(class_dir, filename)
        
        try:
            # Save pattern as numpy array
            pattern_array = np.array(pattern, dtype=np.int8)
            np.save(filepath, pattern_array)
            
            print(f"✅ Pattern saved: {filename}")
            print(f"   Pattern: {pattern}")
            print(f"   Character: '{char}'")
            print(f"   Samples in class: {self.count_samples(class_name)}")
            return True
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return False
    
    def batch_upload_images(self, image_dir, class_name=None):
        """Upload multiple images from a directory"""
        if not class_name:
            class_name = self.current_class
        
        if not class_name:
            print("❌ No class selected. Please create or select a class first.")
            return False
        
        if not os.path.exists(image_dir):
            print(f"❌ Directory not found: {image_dir}")
            return False
        
        # Find image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return False
        
        print(f"📁 Found {len(image_files)} images in {image_dir}")
        print(f"📋 Will upload to class: {class_name}")
        
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Upload cancelled")
            return False
        
        # Upload images
        success_count = 0
        for filename in image_files:
            image_path = os.path.join(image_dir, filename)
            if self.upload_image(image_path, class_name):
                success_count += 1
        
        print(f"\n✅ Upload completed: {success_count}/{len(image_files)} images uploaded")
        return success_count > 0
    
    def show_statistics(self):
        """Show data collection statistics"""
        classes = self.list_classes()
        
        print("\n📊 BRAILLE DATA COLLECTION STATISTICS")
        print("=" * 50)
        
        if not classes:
            print("No classes found. Start collecting data!")
            return
        
        total_samples = 0
        for class_name in classes:
            count = self.count_samples(class_name)
            total_samples += count
            print(f"{class_name}: {count} samples")
        
        print(f"\nTotal samples: {total_samples}")
        print(f"Total classes: {len(classes)}")
    
    def show_braille_reference(self):
        """Show braille pattern reference"""
        print("\n🔤 BRAILLE PATTERN REFERENCE")
        print("=" * 50)
        print("Standard 6-dot braille patterns:")
        print()
        
        # Show some common patterns
        common_patterns = [
            ([1,0,0,0,0,0], 'A'),
            ([1,1,0,0,0,0], 'B'),
            ([1,0,0,1,0,0], 'C'),
            ([1,0,0,1,1,0], 'D'),
            ([1,0,0,0,1,0], 'E'),
            ([1,1,0,1,0,0], 'F'),
            ([1,1,0,1,1,0], 'G'),
            ([1,1,0,0,1,0], 'H'),
            ([0,1,0,1,0,0], 'I'),
            ([0,1,0,1,1,0], 'J'),
        ]
        
        for pattern, char in common_patterns:
            pattern_str = ','.join(map(str, pattern))
            print(f"  {pattern_str} = '{char}'")
        
        print("\nPattern format: [top_left, top_right, middle_left, middle_right, bottom_left, bottom_right]")
        print("1 = raised dot, 0 = flat dot")
    
    def interactive_collection(self):
        """Interactive data collection session"""
        print("🔤 BRAILLE DATA COLLECTION")
        print("=" * 50)
        
        while True:
            print("\n📋 OPTIONS:")
            print("1. Create new class")
            print("2. Select existing class")
            print("3. Upload single image")
            print("4. Upload multiple images")
            print("5. Manual pattern input")
            print("6. Show statistics")
            print("7. List classes")
            print("8. Show braille reference")
            print("0. Exit")
            
            choice = input("\nEnter choice (0-8): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                class_name = input("Enter class name (e.g., 'A', 'B', 'Hello'): ").strip()
                if class_name:
                    self.create_class(class_name)
            elif choice == '2':
                classes = self.list_classes()
                if classes:
                    print("\nAvailable classes:")
                    for i, cls in enumerate(classes, 1):
                        count = self.count_samples(cls)
                        print(f"{i}. {cls} ({count} samples)")
                    
                    try:
                        idx = int(input("Select class number: ")) - 1
                        if 0 <= idx < len(classes):
                            self.current_class = classes[idx]
                            print(f"✅ Selected class: {self.current_class}")
                    except ValueError:
                        print("❌ Invalid selection")
                else:
                    print("❌ No classes found. Create a class first.")
            elif choice == '3':
                if self.current_class:
                    image_path = input("Enter image file path: ").strip()
                    if image_path:
                        self.upload_image(image_path)
                else:
                    print("❌ No class selected. Please create or select a class first.")
            elif choice == '4':
                if self.current_class:
                    image_dir = input("Enter directory path containing images: ").strip()
                    if image_dir:
                        self.batch_upload_images(image_dir)
                else:
                    print("❌ No class selected. Please create or select a class first.")
            elif choice == '5':
                self.manual_input()
            elif choice == '6':
                self.show_statistics()
            elif choice == '7':
                classes = self.list_classes()
                if classes:
                    print("\nAvailable classes:")
                    for cls in classes:
                        count = self.count_samples(cls)
                        print(f"  {cls}: {count} samples")
                else:
                    print("No classes found.")
            elif choice == '8':
                self.show_braille_reference()
            else:
                print("❌ Invalid choice")

def main():
    collector = BrailleDataCollector()
    collector.interactive_collection()

if __name__ == "__main__":
    main()
