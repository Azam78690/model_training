"""
Sign Language Data Collection Tool
Collects sign language data via webcam using MediaPipe for real-time keypoint extraction.
"""

import cv2
import numpy as np
import os
import sys
import time
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.mediapipe_utils import HandLandmarkExtractor
except ImportError:
    print("❌ Could not import MediaPipe utils. Make sure you're running from the project root.")
    sys.exit(1)

class SignLanguageDataCollector:
    def __init__(self, output_dir="datasets/sign_language"):
        self.output_dir = output_dir
        self.extractor = HandLandmarkExtractor()
        self.cap = None
        self.recording = False
        self.current_class = None
        self.sequence_data = []
        self.sequence_length = 16  # Number of frames to record per gesture
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def start_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized successfully")
    
    def stop_camera(self):
        """Stop camera"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera stopped")
    
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
        
        files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        return len(files)
    
    def record_sequence(self):
        """Record a sequence of hand landmarks"""
        if not self.current_class:
            print("❌ No class selected. Please create or select a class first.")
            return False
        
        print(f"\n🎬 Recording sequence for class: {self.current_class}")
        print("Press SPACE to start recording, ESC to cancel")
        
        self.sequence_data = []
        recording_started = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame")
                return False
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks = self.extractor.extract_landmarks(frame)
            
            # Draw landmarks on frame
            if landmarks is not None:
                # Draw hand landmarks (simplified visualization)
                for i in range(0, len(landmarks), 3):
                    x = int(landmarks[i] * frame.shape[1])
                    y = int(landmarks[i+1] * frame.shape[1])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Add text overlay
            if recording_started:
                cv2.putText(frame, f"Recording... {len(self.sequence_data)}/{self.sequence_length}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Press SPACE to stop", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Class: {self.current_class}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Press SPACE to start recording", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, "ESC to cancel", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Sign Language Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("❌ Recording cancelled")
                return False
            
            elif key == ord(' '):  # SPACE
                if not recording_started:
                    recording_started = True
                    print("🎬 Recording started...")
                else:
                    # Stop recording
                    break
            
            # Record landmarks if recording
            if recording_started and landmarks is not None:
                self.sequence_data.append(landmarks)
                
                if len(self.sequence_data) >= self.sequence_length:
                    print("✅ Sequence length reached, stopping...")
                    break
        
        # Save sequence
        if len(self.sequence_data) > 0:
            return self.save_sequence()
        else:
            print("❌ No data recorded")
            return False
    
    def save_sequence(self):
        """Save the recorded sequence"""
        if len(self.sequence_data) == 0:
            return False
        
        # Pad or truncate to sequence_length
        while len(self.sequence_data) < self.sequence_length:
            self.sequence_data.append(np.zeros(63))  # Pad with zeros
        
        self.sequence_data = self.sequence_data[:self.sequence_length]  # Truncate if too long
        
        # Convert to numpy array
        sequence_array = np.array(self.sequence_data, dtype=np.float32)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_class}_{timestamp}.npy"
        filepath = os.path.join(self.output_dir, self.current_class, filename)
        
        # Save
        np.save(filepath, sequence_array)
        
        print(f"✅ Sequence saved: {filename}")
        print(f"   Shape: {sequence_array.shape}")
        print(f"   Samples in class: {self.count_samples(self.current_class)}")
        
        return True
    
    def upload_data(self, file_path, class_name):
        """Upload and process existing data file"""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return False
        
        # Create class directory if it doesn't exist
        class_dir = os.path.join(self.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy file to class directory
        filename = os.path.basename(file_path)
        dest_path = os.path.join(class_dir, filename)
        
        try:
            import shutil
            shutil.copy2(file_path, dest_path)
            print(f"✅ File uploaded: {filename}")
            print(f"   Class: {class_name}")
            print(f"   Samples in class: {self.count_samples(class_name)}")
            return True
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    def show_statistics(self):
        """Show data collection statistics"""
        classes = self.list_classes()
        
        print("\n📊 DATA COLLECTION STATISTICS")
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
    
    def interactive_collection(self):
        """Interactive data collection session"""
        print("🎬 SIGN LANGUAGE DATA COLLECTION")
        print("=" * 50)
        
        try:
            self.start_camera()
            
            while True:
                print("\n📋 OPTIONS:")
                print("1. Create new class")
                print("2. Select existing class")
                print("3. Record sequence")
                print("4. Upload data file")
                print("5. Show statistics")
                print("6. List classes")
                print("0. Exit")
                
                choice = input("\nEnter choice (0-6): ").strip()
                
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
                        self.record_sequence()
                    else:
                        print("❌ No class selected. Please create or select a class first.")
                elif choice == '4':
                    file_path = input("Enter file path: ").strip()
                    if file_path and self.current_class:
                        self.upload_data(file_path, self.current_class)
                    else:
                        print("❌ Please provide file path and select a class.")
                elif choice == '5':
                    self.show_statistics()
                elif choice == '6':
                    classes = self.list_classes()
                    if classes:
                        print("\nAvailable classes:")
                        for cls in classes:
                            count = self.count_samples(cls)
                            print(f"  {cls}: {count} samples")
                    else:
                        print("No classes found.")
                else:
                    print("❌ Invalid choice")
        
        except KeyboardInterrupt:
            print("\n👋 Collection interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.stop_camera()

def main():
    collector = SignLanguageDataCollector()
    collector.interactive_collection()

if __name__ == "__main__":
    main()
