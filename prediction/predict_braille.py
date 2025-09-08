"""
Braille Prediction - Image-based and manual input inference
"""

import os
import sys
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.braille_model import BrailleCNNClassifier
from utils.braille_mapping import braille_to_char, char_to_braille
from utils.tts_utils import TTS_System
from versioning.registry_manager import ModelRegistry

class BraillePredictor:
    def __init__(self, model_path: Optional[str] = None, use_tts: bool = True):
        """
        Initialize Braille Predictor
        
        Args:
            model_path: Path to trained model (optional for manual input)
            use_tts: Whether to use text-to-speech
        """
        self.model_path = model_path
        self.model = None
        self.classes = None
        self.class_to_idx = None
        self.transform = None
        self.tts = TTS_System() if use_tts else None
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Get model info
        model_info = checkpoint['model_info']
        self.classes = checkpoint['classes']
        self.class_to_idx = checkpoint['class_to_idx']
        
        # Initialize model
        self.model = BrailleCNNClassifier(num_classes=model_info['num_classes'])
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        print(f"Model loaded successfully!")
        print(f"Classes: {self.classes}")
        print(f"Model info: {model_info}")
    
    def predict_image(self, image_path: str) -> Tuple[str, float]:
        """
        Predict class for a braille image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('L')
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        # Predict
        with torch.no_grad():
            probabilities = self.model.predict(image_tensor)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.classes[predicted_idx.item()]
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def predict_manual_input(self, pattern: List[int]) -> str:
        """
        Predict from manual 6-dot pattern input
        
        Args:
            pattern: List of 6 binary values
            
        Returns:
            Predicted character
        """
        if len(pattern) != 6:
            raise ValueError("Pattern must have exactly 6 dots")
        
        if not all(x in [0, 1] for x in pattern):
            raise ValueError("Pattern must contain only 0s and 1s")
        
        # Use dictionary lookup
        predicted_char = braille_to_char(pattern)
        
        return predicted_char
    
    def predict_realtime(self, duration: int = 3) -> Tuple[str, float]:
        """
        Real-time prediction using webcam
        
        Args:
            duration: Duration to record in seconds
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Recording for {duration} seconds...")
        print("Press 'q' to stop early")
        
        start_time = cv2.getTickCount()
        captured_image = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Add text overlay
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                remaining = max(0, duration - elapsed)
                
                cv2.putText(frame, f"Recording... {remaining:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to stop", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Braille Prediction', frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Check duration
                if elapsed >= duration:
                    captured_image = frame.copy()
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if captured_image is None:
            raise RuntimeError("No image captured")
        
        # Save temporary image
        temp_path = "temp_braille_image.jpg"
        cv2.imwrite(temp_path, captured_image)
        
        try:
            # Predict from captured image
            predicted_class, confidence = self.predict_image(temp_path)
            return predicted_class, confidence
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def predict_with_tts(self, predicted_class: str, confidence: Optional[float] = None):
        """Predict and speak the result"""
        if confidence is not None:
            print(f"Predicted: {predicted_class} (confidence: {confidence:.2f})")
        else:
            print(f"Predicted: {predicted_class}")
        
        if self.tts:
            self.tts.speak(predicted_class)

def main():
    parser = argparse.ArgumentParser(description='Braille Prediction')
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model (optional for manual input)')
    parser.add_argument('--input', type=str, choices=['image', 'realtime', 'manual'],
                       default='manual', help='Input type')
    parser.add_argument('--image_path', type=str,
                       help='Path to input image (if input=image)')
    parser.add_argument('--pattern', type=str,
                       help='6-dot pattern as comma-separated values (e.g., "1,0,0,0,0,0")')
    parser.add_argument('--duration', type=int, default=3,
                       help='Recording duration in seconds (if input=realtime)')
    parser.add_argument('--no_tts', action='store_true',
                       help='Disable text-to-speech')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = BraillePredictor(
            model_path=args.model_path,
            use_tts=not args.no_tts
        )
        
        # Make prediction
        if args.input == 'image':
            if not args.image_path:
                print("Error: image_path required when input=image")
                return
            predicted_class, confidence = predictor.predict_image(args.image_path)
            predictor.predict_with_tts(predicted_class, confidence)
            
        elif args.input == 'realtime':
            if not args.model_path:
                print("Error: model_path required when input=realtime")
                return
            predicted_class, confidence = predictor.predict_realtime(args.duration)
            predictor.predict_with_tts(predicted_class, confidence)
            
        elif args.input == 'manual':
            if not args.pattern:
                print("Error: pattern required when input=manual")
                return
            
            # Parse pattern
            try:
                pattern = [int(x.strip()) for x in args.pattern.split(',')]
            except ValueError:
                print("Error: Invalid pattern format. Use comma-separated values (e.g., '1,0,0,0,0,0')")
                return
            
            predicted_class = predictor.predict_manual_input(pattern)
            predictor.predict_with_tts(predicted_class)
            
        else:
            print("Error: Invalid input type")
            return
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
