"""
Sign Language Prediction - Real-time and file-based inference
"""

import os
import sys
import torch
import cv2
import numpy as np
import argparse
from typing import Optional, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sign_model import SignGRUClassifier
from utils.mediapipe_utils import HandLandmarkExtractor
from utils.tts_utils import TTS_System
from versioning.registry_manager import ModelRegistry

class SignLanguagePredictor:
    def __init__(self, model_path: str, use_tts: bool = True):
        """
        Initialize Sign Language Predictor
        
        Args:
            model_path: Path to trained model
            use_tts: Whether to use text-to-speech
        """
        self.model_path = model_path
        self.model = None
        self.classes = None
        self.class_to_idx = None
        self.max_seq_length = 16
        self.extractor = HandLandmarkExtractor()
        self.tts = TTS_System() if use_tts else None
        
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
        self.max_seq_length = checkpoint.get('max_seq_length', 16)
        
        # Initialize model
        self.model = SignGRUClassifier(
            input_size=model_info['input_size'],
            hidden_size=model_info['hidden_size'],
            num_classes=model_info['num_classes'],
            num_layers=model_info['num_layers']
        )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Classes: {self.classes}")
        print(f"Model info: {model_info}")
    
    def predict_sequence(self, sequence: np.ndarray) -> Tuple[str, float]:
        """
        Predict class for a sequence of landmarks
        
        Args:
            sequence: Array of shape (seq_len, 63)
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Pad or truncate sequence
        if len(sequence) < self.max_seq_length:
            padding = np.zeros((self.max_seq_length - len(sequence), 63))
            sequence = np.vstack([sequence, padding])
        else:
            sequence = sequence[:self.max_seq_length]
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        
        # Predict
        with torch.no_grad():
            probabilities = self.model.predict(sequence_tensor)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.classes[predicted_idx.item()]
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def predict_from_file(self, file_path: str) -> Tuple[str, float]:
        """
        Predict from a saved sequence file
        
        Args:
            file_path: Path to .npy file containing sequence
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sequence = np.load(file_path)
        return self.predict_sequence(sequence)
    
    def predict_realtime(self, duration: int = 5) -> Tuple[str, float]:
        """
        Real-time prediction using webcam
        
        Args:
            duration: Duration to record in seconds
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Recording for {duration} seconds...")
        print("Press 'q' to stop early")
        
        sequence = []
        start_time = cv2.getTickCount()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Extract landmarks
                landmarks = self.extractor.extract_landmarks(frame)
                
                if landmarks is not None:
                    sequence.append(landmarks)
                
                # Draw landmarks on frame
                if landmarks is not None:
                    frame = self.extractor.draw_landmarks(frame, landmarks)
                
                # Add text overlay
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                remaining = max(0, duration - elapsed)
                
                cv2.putText(frame, f"Recording... {remaining:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to stop", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Sign Language Prediction', frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Check duration
                if elapsed >= duration:
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if len(sequence) == 0:
            raise RuntimeError("No hand landmarks detected during recording")
        
        # Convert to numpy array
        sequence = np.array(sequence)
        
        # Predict
        predicted_class, confidence = self.predict_sequence(sequence)
        
        return predicted_class, confidence
    
    def predict_with_tts(self, predicted_class: str, confidence: float):
        """Predict and speak the result"""
        print(f"Predicted: {predicted_class} (confidence: {confidence:.2f})")
        
        if self.tts:
            self.tts.speak(predicted_class)

def main():
    parser = argparse.ArgumentParser(description='Sign Language Prediction')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--input', type=str, choices=['realtime', 'file'],
                       default='realtime', help='Input type')
    parser.add_argument('--file_path', type=str,
                       help='Path to input file (if input=file)')
    parser.add_argument('--duration', type=int, default=5,
                       help='Recording duration in seconds (if input=realtime)')
    parser.add_argument('--no_tts', action='store_true',
                       help='Disable text-to-speech')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = SignLanguagePredictor(
            model_path=args.model_path,
            use_tts=not args.no_tts
        )
        
        # Make prediction
        if args.input == 'realtime':
            predicted_class, confidence = predictor.predict_realtime(args.duration)
        elif args.input == 'file':
            if not args.file_path:
                print("Error: file_path required when input=file")
                return
            predicted_class, confidence = predictor.predict_from_file(args.file_path)
        else:
            print("Error: Invalid input type")
            return
        
        # Output result
        predictor.predict_with_tts(predicted_class, confidence)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
