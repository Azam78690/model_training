"""
MediaPipe utilities for hand landmark extraction
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple

class HandLandmarkExtractor:
    def __init__(self):
        """Initialize MediaPipe hands solution"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Flattened landmarks array of shape (63,) or None if no hand detected
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Get the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmarks and flatten
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks, dtype=np.float32)
        
        return None
    
    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw hand landmarks on image
        
        Args:
            image: Input image
            landmarks: Landmarks array of shape (63,)
            
        Returns:
            Image with landmarks drawn
        """
        if landmarks is None:
            return image
        
        # Reshape landmarks to (21, 3)
        landmarks_3d = landmarks.reshape(-1, 3)
        
        # Convert normalized coordinates to pixel coordinates
        h, w = image.shape[:2]
        for landmark in landmarks_3d:
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        return image
    
    def close(self):
        """Close MediaPipe hands solution"""
        self.hands.close()
