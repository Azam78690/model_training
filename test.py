import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import time
import os

# ===== Config =====
MODEL_PATH = "sign_model.pt"
SEQUENCE_LEN = 30
INPUT_SIZE = 63
TRIGGER_KEY = ord(' ')  # Space key
LABELS = ["two", "five"]  
# ===================


# ===== Model Definition (same as training) =====
class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ==============================================

# ===== Load model =====
model = SignLSTM(INPUT_SIZE, 128, num_classes=len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()
print("[✔] Model loaded")

# ===== Setup Mediapipe =====
mp_hands = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)

print("[INFO] Hold SPACE to record a sign (2 sec). Release to predict. ESC to quit.")

sequence = []
recording = False
start_time = None
space_was_pressed = False
last_predict_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(rgb)

        # Check for key presses using OpenCV
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()
        
        # Simple state machine for space key
        space_is_pressed = (key == TRIGGER_KEY)
        
        # Start recording when space is pressed and wasn't pressed before
        if space_is_pressed and not space_was_pressed and not recording:
            recording = True
            sequence = []
            start_time = time.time()
            print("[REC] Recording started...")
        
        # Stop recording when space is released and we were recording
        elif not space_is_pressed and space_was_pressed and recording:
            recording = False
            print(f"[STOP] Recording ended. Collected {len(sequence)} frames.")
            
            # Predict with debouncing (prevent multiple predictions)
            if sequence and (current_time - last_predict_time) > 0.5:  # 500ms debounce
                # Pad or trim to fixed SEQUENCE_LEN
                if len(sequence) < SEQUENCE_LEN:
                    pad = [[0] * INPUT_SIZE] * (SEQUENCE_LEN - len(sequence))
                    sequence.extend(pad)
                elif len(sequence) > SEQUENCE_LEN:
                    sequence = sequence[:SEQUENCE_LEN]

                # Convert to tensor and predict
                input_tensor = torch.tensor([sequence], dtype=torch.float32)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                    pred_label = LABELS[pred_idx]
                    print(f"[PREDICTION] {pred_label}")
                last_predict_time = current_time
        
        # Update state for next iteration
        space_was_pressed = space_is_pressed
        
        # Record frames while recording
        if recording:
            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]
                frame_data = [
                    coord for pt in landmarks.landmark for coord in (pt.x, pt.y, pt.z)
                ]
                sequence.append(frame_data)

        # Draw webcam
        cv2.imshow("Webcam", frame)

        if key == 27:  # ESC
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

cap.release()
cv2.destroyAllWindows()
