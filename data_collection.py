import cv2
import mediapipe as mp
import json
import time
import os
import keyboard
import random

# ========== Prompt for Label ==========
label = (
    input("Enter the sign label you're recording (e.g., 'one', 'hello'):\n> ")
    .strip()
    .lower()
)
if not label:
    print("Label is required. Exiting.")
    exit()
# ======================================

# ========== Setup ==========
SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)
SEQUENCE_LEN = 30
INPUT_SIZE = 63
TRIGGER_KEY = "space"
# ===========================

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands.Hands()

recording = False
sequence = []

print("[INFO] Hold SPACE to record. Release to stop. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    if keyboard.is_pressed(TRIGGER_KEY):
        if not recording:
            print("[REC] Recording started...")
            recording = True
            sequence = []

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            frame_data = [coord for pt in hand.landmark for coord in (pt.x, pt.y, pt.z)]
            sequence.append(frame_data)

    elif recording:
        recording = False
        print(f"[STOP] Recording ended. Frames captured: {len(sequence)}")

        if sequence:
            # Pad or trim sequence
            if len(sequence) < SEQUENCE_LEN:
                pad = [[0] * INPUT_SIZE] * (SEQUENCE_LEN - len(sequence))
                sequence.extend(pad)
            elif len(sequence) > SEQUENCE_LEN:
                sequence = sequence[:SEQUENCE_LEN]

            # Generate unique filename
            while True:
                rand_id = random.randint(100, 99999)
                filename = f"{label}_{rand_id}.json"
                path = os.path.join(SAVE_DIR, filename)
                if not os.path.exists(path):
                    break  # found a unique name

            # Save file
            with open(path, "w") as f:
                json.dump(sequence, f)
            print(f"[✔] Saved to {path}")

    # Show camera feed
    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
