import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import time
import keyboard

# ==== Config ====
MODEL_PATH = "sign_model.pt"
SEQUENCE_LEN = 30
INPUT_SIZE = 63
LABELS = ["two", "five"]


# ==== Model Definition ====
class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ==== Load Model ====
model = SignLSTM(INPUT_SIZE, 128, num_classes=len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()
print("[✔] Model loaded")

# ==== Mediapipe Hands ====
mp_hands = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)

print("[INFO] Hold SPACE to record and predict. ESC to quit.")
sequence = []
last_predict_time = 0
recording = False
current_prediction = ""

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[✖] Cannot read frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # === Show prediction on the frame ===
        cv2.putText(
            frame,
            f"Prediction: {current_prediction}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Webcam", frame)
        cv2.waitKey(1)

        if keyboard.is_pressed("esc"):
            print("[EXIT] ESC pressed")
            break

        if keyboard.is_pressed("space"):
            if not recording:
                print("[REC] Started recording...")
                recording = True

            result = mp_hands.process(rgb)
            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]
                coords = [
                    coord for pt in landmarks.landmark for coord in (pt.x, pt.y, pt.z)
                ]
                sequence.append(coords)

                if len(sequence) > SEQUENCE_LEN:
                    sequence = sequence[-SEQUENCE_LEN:]  # keep only the latest 30

            if (
                len(sequence) == SEQUENCE_LEN
                and (time.time() - last_predict_time) > 0.5
            ):
                input_tensor = torch.tensor([sequence], dtype=torch.float32)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                    current_prediction = LABELS[pred_idx]
                    print(f"[PREDICTION] {current_prediction}")
                last_predict_time = time.time()

        else:
            if recording:
                print("[REC] Stopped recording.")
                recording = False
                sequence = []
                current_prediction = ""

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

cap.release()
cv2.destroyAllWindows()


# ubuntu run command
# sudo /home/dextra/projects/model_training/.venv/bin/python /home/dextra/projects/model_training/test.py
