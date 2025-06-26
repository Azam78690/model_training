import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ========== CONFIG ========== #
DATA_DIR = "data"
SEQUENCE_LEN = 30
INPUT_SIZE = 63  # 21 landmarks * 3 (x, y, z)
BATCH_SIZE = 16
EPOCHS = 500
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================ #


# Extract label from filename
def extract_label(fname):
    return fname.split("_")[0]


# ========== Dataset ========== #
class SignDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.labels = []
        self.label_map = {}
        label_idx = 0

        for file in os.listdir(data_dir):
            if file.endswith(".json"):
                label = extract_label(file)
                if label not in self.label_map:
                    self.label_map[label] = label_idx
                    label_idx += 1

                with open(os.path.join(data_dir, file), "r") as f:
                    seq = json.load(f)

                # Pad or truncate to SEQUENCE_LEN
                if len(seq) < SEQUENCE_LEN:
                    pad = [[0] * INPUT_SIZE] * (SEQUENCE_LEN - len(seq))
                    seq.extend(pad)
                elif len(seq) > SEQUENCE_LEN:
                    seq = seq[:SEQUENCE_LEN]

                self.samples.append(seq)
                self.labels.append(self.label_map[label])

        self.samples = torch.tensor(self.samples, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# ========== Model ========== #
class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last timestep
        return self.fc(out)


# ========== Training ========== #
def train():
    dataset = SignDataset(DATA_DIR)
    num_classes = len(dataset.label_map)
    print(
        f"Loaded {len(dataset)} samples with {num_classes} classes: {dataset.label_map}"
    )

    X_train, X_val, y_train, y_val = train_test_split(
        dataset.samples, dataset.labels, test_size=0.2, stratify=dataset.labels
    )

    train_loader = DataLoader(
        list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=BATCH_SIZE)

    model = SignLSTM(INPUT_SIZE, 128, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            preds = model(batch_X)
            loss = criterion(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                preds = model(batch_X)
                predicted = torch.argmax(preds, dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Val Acc: {acc:.4f}")

    torch.save(model.state_dict(), "sign_model.pt")
    print("[✔] Model saved to sign_model.pt")


if __name__ == "__main__":
    train()
