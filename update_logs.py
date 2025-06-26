import os
from collections import Counter

DATA_DIR = "data"
OUTPUT_FILE = "dataset_summary.txt"

if not os.path.exists(DATA_DIR):
    print(f"[✖] Directory '{DATA_DIR}' does not exist.")
    exit()

# Get all json files and extract labels
filenames = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
labels = [f.split("_")[0] for f in filenames]

# Count occurrences
counts = Counter(labels)

# Print summary
print("[✔] Dataset Summary:")
for label, count in counts.items():
    print(f"{label}: {count}")

# Write to file
with open(OUTPUT_FILE, "w") as f:
    f.write("Dataset Summary:\n")
    for label, count in counts.items():
        f.write(f"{label}: {count}\n")

print(f"[✔] Summary saved to {OUTPUT_FILE}")
