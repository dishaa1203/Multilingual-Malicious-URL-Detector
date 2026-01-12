import os
import pandas as pd

# Paths
DATASET_PATH = "data/file_dataset.csv"
FILES_DIR = "data/files"

# Ensure files directory exists
os.makedirs(FILES_DIR, exist_ok=True)

# Load the dataset
df = pd.read_csv(DATASET_PATH)

# Create dummy files if missing
for fpath in df["file_path"]:
    if not os.path.exists(fpath):
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("This is a dummy file for testing.\n")
        print(f"Created dummy file: {fpath}")
    else:
        print(f"Already exists: {fpath}")

print("\n✅ All dummy files are ready.")
