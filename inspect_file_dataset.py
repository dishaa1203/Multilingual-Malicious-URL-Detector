import pandas as pd
import os

# Load your file dataset
df = pd.read_csv("data/file_dataset.csv")

print("=== First 5 rows of file dataset ===")
print(df.head())

print("\n=== Unique labels ===")
print(df["label"].unique())

print("\n=== Count by label ===")
print(df["label"].value_counts())

# -----------------------------
# Function to detect if a file is text or binary
# -----------------------------
def is_text_file(fpath, blocksize=512):
    try:
        with open(fpath, "rb") as f:
            block = f.read(blocksize)
        return b"\0" not in block  # null byte = likely binary
    except Exception as e:
        print(f"Error checking file {fpath}: {e}")
        return False

# -----------------------------
# Peek into the first sample file
# -----------------------------
if "file_path" in df.columns and not df["file_path"].empty:
    sample_file = df["file_path"].iloc[0]

    if sample_file and os.path.exists(sample_file):
        if is_text_file(sample_file):
            with open(sample_file, "r", encoding="utf-8", errors="ignore") as f:
                print("\n=== First 200 characters of text file ===")
                print(f.read(200))
        else:
            print(f"\n{sample_file} looks like a binary file (not safe to open as text).")
    else:
        print("\n⚠️ Sample file path is missing or does not exist.")
else:
    print("\n⚠️ No 'file_path' column or dataset is empty.")
