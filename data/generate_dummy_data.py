# generate_dummy_data.py
import os
import csv

# -----------------------------
# Config / Paths
# -----------------------------
DATA_DIR = "data"
FILES_DIR = os.path.join(DATA_DIR, "files")
os.makedirs(FILES_DIR, exist_ok=True)

# -----------------------------
# 1️⃣ URL dataset
# -----------------------------
url_dataset = [
    ("http://malicious.com/login", 1),
    ("http://phishingsite.net", 1),
    ("http://example.com", 0),
    ("http://safe-site.org", 0)
]

with open(os.path.join(DATA_DIR, "dataset.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["url", "label"])
    writer.writerows(url_dataset)

# -----------------------------
# 2️⃣ Text dataset
# -----------------------------
text_dataset = [
    ("Login to verify your account", 1),
    ("Your account will be suspended", 1),
    ("Hello, how are you?", 0),
    ("Team lunch at 3pm", 0)
]

with open(os.path.join(DATA_DIR, "text_dataset.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(text_dataset)

# -----------------------------
# 3️⃣ File dataset
# -----------------------------
file_dataset = [
    ("file1.pdf", 1),
    ("file2.docx", 0),
    ("file3.txt", 0),
    ("file4.xls", 1)
]

with open(os.path.join(DATA_DIR, "file_dataset.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file_path", "label"])
    for fname, label in file_dataset:
        path = os.path.join(FILES_DIR, fname)
        # Create dummy files
        with open(path, "w", encoding="utf-8") as ff:
            ff.write(f"Dummy content for {fname}\n")
        writer.writerow([path, label])

print("✅ Dummy datasets and files generated successfully!")
