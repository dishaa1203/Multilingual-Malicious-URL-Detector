import os
import json
import csv
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # reproducibility

# Input files
phishing_file = os.path.join("data", "merged_urls.txt")
benign_file = os.path.join("data", "benign_urls.txt")
text_file = os.path.join("data", "text_samples.txt")  # optional text samples
file_hash_file = os.path.join("data", "file_hashes.txt")  # optional malicious files

# Output file
output_file = os.path.join("data", "merged.jsonl")
input_csv = "data/dataset.csv"  # or whatever your source is
output_jsonl = "data/merged.jsonl"

with open(input_csv, "r", encoding="utf-8") as f_in, open(output_jsonl, "w", encoding="utf-8") as f_out:
    reader = csv.DictReader(f_in)
    for row in reader:
        json_line = {
            "modality": "url",
            "url": row["url"],
            "label": int(row["label"])
        }
        f_out.write(json.dumps(json_line) + "\n")
# Helper function for language detection
def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Initialize dataset list
dataset = []

# 1️⃣ Process URL data
def process_urls(file_path, label, modality="url"):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    entries = []
    for url in urls:
        text = url  # replace with page title/snippet if available
        language = detect_lang(text)
        entry = {
            "id": f"url_{len(dataset)+len(entries)}",
            "timestamp": "",
            "source": os.path.basename(file_path),
            "modality": modality,
            "language": language,
            "text": text,
            "url": url,
            "file_hash": "",
            "label": label,
            "metadata": {}
        }
        entries.append(entry)
    return entries

dataset += process_urls(phishing_file, 1)
dataset += process_urls(benign_file, 0)

# 2️⃣ Process text samples (misinformation)
if os.path.exists(text_file):
    with open(text_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            text = line.strip()
            if not text:
                continue
            language = detect_lang(text)
            entry = {
                "id": f"text_{len(dataset)}",
                "timestamp": "",
                "source": os.path.basename(text_file),
                "modality": "text",
                "language": language,
                "text": text,
                "url": "",
                "file_hash": "",
                "label": 1,  # adjust based on your labeling
                "metadata": {}
            }
            dataset.append(entry)

# 3️⃣ Process file hashes (malicious files)
if os.path.exists(file_hash_file):
    with open(file_hash_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            file_hash = line.strip()
            if not file_hash:
                continue
            entry = {
                "id": f"file_{len(dataset)}",
                "timestamp": "",
                "source": os.path.basename(file_hash_file),
                "modality": "file",
                "language": "unknown",
                "text": "",
                "url": "",
                "file_hash": file_hash,
                "label": 1,  # 1 = malicious
                "metadata": {}
            }
            dataset.append(entry)

# Save to JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Merged dataset created with {len(dataset)} entries and saved to {output_file}")
