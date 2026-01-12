import os
import random

# Input files
phishing_file = os.path.join("data", "merged_urls.txt")
benign_file = os.path.join("data", "benign_urls.txt")
output_file = os.path.join("data", "dataset.csv")

# Check if files exist
if not os.path.exists(phishing_file):
    print(f"Error: {phishing_file} not found.")
    exit()

if not os.path.exists(benign_file):
    print(f"Error: {benign_file} not found.")
    exit()

# Read phishing URLs
with open(phishing_file, "r", encoding="utf-8") as f:
    phishing_urls = [line.strip() for line in f if line.strip()]

# Read benign URLs
with open(benign_file, "r", encoding="utf-8") as f:
    benign_urls = [line.strip() for line in f if line.strip()]

# Label datasets (1 = phishing, 0 = safe)
phishing_data = [(url, 1) for url in phishing_urls]
benign_data = [(url, 0) for url in benign_urls]

# Merge and shuffle
dataset = phishing_data + benign_data
random.shuffle(dataset)

# Save to CSV
with open(output_file, "w", encoding="utf-8") as f:
    f.write("url,label\n")
    for url, label in dataset:
        f.write(f"{url},{label}\n")

print(f"Dataset created with {len(dataset)} URLs and saved to {output_file}")
