import csv
import os
import random
import sys

# Let CSV handle very long URLs
csv.field_size_limit(sys.maxsize)

dataset_file = os.path.join("data", "dataset.csv")

if not os.path.exists(dataset_file):
    print(f"Error: {dataset_file} not found. Run prepare_dataset.py first.")
    raise SystemExit

urls = []
skipped = 0

with open(dataset_file, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Defensive parsing
        url = (row.get("url") or "").strip()
        label_raw = row.get("label")

        if not url or label_raw is None or label_raw == "":
            skipped += 1
            continue

        try:
            label = int(label_raw)
        except ValueError:
            skipped += 1
            continue

        if label not in (0, 1):
            skipped += 1
            continue

        urls.append((url, label))

total_urls = len(urls)
phishing_count = sum(1 for _, l in urls if l == 1)
benign_count = total_urls - phishing_count

print("\n=== Dataset Summary ===")
print(f"Total valid rows: {total_urls}")
print(f"Phishing URLs:    {phishing_count}")
print(f"Benign URLs:      {benign_count}")
print(f"Skipped rows:     {skipped}")

# Guard against sampling from very small sets
sample_n = min(10, total_urls)
if sample_n > 0:
    print("\n=== Sample URLs ===")
    for url, label in random.sample(urls, sample_n):
        print(f"{'Phishing' if label == 1 else 'Benign'} → {url}")
else:
    print("\nNo valid rows to sample.")
