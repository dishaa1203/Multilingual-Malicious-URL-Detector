import csv
import os

# Input and output paths
input_csv = os.path.join("data", "top-1m.csv")
output_txt = os.path.join("data", "benign_urls.txt")

limit = 51000  # Number of URLs to save
count = 0

if not os.path.exists(input_csv):
    print(f"Error: {input_csv} not found. Place top-1m.csv in the data folder.")
    exit()

with open(input_csv, "r", encoding="utf-8") as f, open(output_txt, "w", encoding="utf-8") as out:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 2:
            continue
        out.write(f"http://{row[1]}\n")
        count += 1
        if count >= limit:
            break

print(f"Top {limit} benign URLs saved to {output_txt}")
