import requests
import os

# Create a folder for data if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Function to download and save data
def download_file(url, filename):
    print(f"Downloading from {url} ...")
    response = requests.get(url)
    if response.status_code == 200:
        path = os.path.join("data", filename)
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Saved: {path}")
        return path
    else:
        print(f"Failed to download from {url} (status: {response.status_code})")
        return None

# Download PhishTank
phishtank_path = download_file("http://data.phishtank.com/data/online-valid.csv", "phishtank.csv")

# Download OpenPhish
openphish_path = download_file("https://openphish.com/feed.txt", "openphish.txt")

# Merge PhishTank and OpenPhish URLs
all_urls = set()

# Add PhishTank URLs
if phishtank_path and os.path.exists(phishtank_path):
    with open(phishtank_path, "r", encoding="utf-8") as f:
        next(f)  # skip header line
        for line in f:
            all_urls.add(line.split(",")[1].strip())  # URL column

# Add OpenPhish URLs
if openphish_path and os.path.exists(openphish_path):
    with open(openphish_path, "r", encoding="utf-8") as f:
        for line in f:
            all_urls.add(line.strip())

# Save merged URLs
merged_path = os.path.join("data", "merged_urls.txt")
with open(merged_path, "w", encoding="utf-8") as f:
    for url in all_urls:
        f.write(url + "\n")

print(f"\nTotal URLs collected: {len(all_urls)}")
print("First 10 URLs:")
for i, url in enumerate(list(all_urls)[:10]):
    print(f"{i+1}. {url}")
