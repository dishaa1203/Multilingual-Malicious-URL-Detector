import requests

print("Downloading data from PhishTank...")

url = "http://data.phishtank.com/data/online-valid.csv"
response = requests.get(url)

print("HTTP Status Code:", response.status_code)

if response.status_code == 200:
    with open("phishtank.csv", "wb") as file:
        file.write(response.content)
    print("Download complete! File saved as phishtank.csv")

    # Preview first 5 lines of the downloaded CSV
    print("\nPreview of phishing URLs (first 5 lines):")
    with open("phishtank.csv", "r", encoding="utf-8") as file:
        for i in range(5):
            print(file.readline().strip())
else:
    print("Download failed. Status code:", response.status_code)
