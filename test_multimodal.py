# test_multimodal.py
import os
from predict_multimodal import predict_sample

# -----------------------------
# Define test samples
# -----------------------------
test_urls = [
    "http://example-phishing.com/login",
    "http://safe-website.com"
]

test_texts = [
    "Please verify your account immediately",
    "Hello, how are you doing today?"
]

test_files = [
    "data/files/file1.pdf",  # make sure these exist
    "data/files/file2.txt"
]

# -----------------------------
# Run predictions
# -----------------------------
print("\n=== Multimodal Test Predictions ===\n")

for i in range(max(len(test_urls), len(test_texts), len(test_files))):
    url = test_urls[i] if i < len(test_urls) else None
    text = test_texts[i] if i < len(test_texts) else None
    file_path = test_files[i] if i < len(test_files) else None

    pred_class, pred_prob = predict_sample(url=url, text=text, file_path=file_path)

    print(f"Sample {i+1}:")
    if url:
        print(f"  URL: {url}")
    if text:
        print(f"  Text: {text}")
    if file_path:
        print(f"  File: {file_path}")
    print(f"  Predicted class: {pred_class}")
    print(f"  Class probabilities: {pred_prob}\n")
