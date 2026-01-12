# stacking_meta_aligned.py
import os
import csv
import joblib
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import warnings

from train_model import extract_features
from modules.url_cnn import URLCNN, URLDataset

warnings.filterwarnings("ignore")  # suppress LightGBM warnings

# -----------------------------
# Config / Paths
# -----------------------------
URL_DATASET_PATH  = "data/dataset.csv"
TEXT_DATASET_PATH = "data/text_dataset.csv"
FILE_DATASET_PATH = "data/file_dataset.csv"

META_MODEL_PATH   = "models/meta_model_aligned_multimodal.joblib"
CNN_MODEL_PATH    = "models/url_cnn_baseline.pth"

os.makedirs("data/files", exist_ok=True)

# -----------------------------
# Create dummy datasets if missing
# -----------------------------
if not os.path.exists(URL_DATASET_PATH):
    with open(URL_DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerow(["http://phishing.com/login", 1])
        writer.writerow(["http://example.com", 0])

if not os.path.exists(TEXT_DATASET_PATH):
    with open(TEXT_DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerow(["Login to verify your account", 1])
        writer.writerow(["Hello, how are you?", 0])

if not os.path.exists(FILE_DATASET_PATH):
    # create dummy files
    dummy_file1 = "data/files/file1.txt"
    dummy_file2 = "data/files/file2.txt"
    with open(dummy_file1, "w", encoding="utf-8") as f: f.write("Malicious content")
    with open(dummy_file2, "w", encoding="utf-8") as f: f.write("Benign content")
    # CSV
    with open(FILE_DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "file_hash", "label"])
        writer.writerow([dummy_file1, None, 1])
        writer.writerow([dummy_file2, None, 0])

# -----------------------------
# Load URL dataset
# -----------------------------
urls_df = pd.read_csv(URL_DATASET_PATH)
urls = urls_df["url"].tolist()
y_url = urls_df["label"].values

# -----------------------------
# Load text dataset
# -----------------------------
texts_df = pd.read_csv(TEXT_DATASET_PATH)
texts = texts_df["text"].tolist()
y_text = texts_df["label"].values

# -----------------------------
# Load file dataset
# -----------------------------
files_df = pd.read_csv(FILE_DATASET_PATH)
file_paths = files_df.get("file_path", [None]*len(files_df)).tolist()
# file_hashes is already built from CSV reading
# no need to convert
# file_hashes = ...
pass  # or just remove this line

y_file = files_df["label"].values

print(f"Loaded {len(file_paths)} file samples (paths + hashes)")

# -----------------------------
# Align datasets
# -----------------------------
n_samples = min(len(urls), len(texts), len(file_paths))
urls = urls[:n_samples]
texts = texts[:n_samples]
file_paths = file_paths[:n_samples]
y = y_url[:n_samples]  # assume labels aligned

# -----------------------------
# Split train/test (small datasets may skip stratify)
# -----------------------------
if len(np.unique(y)) > 1:
    stratify = y
else:
    stratify = None

urls_train, urls_test, texts_train, texts_test, files_train, files_test, y_train, y_test = train_test_split(
    urls, texts, file_paths, y, test_size=0.5, random_state=42, stratify=stratify
)

# -----------------------------
# Load classical URL models
# -----------------------------
rf = joblib.load("models/url_rf_baseline.joblib")["model"]
lgbm = joblib.load("models/url_lgbm_baseline.joblib")["model"]
xgb = joblib.load("models/url_xgb_baseline.joblib")["model"]

def get_url_probs(url_list):
    X = pd.DataFrame([extract_features(u) for u in url_list])
    return rf.predict_proba(X)[:,1], lgbm.predict_proba(X)[:,1], xgb.predict_proba(X)[:,1]

rf_train, lgb_train, xgb_train = get_url_probs(urls_train)
rf_test,  lgb_test,  xgb_test  = get_url_probs(urls_test)

# -----------------------------
# Load CNN URL model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = URLCNN().to(device)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH))
cnn_model.eval()

def get_cnn_probs(url_list, labels_list):
    dataset = URLDataset(url_list, labels_list)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    probs = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            y_pred = cnn_model(x_batch).squeeze()
            if y_pred.ndim == 0:
                y_pred = y_pred.unsqueeze(0)
            probs.extend(y_pred.cpu().numpy())
    return np.array(probs)

cnn_train = get_cnn_probs(urls_train, y_train)
cnn_test  = get_cnn_probs(urls_test, y_test)

# -----------------------------
# Text modality
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_text_train = vectorizer.fit_transform(texts_train)
X_text_test  = vectorizer.transform(texts_test)

text_clf = LogisticRegression(max_iter=1000)
text_clf.fit(X_text_train, y_train)
text_train = text_clf.predict_proba(X_text_train)[:,1]
text_test  = text_clf.predict_proba(X_text_test)[:,1]

# ✅ Save the fitted text vectorizer and classifier for later prediction
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/text_vectorizer.joblib")
joblib.dump(text_clf, "models/text_clf.joblib")

# -----------------------------
# File modality
# -----------------------------
def is_text_file(fpath, blocksize=512):
    try:
        with open(fpath, "rb") as f:
            block = f.read(blocksize)
        return b"\0" not in block
    except:
        return False

text_file_contents = []
binary_file_features = []

for fpath in files_train + files_test:
    if fpath is None or not os.path.exists(fpath):
        text_file_contents.append("")
        binary_file_features.append(np.zeros(256))
        continue
    if is_text_file(fpath):
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text_file_contents.append(f.read())
        binary_file_features.append(np.zeros(256))
    else:
        with open(fpath, "rb") as f:
            content = f.read()
        hist = np.zeros(256)
        for b in content:
            hist[b] += 1
        hist /= len(content)
        binary_file_features.append(hist)
        text_file_contents.append("")

# Split back train/test
text_file_train  = text_file_contents[:len(urls_train)]
text_file_test   = text_file_contents[len(urls_train):]
binary_file_train = np.array(binary_file_features[:len(urls_train)])
binary_file_test  = np.array(binary_file_features[len(urls_train):])

# Train text-based file classifier
file_text_clf = LogisticRegression(max_iter=1000)
X_file_text_train = vectorizer.transform(text_file_train)
file_text_clf.fit(X_file_text_train, y_train)
text_file_train_probs = file_text_clf.predict_proba(X_file_text_train)[:,1]
X_file_text_test = vectorizer.transform(text_file_test)
text_file_test_probs = file_text_clf.predict_proba(X_file_text_test)[:,1]

# ✅ Save file text classifier
joblib.dump(file_text_clf, "models/file_text_clf.joblib")

# Train binary file classifier
file_binary_clf = LogisticRegression(max_iter=1000)
file_binary_clf.fit(binary_file_train, y_train)
binary_file_train_probs = file_binary_clf.predict_proba(binary_file_train)[:,1]
binary_file_test_probs  = file_binary_clf.predict_proba(binary_file_test)[:,1]

# ✅ Save file binary classifier
joblib.dump(file_binary_clf, "models/file_binary_clf.joblib")

file_train = np.maximum(text_file_train_probs, binary_file_train_probs)
file_test  = np.maximum(text_file_test_probs,  binary_file_test_probs)
# -----------------------------
# Build meta features
# -----------------------------
meta_X_train = np.column_stack([rf_train, lgb_train, xgb_train, cnn_train, text_train, file_train])
meta_X_test  = np.column_stack([rf_test,  lgb_test,  xgb_test,  cnn_test, text_test, file_test])

# -----------------------------
# Train meta-classifier
# -----------------------------
meta_clf = LogisticRegression(max_iter=1000)
meta_clf.fit(meta_X_train, y_train)
calibrated_clf = CalibratedClassifierCV(meta_clf, method='sigmoid', cv='prefit')
calibrated_clf.fit(meta_X_train, y_train)
y_prob = calibrated_clf.predict_proba(meta_X_test)[:,1]

# -----------------------------
# Evaluate
# -----------------------------
y_pred = calibrated_clf.predict(meta_X_test)
print("\n=== Fully Multimodal Meta-classifier Metrics ===")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# Save meta-model
# -----------------------------
os.makedirs(os.path.dirname(META_MODEL_PATH), exist_ok=True)
joblib.dump({
    "model": calibrated_clf,
    "features": ["rf","lgbm","xgb","cnn","text","file"]
}, META_MODEL_PATH)

print("\nFully plug-and-play multimodal meta-model saved to:", META_MODEL_PATH)
