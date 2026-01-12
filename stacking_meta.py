# stacking_meta.py

import os
import joblib
import torch
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from modules.url_cnn import URLCNN, URLDataset
from train_model import extract_features  # reuse URL feature extraction

# -----------------------------
# Config / Paths
# -----------------------------
URL_DATASET_PATH  = "data/dataset.csv"
TEXT_DATASET_PATH = "data/text_dataset.csv"
FILE_DATASET_PATH = "data/file_dataset.csv"
META_MODEL_PATH   = "models/meta_model.joblib"

# -----------------------------
# Load URL dataset
# -----------------------------
urls, url_labels = [], []
with open(URL_DATASET_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        urls.append(row["url"])
        url_labels.append(int(row["label"]))
y_url = np.array(url_labels)

# Split train/test for URL
urls_train, urls_test, y_train, y_test = train_test_split(
    urls, y_url, test_size=0.2, random_state=42, stratify=y_url
)

# -----------------------------
# Load classical URL models
# -----------------------------
rf = joblib.load("models/url_rf_baseline.joblib")["model"]
lgbm = joblib.load("models/url_lgbm_baseline.joblib")["model"]
xgb = joblib.load("models/url_xgb_baseline.joblib")["model"]

def get_url_probs(url_list):
    X = np.array([extract_features(u) for u in url_list], dtype=float)
    return rf.predict_proba(X)[:,1], lgbm.predict_proba(X)[:,1], xgb.predict_proba(X)[:,1]

rf_train, lgb_train, xgb_train = get_url_probs(urls_train)
rf_test,  lgb_test,  xgb_test  = get_url_probs(urls_test)

# -----------------------------
# Load CNN URL model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = URLCNN().to(device)
cnn_model.load_state_dict(torch.load("models/url_cnn_baseline.pth", map_location=device))
cnn_model.eval()

def get_cnn_probs(url_list, labels_list):
    dataset = URLDataset(url_list, labels_list)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    probs = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            y_pred = cnn_model(x_batch).squeeze()
            probs.extend(y_pred.cpu().numpy())
    return np.array(probs)

cnn_train = get_cnn_probs(urls_train, y_train)
cnn_test  = get_cnn_probs(urls_test, y_test)

# -----------------------------
# Load text dataset
# -----------------------------
text_samples, text_labels = [], []
with open(TEXT_DATASET_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text_samples.append(row["text"])
        text_labels.append(int(row["label"]))
y_text = np.array(text_labels)

# Split train/test
text_train_samples, text_test_samples, y_text_train, y_text_test = train_test_split(
    text_samples, y_text, test_size=0.2, random_state=42, stratify=y_text
)

# TF-IDF + LogisticRegression
vectorizer = TfidfVectorizer(max_features=5000)
X_text_train = vectorizer.fit_transform(text_train_samples)
X_text_test  = vectorizer.transform(text_test_samples)

text_clf = LogisticRegression(max_iter=1000)
text_clf.fit(X_text_train, y_text_train)

text_train_probs = text_clf.predict_proba(X_text_train)[:,1]
text_test_probs  = text_clf.predict_proba(X_text_test)[:,1]

# -----------------------------
# Load file dataset
# -----------------------------
file_hashes, file_labels = [], []
with open(FILE_DATASET_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        file_hashes.append(row["file_hash"])
        file_labels.append(int(row["label"]))
y_file = np.array(file_labels)

# Split train/test
file_train_hashes, file_test_hashes, y_file_train, y_file_test = train_test_split(
    file_hashes, y_file, test_size=0.2, random_state=42, stratify=y_file
)

# Placeholder: hash-based classifier
file_train_probs = np.zeros_like(y_train, dtype=float)
file_test_probs  = np.zeros_like(y_test, dtype=float)

# -----------------------------
# Build meta features
# -----------------------------
meta_X_train = np.column_stack([rf_train, lgb_train, xgb_train, cnn_train, text_train_probs, file_train_probs])
meta_X_test  = np.column_stack([rf_test,  lgb_test,  xgb_test,  cnn_test,  text_test_probs,  file_test_probs])

# -----------------------------
# Train meta-classifier
# -----------------------------
meta_clf = LogisticRegression(max_iter=1000)
meta_clf.fit(meta_X_train, y_train)

# Optional: calibrate probabilities
calibrated_clf = CalibratedClassifierCV(meta_clf, method='sigmoid', cv='prefit')
calibrated_clf.fit(meta_X_train, y_train)
meta_probs = calibrated_clf.predict_proba(meta_X_test)[:,1]

# -----------------------------
# Evaluate meta-classifier
# -----------------------------
y_pred = meta_clf.predict(meta_X_test)
print("\n=== Meta-classifier Metrics ===")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save meta-model and predictions
# -----------------------------
os.makedirs(os.path.dirname(META_MODEL_PATH), exist_ok=True)
joblib.dump({
    "model": meta_clf,
    "features": ["rf", "lgbm", "xgb", "cnn", "text", "file"]
}, META_MODEL_PATH)

np.savez("models/meta_predictions.npz",
         y_test=y_test,
         y_prob=meta_probs,
         rf=rf_test,
         lgb=lgb_test,
         xgb=xgb_test,
         cnn=cnn_test,
         text=text_test_probs,
         file=file_test_probs)

print("\nMeta-model saved to:", META_MODEL_PATH)
print("Per-modality predictions saved to models/meta_predictions.npz")
