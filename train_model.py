import csv
import os
import sys
import math
import re
import random
import json
import lightgbm as lgb
import xgboost as xgb
import torch
from torch.utils.data import DataLoader
from modules.url_cnn import URLCNN, URLDataset
from urllib.parse import urlparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import joblib
csv.field_size_limit(sys.maxsize)
# -----------------------------
# Config / Paths
# -----------------------------
DATASET_CSV_PATH = os.path.join("data", "dataset.csv")
DATASET_JSONL_PATH = os.path.join("data", "merged.jsonl")
RF_MODEL_PATH = os.path.join("models", "url_rf_baseline.joblib")
LGB_MODEL_PATH = os.path.join("models", "url_lgbm_baseline.joblib")
XGB_MODEL_PATH = os.path.join("models", "url_xgb_baseline.joblib")
CNN_MODEL_PATH = os.path.join("models", "url_cnn_baseline.pth")
RANDOM_STATE = 42
TEST_SIZE = 0.2
EPOCHS = 5
BATCH_SIZE = 32

# -----------------------------
# Feature extraction functions
# -----------------------------
SPECIAL_TOKENS = ["@", "-", "_", "?", "=", ".", "#", "%", "&"]
SUSPICIOUS_WORDS = [
    "login", "verify", "account", "update", "secure", "bank", "confirm",
    "password", "ebayisapi", "webscr", "signin", "pay", "credential"
]

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def count_digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)

def count_letters(s: str) -> int:
    return sum(ch.isalpha() for ch in s)

def has_ip_address(netloc: str) -> bool:
    return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", netloc))

def extract_features(url: str):
    parsed = urlparse(url)
    scheme, netloc, path, query = parsed.scheme, parsed.netloc, parsed.path, parsed.query
    full = url.strip()

    url_len, host_len, path_len, query_len = len(full), len(netloc), len(path), len(query)
    num_dots, num_numbers, num_letters_ = full.count("."), count_digits(full), count_letters(full)
    digits_ratio = (num_numbers / url_len) if url_len > 0 else 0
    specials = [full.count(tok) for tok in SPECIAL_TOKENS]
    words_count = sum(1 for w in SUSPICIOUS_WORDS if w in full.lower())
    ent = shannon_entropy(full)
    subdomain_count = netloc.count(".")
    has_https = 1 if scheme.lower() == "https" else 0
    double_slash_in_path = full.count("//") - (1 if "://" in full else 0)
    has_ip = 1 if has_ip_address(netloc) else 0

    return [
        url_len, host_len, path_len, query_len, num_dots,
        num_numbers, num_letters_, digits_ratio, *specials,
        words_count, ent, subdomain_count, has_https,
        double_slash_in_path, has_ip
    ]

FEATURE_NAMES = [
    "url_len", "host_len", "path_len", "query_len",
    "num_dots", "num_numbers", "num_letters", "digits_ratio"
] + [f"tok_count_{tok}" for tok in SPECIAL_TOKENS] + [
    "suspicious_words_count", "shannon_entropy",
    "subdomain_count", "has_https", "double_slash_in_path", "has_ip"
]

# -----------------------------
# Load dataset for tabular models (RF/LGB/XGB)
# -----------------------------
if not os.path.exists(DATASET_CSV_PATH):
    print(f"Error: {DATASET_CSV_PATH} not found. Run prepare_dataset.py first.")
    sys.exit(1)

urls, labels, skipped = [], [], 0
with open(DATASET_CSV_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        url = (row.get("url") or "").strip()
        lbl = row.get("label")
        if not url or lbl is None or lbl == "":
            skipped += 1
            continue
        try:
            lbl = int(lbl)
        except ValueError:
            skipped += 1
            continue
        if lbl not in (0, 1):
            skipped += 1
            continue
        urls.append(url)
        labels.append(lbl)

print(f"Loaded {len(urls)} valid rows. Skipped {skipped} malformed rows.")

X = np.array([extract_features(u) for u in urls], dtype=float)
y = np.array(labels, dtype=int)
X, y = shuffle(X, y, random_state=RANDOM_STATE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# -----------------------------
# Train RandomForest
# -----------------------------
print("\nTraining RandomForest...")
rf_clf = RandomForestClassifier(
    n_estimators=300, max_depth=None, n_jobs=-1,
    random_state=RANDOM_STATE, class_weight="balanced"
)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
print("\n=== RandomForest Metrics ===")
print(classification_report(y_test, y_pred, digits=4))
joblib.dump({"model": rf_clf, "features": FEATURE_NAMES}, RF_MODEL_PATH)

# -----------------------------
# Train LightGBM
# -----------------------------
print("\nTraining LightGBM...")
lgbm_clf = lgb.LGBMClassifier(
    n_estimators=300, max_depth=-1,
    random_state=RANDOM_STATE, class_weight="balanced"
)
lgbm_clf.fit(X_train, y_train)
y_pred_lgb = lgbm_clf.predict(X_test)
print("\n=== LightGBM Metrics ===")
print(classification_report(y_test, y_pred_lgb, digits=4))
joblib.dump({"model": lgbm_clf, "features": FEATURE_NAMES}, LGB_MODEL_PATH)

# -----------------------------
# Train XGBoost
# -----------------------------
print("\nTraining XGBoost...")
xgb_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=6,
    learning_rate=0.1, use_label_encoder=False,
    eval_metric="logloss", random_state=RANDOM_STATE
)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
print("\n=== XGBoost Metrics ===")
print(classification_report(y_test, y_pred_xgb, digits=4))
joblib.dump({"model": xgb_clf, "features": FEATURE_NAMES}, XGB_MODEL_PATH)

# -----------------------------
# Train character-level CNN on GPU if available
# -----------------------------
print("\nTraining Character-level CNN...")
if not os.path.exists(DATASET_JSONL_PATH):
    print(f"Error: {DATASET_JSONL_PATH} not found. CNN cannot train.")
    sys.exit(1)

# Training Character-level CNN
import json

data_jsonl = []
with open("data/merged.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # skip empty lines
        try:
            obj = json.loads(line)
            data_jsonl.append(obj)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid line: {line[:50]} ...")
            continue

print(f"Loaded {len(data_jsonl)} valid JSON objects")

url_data_jsonl = [d for d in data_jsonl if d["modality"] == "url"]
urls_cnn = [d["url"] for d in url_data_jsonl]
labels_cnn = [d["label"] for d in url_data_jsonl]

dataset_cnn = URLDataset(urls_cnn, labels_cnn)
loader = DataLoader(dataset_cnn, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cnn = URLCNN().to(device)
optimizer = torch.optim.Adam(model_cnn.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

for epoch in range(EPOCHS):
    model_cnn.train()
    running_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model_cnn(x_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} loss: {running_loss/len(loader):.4f}")

os.makedirs(os.path.dirname(CNN_MODEL_PATH), exist_ok=True)
torch.save(model_cnn.state_dict(), CNN_MODEL_PATH)
print(f"Character-level CNN model saved to: {CNN_MODEL_PATH}")
