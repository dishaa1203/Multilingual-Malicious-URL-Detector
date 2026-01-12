import os
import numpy as np
import joblib
import torch
from torch.utils.data import DataLoader
from train_model import extract_features
from modules.url_cnn import URLCNN, URLDataset

# ---------------------------
# Paths to saved models
# ---------------------------
META_MODEL_PATH = "models/meta_model_aligned_multimodal.joblib"
CNN_MODEL_PATH = "models/url_cnn_baseline.pth"
URL_RF_PATH = "models/url_rf_baseline.joblib"
URL_LGBM_PATH = "models/url_lgbm_baseline.joblib"
URL_XGB_PATH = "models/url_xgb_baseline.joblib"
TEXT_VECTORIZER_PATH = "models/text_vectorizer.joblib"
TEXT_CLF_PATH = "models/text_clf.joblib"
FILE_TEXT_CLF_PATH = "models/file_text_clf.joblib"
FILE_BINARY_CLF_PATH = "models/file_binary_clf.joblib"

# ---------------------------
# Load models
# ---------------------------
meta_data = joblib.load(META_MODEL_PATH)
meta_model = meta_data["model"]

# CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = URLCNN().to(device)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
cnn_model.eval()

# Classical models (RF, LGB, XGB)
rf_data = joblib.load(URL_RF_PATH)
rf = rf_data["model"]
rf_features = rf_data["features"]

lgb_data = joblib.load(URL_LGBM_PATH)
lgbm = lgb_data["model"]

xgb_data = joblib.load(URL_XGB_PATH)
xgb = xgb_data["model"]

# Text + file models
vectorizer = joblib.load(TEXT_VECTORIZER_PATH)
text_clf = joblib.load(TEXT_CLF_PATH)
file_text_clf = joblib.load(FILE_TEXT_CLF_PATH)
file_binary_clf = joblib.load(FILE_BINARY_CLF_PATH)

# ---------------------------
# Helper functions
# ---------------------------
def is_text_file(fpath, blocksize=512):
    try:
        with open(fpath, "rb") as f:
            block = f.read(blocksize)
        return b"\0" not in block
    except:
        return False

def get_url_probs(url):
    # Extract features and align with training
    features = extract_features(url)
    import pandas as pd
    X = pd.DataFrame([features], columns=rf_features)  # keep feature names aligned


    rf_prob = rf.predict_proba(X)[:, 1][0]
    lgb_prob = lgbm.predict_proba(X)[:, 1][0]
    xgb_prob = xgb.predict_proba(X)[:, 1][0]

    # CNN prediction
    dataset = URLDataset([url], [0])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            y_pred = cnn_model(x_batch).squeeze()
            cnn_prob = float(y_pred.cpu().numpy())

    return rf_prob, lgb_prob, xgb_prob, cnn_prob

def get_text_prob(text):
    X_text = vectorizer.transform([text])
    return text_clf.predict_proba(X_text)[:, 1][0]

def get_file_prob(fpath):
    if fpath is None or not os.path.exists(fpath):
        return 0.5
    if is_text_file(fpath):
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return file_text_clf.predict_proba(vectorizer.transform([content]))[:, 1][0]
    else:
        with open(fpath, "rb") as f:
            content = f.read()
        hist = np.zeros(256)
        for b in content:
            hist[b] += 1
        hist /= len(content)
        return file_binary_clf.predict_proba([hist])[:, 1][0]

# ---------------------------
# Prediction function
# ---------------------------
def predict_sample(url=None, text=None, file_path=None):
    rf_prob, lgb_prob, xgb_prob, cnn_prob = get_url_probs(url) if url else (0.5, 0.5, 0.5, 0.5)
    text_prob = get_text_prob(text) if text else 0.5
    file_prob = get_file_prob(file_path) if file_path else 0.5

    # Meta-model features
    X_meta = np.array([[rf_prob, lgb_prob, xgb_prob, cnn_prob, text_prob, file_prob]])
    raw_pred_prob = meta_model.predict_proba(X_meta)[0]

    # Apply threshold adjustment (reduce false positives)
    pred_class = int(raw_pred_prob[1] > 0.6)

    return pred_class, raw_pred_prob

# ---------------------------
# Direct run example & CLI
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    pred_class, pred_prob = predict_sample(url=args.url, text=args.text, file_path=args.file)
    print("Predicted class:", pred_class)
    print("Class probabilities:", pred_prob)

    # Example hardcoded test if nothing is passed
    if not args.url and not args.text and not args.file:
        url = "http://free-gift-login.com/verify"
        text = "Login to verify your account"
        file_path = "data/files/file1.pdf"
        pred_class, pred_prob = predict_sample(url=url, text=text, file_path=file_path)
        print("\nExample prediction:")
        print("URL:", url)
        print("Text:", text)
        print("File:", file_path)
        print("Predicted class:", pred_class)
        print("Class probabilities:", pred_prob)
