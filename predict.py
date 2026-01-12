import sys
import os
import joblib
import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# 1. Load meta-model
# -----------------------------
meta = joblib.load("models/meta_model_aligned_multimodal.joblib")
print("✅ Meta-model loaded successfully.")
print("Expected features:", meta["features"])  # ['rf','lgbm','xgb','cnn','text','file']

# -----------------------------
# 2. Load base models
# -----------------------------
rf = joblib.load("models/random_forest.joblib")
lgbm = joblib.load("models/lightgbm.joblib")
xgb = joblib.load("models/xgboost.joblib")
text_clf = joblib.load("models/text_classifier.joblib")
file_clf = joblib.load("models/file_classifier.joblib")

# CNN (PyTorch)
class CharCNN(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=32, num_classes=2):
        super(CharCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, 64, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64*30, 128)  # Adjust if sequence length differs
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embed(x).permute(0,2,1)
        x = self.pool(torch.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

cnn_model = CharCNN()
cnn_model.load_state_dict(torch.load("models/url_cnn_baseline.pth"))
cnn_model.eval()

# -----------------------------
# 3. Feature extractors (same logic as training)
# -----------------------------
def extract_url_features(url: str):
    # simple example: length and count of "http"
    return [len(url), url.count("http")]

def extract_text_features(texts):
    return text_clf.named_steps['tfidf'].transform(texts)

def extract_file_features(fpath):
    # very basic: file size and extension length
    try:
        size = os.path.getsize(fpath)
    except:
        size = 0
    ext_len = len(os.path.splitext(fpath)[-1])
    return [size, ext_len]

def extract_cnn_features(url: str):
    seq = [ord(c) % 128 for c in url][:64]  # truncate/pad length 64
    seq += [0]*(64-len(seq))
    x = torch.tensor([seq], dtype=torch.long)
    with torch.no_grad():
        logits = cnn_model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs[1]  # probability of phishing

# -----------------------------
# 4. Input handling
# -----------------------------
if len(sys.argv) < 3:
    print("Usage:")
    print("  python predict.py url <URL>")
    print("  python predict.py text \"<some text>\"")
    print("  python predict.py file <path-to-file>")
    sys.exit(1)

mode = sys.argv[1]
value = sys.argv[2]

rf_pred = lgbm_pred = xgb_pred = cnn_pred = text_pred = file_pred = 0.5

if mode == "url":
    feats = extract_url_features(value)
    rf_pred = rf.predict_proba([feats])[0][1]
    lgbm_pred = lgbm.predict_proba([feats])[0][1]
    xgb_pred = xgb.predict_proba([feats])[0][1]
    cnn_pred = extract_cnn_features(value)

elif mode == "text":
    X_text = extract_text_features([value])
    text_pred = text_clf.predict_proba(X_text)[0][1]

elif mode == "file":
    feats = extract_file_features(value)
    file_pred = file_clf.predict_proba([feats])[0][1]

else:
    print("Unknown mode:", mode)
    sys.exit(1)

# -----------------------------
# 5. Stack into final vector
# -----------------------------
features_vector = [rf_pred, lgbm_pred, xgb_pred, cnn_pred, text_pred, file_pred]

pred_class = meta["model"].predict([features_vector])[0]
pred_proba = meta["model"].predict_proba([features_vector])[0]

print("\n=== Final Prediction ===")
print("Predicted class:", pred_class)
print("Class probabilities:", pred_proba)
