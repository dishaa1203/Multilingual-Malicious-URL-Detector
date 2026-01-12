import joblib
import torch
import numpy as np
from modules.url_cnn import URLCNN, URLDataset
from torch.utils.data import DataLoader

from train_model import extract_features, urls, labels  # reuse your feature function
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
# -------------------------------
# Load classical models
# -------------------------------
import csv

urls, labels = [], []
with open("data/dataset.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        urls.append(row["url"])
        labels.append(int(row["label"]))
y_test = np.array(labels)

rf = joblib.load("models/url_rf_baseline.joblib")["model"]
lgbm = joblib.load("models/url_lgbm_baseline.joblib")["model"]
xgb = joblib.load("models/url_xgb_baseline.joblib")["model"]

X = np.array([extract_features(u) for u in urls], dtype=float)

rf_probs = rf.predict_proba(X)[:, 1]  # phishing probability
lgb_probs = lgbm.predict_proba(X)[:, 1]
xgb_probs = xgb.predict_proba(X)[:, 1]

# -------------------------------
# Load CNN model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = URLCNN().to(device)
cnn_model.load_state_dict(torch.load("models/url_cnn_baseline.pth"))
cnn_model.eval()

dataset = URLDataset(urls, labels)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

cnn_probs = []
with torch.no_grad():
    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        y_pred = cnn_model(x_batch).squeeze()
        cnn_probs.extend(y_pred.cpu().numpy())

cnn_probs = np.array(cnn_probs)
# -------------------------------
# Stack predictions for meta-classifier
# -------------------------------
meta_X = np.column_stack([rf_probs, lgb_probs, xgb_probs, cnn_probs])
meta_y = y_test  # same labels as before

# Split for training/testing meta-classifier
from sklearn.model_selection import train_test_split
X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
    meta_X, meta_y, test_size=0.2, random_state=42, stratify=meta_y
)

# -------------------------------
# Train meta-classifier
# -------------------------------
meta_clf = LogisticRegression()
meta_clf.fit(X_meta_train, y_meta_train)

# -------------------------------
# Evaluate
# -------------------------------
y_meta_pred = meta_clf.predict(X_meta_test)
y_meta_prob = meta_clf.predict_proba(X_meta_test)[:, 1]

print("=== Meta-classifier Metrics ===")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

metrics = {
    "accuracy": accuracy_score(y_meta_test, y_meta_pred),
    "precision": precision_score(y_meta_test, y_meta_pred),
    "recall": recall_score(y_meta_test, y_meta_pred),
    "f1": f1_score(y_meta_test, y_meta_pred),
    "roc_auc": roc_auc_score(y_meta_test, y_meta_prob)
}

for k, v in metrics.items():
    print(f"{k:>10}: {v:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_meta_test, y_meta_pred, digits=4))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_meta_test, y_meta_pred))

# -------------------------------
# Save meta-classifier
# -------------------------------
joblib.dump(meta_clf, "models/url_meta_classifier.joblib")
print("\nMeta-classifier saved to: models/url_meta_classifier.joblib")
