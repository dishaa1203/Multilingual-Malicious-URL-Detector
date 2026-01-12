import joblib
from urllib.parse import urlparse
import math
import re

# Special patterns and suspicious words
SPECIAL_TOKENS = ["@", "-", "_", "?", "=", ".", "#", "%", "&"]
SUSPICIOUS_WORDS = [
    "login", "verify", "account", "update", "secure", "bank", "confirm",
    "password", "ebayisapi", "webscr", "signin", "pay", "credential"
]

WHITELIST = ["www.google.com", "google.com", "youtube.com"]

# ---- Utilities ----
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

def clean_url(url: str) -> str:
    if url.startswith("https://"):
        url = url.replace("https://", "http://")
    return url.lower().strip().rstrip("/")

def extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower()

# ---- Feature extraction ----
def extract_features(url: str):
    parsed = urlparse(url)
    scheme, netloc, path, query = parsed.scheme, parsed.netloc, parsed.path, parsed.query
    full = url.strip()

    return [
        len(full), len(netloc), len(path), len(query),
        full.count("."), count_digits(full), count_letters(full),
        (count_digits(full)/len(full)) if len(full) > 0 else 0,
        *[full.count(tok) for tok in SPECIAL_TOKENS],
        sum(1 for w in SUSPICIOUS_WORDS if w in full.lower()),
        shannon_entropy(full),
        netloc.count("."), 1 if scheme.lower() == "https" else 0,
        full.count("//") - (1 if "://" in full else 0),
        1 if has_ip_address(netloc) else 0
    ]

# ---- Prediction ----
MODEL_PATH = "models/url_rf_baseline.joblib"

def predict_url(url: str, model=None):
    """
    Predict whether a URL is phishing or safe.
    
    Args:
        url (str): URL to check
        model: Optional pre-loaded classifier; if None, loads default model
    
    Returns:
        tuple: (confidence, ["phishing"/"safe"])
    """
    if model is None:
        data = joblib.load(MODEL_PATH)
        model = data["model"]

    url = clean_url(url)
    domain = extract_domain(url)
    
    if domain in WHITELIST:
        return 0.0, ["safe"]

    features = [extract_features(url)]
    pred = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][1]
    return confidence, ["phishing" if pred == 1 else "safe"]

# ---- Script execution ----
if __name__ == "__main__":
    test_url = input("Enter a URL to check: ")
    confidence, label = predict_url(test_url)
    print(f"\nURL: {test_url}")
    print(f"Prediction: {label[0].capitalize()}")
    print(f"Confidence: {confidence:.4f}")
