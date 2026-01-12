import math
from urllib.parse import urlparse

# Define special patterns
SPECIAL_TOKENS = ["@", "-", "_", "?", "=", ".", "#", "%", "&"]
SUSPICIOUS_WORDS = [
    "login", "verify", "account", "update", "secure", "bank", "confirm",
    "password", "ebayisapi", "webscr", "signin", "pay", "credential"
]

# Whitelist trusted domains
WHITELIST = ["www.google.com", "google.com", "youtube.com"]

# ---------- Utilities ----------

def shannon_entropy(s: str) -> float:
    if not s:
        return 0
    probs = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def extract_domain(url):
    return urlparse(url).netloc.lower()

def clean_url(url):
    if url.startswith("https://"):
        url = url.replace("https://", "http://")
    return url.lower().strip().rstrip("/")

# Feature extraction for model
def extract_features(url: str):
    parsed = urlparse(url)
    scheme, netloc, path, query = parsed.scheme, parsed.netloc, parsed.path, parsed.query
    full = url.strip()
    return [
        len(full), len(netloc), len(path), len(query),
        full.count("."), sum(ch.isdigit() for ch in full),
        sum(ch.isalpha() for ch in full),
        (sum(ch.isdigit() for ch in full) / len(full)) if len(full) > 0 else 0,
        *[full.count(tok) for tok in SPECIAL_TOKENS],
        sum(1 for w in SUSPICIOUS_WORDS if w in full.lower()),
        shannon_entropy(full),
        netloc.count("."), 1 if scheme.lower() == "https" else 0,
        full.count("//") - (1 if "://" in full else 0),
        0  # IP detection placeholder
    ]

# ---------- Main Prediction Function ----------

def predict_url(url: str, model):
    """
    Predict whether a URL is phishing or safe using the provided model.
    
    Args:
        url (str): The URL to check.
        model: Pre-trained classifier (e.g., scikit-learn model)
    
    Returns:
        tuple: (confidence, [label]) where label is "phishing" or "safe"
    """
    url = clean_url(url)
    domain = extract_domain(url)

    if domain in WHITELIST:
        return 0.0, ["safe"]

    features = [extract_features(url)]
    pred = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][1]

    return confidence, ["phishing" if pred == 1 else "safe"]
