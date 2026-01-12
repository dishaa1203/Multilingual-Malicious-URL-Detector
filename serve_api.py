from flask import Flask, request, jsonify, render_template
import joblib
import math
from urllib.parse import urlparse

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "models/url_rf_baseline.joblib"
data = joblib.load(MODEL_PATH)
clf = data["model"]
FEATURE_NAMES = data["features"]

# Special tokens & suspicious words (same as training)
SPECIAL_TOKENS = ["@", "-", "_", "?", "=", ".", "#", "%", "&"]
SUSPICIOUS_WORDS = [
    "login", "verify", "account", "update", "secure", "bank", "confirm",
    "password", "ebayisapi", "webscr", "signin", "pay", "credential"
]

# Feature extraction
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
    import re
    return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", netloc))

def extract_features(url: str):
    parsed = urlparse(url)
    scheme   = parsed.scheme
    netloc   = parsed.netloc
    path     = parsed.path
    query    = parsed.query
    full     = url.strip()

    url_len      = len(full)
    host_len     = len(netloc)
    path_len     = len(path)
    query_len    = len(query)
    num_dots     = full.count(".")
    num_numbers  = count_digits(full)
    num_letters_ = count_letters(full)
    digits_ratio = (num_numbers / url_len) if url_len > 0 else 0
    specials     = [full.count(tok) for tok in SPECIAL_TOKENS]
    words_count  = sum(1 for w in SUSPICIOUS_WORDS if w in full.lower())
    ent          = shannon_entropy(full)
    subdomain_count = netloc.count(".")
    has_https    = 1 if scheme.lower() == "https" else 0
    double_slash_in_path = full.count("//") - (1 if "://" in full else 0)
    has_ip       = 1 if has_ip_address(netloc) else 0

    return [url_len, host_len, path_len, query_len, num_dots, num_numbers,
            num_letters_, digits_ratio, *specials, words_count, ent,
            subdomain_count, has_https, double_slash_in_path, has_ip]

# API endpoint
@app.route("/predict", methods=["GET"])
def predict():
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "Please provide a URL using ?url=<URL>"}), 400
    if not url.startswith("http"):
        url = "http://" + url

    features = [extract_features(url)]
    prediction = clf.predict(features)[0]
    confidence = clf.predict_proba(features)[0][1]
    

    return jsonify({
        "url": url,
        "prediction": "Phishing" if prediction == 1 else "Safe",
        "confidence": round(float(confidence), 4)
    })

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Run the API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
