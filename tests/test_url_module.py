import os
import joblib
from modules import url_module

# Load the trained model once at the top
data = joblib.load("models/url_rf_baseline.joblib")
MODEL = data["model"]  # Extract the actual classifier


def test_url_model_exists():
    """Check if the trained model file exists"""
    assert os.path.exists("models/url_rf_baseline.joblib")


def test_predict_phishing_url():
    """Check if a known phishing-looking URL is flagged"""
    test_url = "http://free-login-update.security-paypal.com"
    confidence, label = url_module.predict_url(test_url, MODEL)
    
    # Check if the URL is predicted as phishing
    assert label[0] == "phishing"


def test_predict_safe_url():
    """Check if a known safe URL is correctly identified"""
    test_url = "https://www.google.com"
    confidence, label = url_module.predict_url(test_url, MODEL)
    
    # Check if the URL is predicted as safe
    assert label[0] == "safe"

