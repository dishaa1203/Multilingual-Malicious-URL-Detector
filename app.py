from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union, Dict, Any, List
import joblib
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import docx2txt
import PyPDF2
import io
import tempfile
import os
import urllib.parse
import pytesseract
from PIL import Image, ImageEnhance
import requests
from io import BytesIO
import json
import logging
import math
import whois
from datetime import datetime
import urllib.parse
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multilingual Phishing Detector API")

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class AnalysisRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None

# Load models
print("Loading models...")

# Placeholder models for demonstration
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create a URL model with known phishing patterns
URL_MODEL = RandomForestClassifier(n_estimators=100, random_state=42)

# Train on some basic phishing patterns with 28 features
X_url_train = np.array([
    # Phishing pattern (28 features)
    [50, 3, 2, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 5, 2, 3, 2, 1, 0, 30, 5, 15, 3, 1, 1, 2.5, 0, 1],
    # Legitimate pattern (28 features)  
    [20, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 15, 0, 5, 2, 0, 0, 1.2, 0, 0],
    # Phishing pattern (28 features)
    [60, 4, 3, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 6, 3, 4, 3, 2, 1, 35, 8, 20, 4, 1, 1, 3.1, 0, 1],
    # Legitimate pattern (28 features)
    [25, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 20, 2, 5, 2, 0, 0, 1.5, 0, 0]
])
y_url_train = np.array([1, 0, 1, 0])  # 1=phishing, 0=legitimate
URL_MODEL.fit(X_url_train, y_url_train)
print(f"URL model trained with {X_url_train.shape[1]} features")

# Create a text model
TEXT_MODEL = XGBClassifier(random_state=42)

# Train on some basic text patterns (dummy embeddings)
X_text_train = np.random.rand(4, 768)
y_text_train = np.array([1, 0, 1, 0])  # 1=phishing, 0=legitimate
TEXT_MODEL.fit(X_text_train, y_text_train)

# Load mBERT tokenizer and model
print("Loading mBERT model...")
try:
    TOKENIZER = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    BERT_MODEL = AutoModel.from_pretrained("bert-base-multilingual-cased")
    print("mBERT model loaded successfully")
except Exception as e:
    print(f"Error loading mBERT model: {e}")
    # Create placeholder tokenizer and model for demonstration
    TOKENIZER = None
    BERT_MODEL = None

# Utility functions
def get_domain_age(domain):
    """Simple domain age check for demonstration"""
    if any(x in domain for x in [".digital", ".top", ".xyz", ".online"]):
        return 5  # New domain (5 days old)
    elif "google.com" in domain or "amazon.com" in domain:
        return 3650  # Old domain (~10 years)
    else:
        return 100  # Moderately old domain

def is_new_domain(domain, threshold_days=30):
    """Check if domain is newly registered"""
    age = get_domain_age(domain)
    return age < threshold_days if age is not None else False

def is_brand_impersonation(url):
    """Check for brand impersonation in URL"""
    known_brands = ["paypal", "microsoft", "apple", "amazon", "google", 
                   "binance", "coinbase", "pancakeswap", "metamask", "trustwallet",
                   "facebook", "instagram", "twitter", "linkedin", "netflix"]
    domain = urllib.parse.urlparse(url).netloc.lower()
    
    for brand in known_brands:
        if brand in domain and domain != brand + ".com":
            # Check for common impersonation patterns
            if re.search(rf"{brand}[-_]", domain) or re.search(rf"[-_]{brand}", domain):
                return True
            if re.search(rf"{brand}\d+", domain) or re.search(rf"\d+{brand}", domain):
                return True
    return False

def is_suspicious_tld(url):
    """Check for suspicious TLDs"""
    suspicious_tlds = [".top", ".xyz", ".club", ".online", ".digital", ".shop", 
                      ".website", ".tech", ".site", ".space", ".win", ".bid"]
    domain = urllib.parse.urlparse(url).netloc.lower()
    return any(domain.endswith(tld) for tld in suspicious_tlds)

def calculate_entropy(text):
    """Calculate Shannon entropy of a string"""
    if not text:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(text.count(chr(x))) / len(text)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def has_homograph_chars(domain):
    """Check for homograph attack characters"""
    homograph_chars = set("аеіјорѕуΑΒΕΖΗΙΚΜΝΟΡΤΥΧаеорсуᎪΒΕΖΗΙΚΜΝΟΡΤΥΧ")
    return any(char in homograph_chars for char in domain) if domain else False

def extract_url_features(url: str):
    """Extract 28 features from URL for phishing detection"""
    # Ensure URL has a scheme
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        query = parsed.query
    except:
        domain = ""
        path = url
        query = ""
    
    # Get WHOIS-based features first
    domain_age = get_domain_age(url)
    is_new = is_new_domain(url) if domain_age is not None else False
    
    features = [
        len(url),                            # Feature 1: URL length
        url.count("."),                      # Feature 2: Number of dots
        url.count("-"),                      # Feature 3: Number of hyphens
        int(url.startswith("http")),         # Feature 4: Starts with http
        int("@" in url),                     # Feature 5: Contains @
        int("https" in url),                 # Feature 6: Contains https
        int("//" in url),                    # Feature 7: Contains double slash
        int(re.search(r'\d', url) is not None),  # Feature 8: Contains digits
        int("login" in url.lower()),         # Feature 9: Contains login
        int("verify" in url.lower()),        # Feature 10: Contains verify
        int("update" in url.lower()),        # Feature 11: Contains update
        int("free" in url.lower()),          # Feature 12: Contains free
        int("gift" in url.lower()),          # Feature 13: Contains gift
        url.count("/"),                      # Feature 14: Number of slashes
        url.count("?"),                      # Feature 15: Number of question marks
        url.count("="),                      # Feature 16: Number of equals
        url.count("&"),                      # Feature 17: Number of &
        url.count("%"),                      # Feature 18: Number of %
        url.count("#"),                      # Feature 19: Number of #
        len(re.findall(r"[a-zA-Z]", url)),   # Feature 20: Number of letters
        len(re.findall(r"[0-9]", url)),      # Feature 21: Number of digits
        len(re.findall(r"[^\w\s]", url)),    # Feature 22: Number of special characters
        len(domain.split(".")),              # Feature 23: Number of domain segments
        
        # NEW FEATURES (24-28)
        int(is_brand_impersonation(url)),    # Feature 24: Brand impersonation detection
        int(is_suspicious_tld(url)),         # Feature 25: Suspicious TLD detection
        calculate_entropy(domain),           # Feature 26: Entropy analysis
        int(has_homograph_chars(domain)),    # Feature 27: Homograph attack detection
        int(is_new)                          # Feature 28: New domain flag
    ]
    
    # Ensure we have exactly 28 features
    if len(features) != 28:
        logger.warning(f"Expected 28 features, got {len(features)}. Padding with zeros.")
        # Pad with zeros if needed
        features.extend([0] * (28 - len(features)))
    
    return features

def extract_text_from_file(file: UploadFile):
    """Extract text from various file types"""
    content = file.file.read()
    
    if file.content_type == "text/plain":
        try:
            return content.decode("utf-8")
        except:
            return content.decode("latin-1")
    
    elif file.content_type in ["application/pdf", "application/x-pdf"]:
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")
    
    elif file.content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                              "application/msword"]:
        try:
            # Save to temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            text = docx2txt.process(tmp_path)
            os.unlink(tmp_path)
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading Word document: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

def extract_text_from_image(image_bytes):
    """Extract text from image using OCR"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image for better OCR
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # Increase contrast
        
        # Extract text using Tesseract OCR with multiple languages
        text = pytesseract.image_to_string(image, lang='eng+fra')
        
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def get_bert_embeddings(text: str):
    """Generate BERT embeddings for text"""
    if TOKENIZER is None or BERT_MODEL is None:
        # For demo purposes, return embeddings that indicate phishing based on keywords
        phishing_keywords = ["urgent", "update", "login", "verify", "account", "compromised", "free", "gift", "warning"]
        text_lower = text.lower()
        
        # Calculate a score based on phishing keywords
        phishing_score = sum(1 for keyword in phishing_keywords if keyword in text_lower)
        phishing_score = min(phishing_score / len(phishing_keywords), 1.0)  # Normalize to 0-1
        
        # Create embeddings that will lead to phishing detection
        base_embedding = np.random.rand(768)
        if phishing_score > 0.3:  # If text has phishing characteristics
            # Modify embedding to make it more likely to be classified as phishing
            base_embedding[:100] += phishing_score * 0.5
        
        return base_embedding
    
    # Tokenize text
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = BERT_MODEL(**inputs)
    
    # Use mean pooling to get a single vector representation
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    
    return embeddings.numpy()

def get_prediction_with_confidence(model, features):
    """Get prediction with confidence score"""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([features])[0]
        prediction = int(np.argmax(proba))
        confidence = float(np.max(proba))
    else:
        prediction = int(model.predict([features])[0])
        # For models without predict_proba, use a fixed confidence
        confidence = 0.85 if prediction == 1 else 0.75
    
    return prediction, confidence

def is_known_phishing_domain(url):
    """Check against known phishing domain patterns"""
    known_phishing_patterns = [
        r"secure\-?login\-?update", r"login\-?secure\-?verify",
        r"update\-?secure\-?login", r"verify\-?login\-?secure",
        r"urgent\-?update\-?now", r"free\-?gift\-?reward"
    ]
    
    domain = urllib.parse.urlparse(url).netloc.lower()
    
    for pattern in known_phishing_patterns:
        if re.search(pattern, domain):
            logger.info(f"Known phishing pattern: {pattern} in {domain}")
            return True
    
    return False

def is_likely_phishing_url(url):
    """Aggressive phishing URL detection"""
    if not url:
        return False
        
    url_lower = url.lower()
    
    # IMMEDIATE RED FLAGS - These patterns should instantly flag as phishing
    immediate_red_flags = [
        r"secure.*login", r"login.*secure", r"verify.*login", r"login.*verify",
        r"update.*login", r"login.*update", r"account.*verify", r"verify.*account",
        r"urgent.*update", r"update.*now", r"free.*gift", r"gift.*free",
        r"account.*compromised", r"security.*alert", r"alert.*security", 
        r"warning.*account", r"password.*reset", r"banking.*login",
        r"click.*here", r"claim.*now", r"limited.*time"
    ]
    
    for pattern in immediate_red_flags:
        if re.search(pattern, url_lower):
            logger.info(f"IMMEDIATE RED FLAG: {pattern} in {url}")
            return True
    
    # Suspicious TLDs
    if is_suspicious_tld(url):
        # Allow some legitimate exceptions
        legitimate_sites = ["microsoft.net", "github.io", "amazonaws.com"]
        domain = urllib.parse.urlparse(url).netloc.lower()
        if not any(site in domain for site in legitimate_sites):
            logger.info(f"Suspicious TLD in {url}")
            return True
    
    # Multiple suspicious keywords
    suspicious_keywords = ["secure", "login", "verify", "update", "urgent",
                         "account", "bank", "paypal", "amazon", "microsoft"]
    keyword_count = sum(1 for word in suspicious_keywords if word in url_lower)
    if keyword_count >= 2:
        logger.info(f"Multiple suspicious keywords ({keyword_count}) in {url}")
        return True
    
    # Brand impersonation
    if is_brand_impersonation(url):
        logger.info(f"Brand impersonation in {url}")
        return True
        
    return False

import re
import urllib.parse

def is_likely_phishing_text_multilingual(text):
    """Detect phishing content in multilingual text with URL analysis"""
    if not text or len(text.strip()) < 10:
        return False
    
    text_lower = text.lower()
    
    # First, extract and check URLs in the text
    urls = extract_urls_from_text(text)
    for url in urls:
        if is_suspicious_url(url):
            return True
    
    # Multilingual phishing phrases
    phishing_phrases = [
        # English phrases
        "urgent", "immediately", "required", "verify", "account", "security",
        "suspicious activity", "login", "password", "compromised", "limited time",
        "click here", "claim now", "free", "gift", "reward", "warning",
        "action required", "confirm your account", "update your information",
        "locked", "suspended", "terminated", "deletion", "permanent", "restrict",
        "alert", "warning", "important", "security alert", "account alert",
        
        # French phrases
        "urgent", "immédiatement", "requis", "vérifier", "compte", "sécurité",
        "activité suspecte", "connexion", "mot de passe", "compromis", "temps limité",
        "cliquez ici", "réclamez maintenant", "gratuit", "cadeau", "récompense", "avertissement",
        "action requise", "confirmez votre compte", "mettre à jour vos informations",
        "verrouillé", "suspendu", "suppression", "permanent", "restreint",
        
        # Arabic phrases
        "عزيزنا", "عميل", "تم تعليق", "حسابك", "تحقق", "فورا", "فوري", 
        "إهمال", "فقدان", "أموالك", "نشاط", "غير معتاد", "تجنب", 
        "الإغلاق", "الدائم", "جائزة", "هدية", "استرداد", "حسابك", "تأمين",
        
        # Crypto-specific phrases
        "wallet", "connect", "defi", "nft", "airdrop", "token", "swap",
        "portfolio", "transaction", "blockchain", "metamask", "trustwallet"
    ]
    
    # Count phishing indicators
    phishing_indicators = sum(1 for phrase in phishing_phrases if phrase in text_lower)
    
    # Calculate a score based on indicators found
    score = phishing_indicators / len(phishing_phrases)
    
    # Additional checks for different languages
    has_urgency = any(word in text_lower for word in ["urgent", "immediately", "immédiatement", "now", "right away"])
    has_action_request = any(word in text_lower for word in ["login", "connexion", "click", "cliquez", "verify", "vérifier", "connect", "update"])
    has_threat = any(word in text_lower for word in ["locked", "suspended", "deletion", "permanent", "terminated", "compromised"])
    has_reward = any(word in text_lower for word in ["free", "gift", "reward", "gratuit", "cadeau", "récompense", "airdrop"])
    
    # Arabic specific checks
    has_arabic_urgency = any(word in text_lower for word in ["فورا", "فوري", "الآن", "سريع", "عاجل"])
    has_arabic_threat = any(word in text_lower for word in ["إهمال", "فقدان", "إغلاق", "غلق", "مغلق"])
    has_arabic_reward = any(word in text_lower for word in ["جائزة", "هدية", "ربح", "mكافأة"])
    
    # Check for multiple indicators
    if score > 0.12 and (has_urgency or has_action_request or has_threat or has_arabic_urgency or has_arabic_threat):
        return True
    
    # Check all patterns (both general and Arabic)
    patterns = [
        # General patterns
        r"urgent.*update.*required",
        r"mise.*jour.*urgente.*requise",
        r"please.*login.*immediately",
        r"veuillez.*vous.*connecter.*immédiatement",
        r"account.*compromised",
        r"compte.*compromis",
        r"security.*alert",
        r"alerte.*sécurité",
        r"connect.*wallet",
        r"airdrop.*claim",
        r"free.*nft",
        r"account.*locked",
        r"account.*suspended",
        r"permanent.*deletion",
        r"verify.*account.*now",
        r"alert.*security.*reasons",
        
        # Arabic patterns
        r"تم تعليق.*حسابك",
        r"تحقق.*فورا",
        r"إهمال.*فقدان",
        r"جائزة.*iPhone",
        r"استرداد.*حسابك",
        r"تم تعليق.*حساب",
        r"تحقق.*فورًا",
        r"إهمال.*فقدان",
        r"حسابك البنكي.*نشاط",
        r"https?://.*bank.*\.(com|net|org)"
    ]
    
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Check for pig butchering scam patterns (wrong number scam)
    wrong_number_patterns = [
        r"wrong.*number",
        r"yoga.*class",
        r"retreat.*weekend",
        r"since.*connected",
        r"investor.*la",
        r"what.*you.*do",
        r"hey.*this.*is",
        r"my.*mistake",
        r"looking.*forward"
    ]
    
    # If it looks like a wrong number scam opening
    wrong_number_matches = sum(1 for pattern in wrong_number_patterns if re.search(pattern, text_lower))
    if wrong_number_matches >= 3:
        return True
    
    return False

def extract_urls_from_text(text):
    """Extract all URLs from text"""
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*\??[/\w\.-=&%]*'
    urls = re.findall(url_pattern, text)
    return urls

def is_suspicious_url(url):
    """Check if a URL is suspicious"""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check for brand impersonation
        brand_impersonation_patterns = [
            (r"apple", ["apple.com", "icloud.com"]),
            (r"microsoft", ["microsoft.com", "live.com", "outlook.com"]),
            (r"google", ["google.com", "gmail.com"]),
            (r"paypal", ["paypal.com"]),
            (r"amazon", ["amazon.com", "amazon.in"]),
            (r"facebook", ["facebook.com"]),
            (r"instagram", ["instagram.com"]),
            (r"twitter", ["twitter.com", "x.com"]),
            (r"bank", []),  # Generic bank pattern
            (r"icloud", ["apple.com", "icloud.com"]),
        ]
        
        for brand_pattern, legitimate_domains in brand_impersonation_patterns:
            if re.search(brand_pattern, domain, re.IGNORECASE):
                # Check if this is NOT a legitimate domain
                is_legitimate = any(legit_domain in domain for legit_domain in legitimate_domains)
                if not is_legitimate:
                    return True
        
        # Check for suspicious TLDs
        suspicious_tlds = ['.xyz', '.top', '.club', '.online', '.digital', '.tk', '.ml', '.ga', '.cf', '.gq']
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            return True
        
        # Check for suspicious keywords in domain
        suspicious_keywords = [
            'verify', 'verification', 'secure', 'security', 'login', 'account',
            'update', 'confirm', 'validation', 'auth', 'authenticate',
            'support', 'help', 'service', 'alert', 'warning'
        ]
        
        if any(keyword in domain for keyword in suspicious_keywords):
            return True
        
        # Check for IP addresses (often used in phishing)
        ip_pattern = r'\d+\.\d+\.\d+\.\d+'
        if re.search(ip_pattern, domain):
            return True
            
    except Exception as e:
        # If URL parsing fails, it might be malicious
        return True
        
    return False

def calculate_phishing_confidence(text):
    """Calculate confidence score for phishing detection with URL analysis"""
    if not text:
        return 0.5
    
    text_lower = text.lower()
    confidence = 0.5
    
    # Extract and analyze URLs
    urls = extract_urls_from_text(text)
    url_confidence_boost = 0.0
    
    for url in urls:
        if is_suspicious_url(url):
            url_confidence_boost += 0.4  # Significant boost for suspicious URLs
        else:
            url_confidence_boost += 0.1  # Small boost for any URL (could be phishing)
    
    confidence += min(url_confidence_boost, 0.6)  # Cap URL contribution
    
    # All phishing indicators with confidence values
    phishing_indicators = [
        # English indicators
        ("urgent", 0.2), ("immediately", 0.2), ("required", 0.1),
        ("login", 0.15), ("verify", 0.15), ("account", 0.1),
        ("security", 0.1), ("compromised", 0.25), ("free", 0.1),
        ("gift", 0.1), ("warning", 0.15), ("alert", 0.15),
        ("locked", 0.25), ("suspended", 0.25), ("deletion", 0.3),
        ("permanent", 0.2), ("terminated", 0.25), ("restrict", 0.2),
        
        # French indicators
        ("immédiatement", 0.2), ("requis", 0.1), ("connexion", 0.15),
        ("vérifier", 0.15), ("compte", 0.1), ("sécurité", 0.1),
        ("compromis", 0.25), ("gratuit", 0.1), ("cadeau", 0.1),
        ("avertissement", 0.15), ("verrouillé", 0.25), ("suspendu", 0.25),
        
        # Arabic indicators
        ("فورا", 0.2), ("فوري", 0.2), ("الآن", 0.15),
        ("إهمال", 0.25), ("فقدان", 0.3), ("أموالك", 0.3),
        ("جائزة", 0.2), ("هدية", 0.2), ("ربح", 0.15),
        ("تم تعليق", 0.3), ("حسابك", 0.2),
        
        # Crypto indicators
        ("wallet", 0.2), ("connect", 0.2), ("airdrop", 0.25),
        ("nft", 0.15), ("defi", 0.15), ("metamask", 0.25),
        ("trustwallet", 0.25),
        
        # Wrong number scam indicators
        ("wrong number", 0.3), ("yoga class", 0.2), ("investor", 0.25),
        ("retreat", 0.15), ("my mistake", 0.2)
    ]
    
    for indicator, value in phishing_indicators:
        if indicator in text_lower:
            confidence += value
    
    # Additional boost for text that urges clicking links
    if any(word in text_lower for word in ["click", "tap", "visit", "go to", "open"]) and urls:
        confidence += 0.2
    
    # Cap confidence at 0.95
    return min(confidence, 0.95)
def extract_urls(text):
    """Extract URLs from text"""
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.findall(text)

def get_safe_confidence(content):
    """Get confidence score for safe content"""
    # Simple heuristic - longer text gets higher confidence
    length = len(content)
    if length > 500:
        return 0.85
    elif length > 100:
        return 0.75
    else:
        return 0.65

def hybrid_url_detection(url):
    """Multi-layered detection approach for URLs - AGGRESSIVE VERSION"""
    detection_score = 0
    reasons = []
    url_lower = url.lower()
    
    logger.info(f"Analyzing URL: {url}")
    
    # 0. KNOWN PHISHING DOMAIN PATTERNS (Highest priority)
    if is_known_phishing_domain(url):
        detection_score += 1.2  # Override everything else
        reasons.append("known phishing domain pattern")
        logger.info(f"KNOWN PHISHING DOMAIN PATTERN: {url}")
        return True, 0.95, reasons
    
    # 1. HIGH CONFIDENCE PATTERN MATCHING
    high_confidence_patterns = [
        r"secure.*login", r"login.*secure", r"verify.*login", r"login.*verify",
        r"update.*login", r"login.*update", r"account.*verify", r"verify.*account",
        r"urgent.*update", r"update.*now", r"free.*gift", r"gift.*free",
        r"account.*compromised", r"security.*alert", r"alert.*security"
    ]
    
    for pattern in high_confidence_patterns:
        if re.search(pattern, url_lower):
            detection_score += 1.0
            reasons.append(f"high-confidence pattern: {pattern}")
            logger.info(f"HIGH CONFIDENCE PATTERN MATCH: {pattern} in {url}")
            break
    
    # 2. MACHINE LEARNING MODEL PREDICTION
    features = extract_url_features(url)
    ml_prediction, ml_confidence = get_prediction_with_confidence(URL_MODEL, features)
    
    if ml_prediction == 1:
        detection_score += ml_confidence
        reasons.append(f"ML model prediction (confidence: {ml_confidence:.2f})")
        logger.info(f"ML PREDICTION: Phishing with confidence {ml_confidence}")
    
    # 3. BRAND IMPERSONATION DETECTION
    if is_brand_impersonation(url):
        detection_score += 0.8
        reasons.append("brand impersonation detected")
        logger.info(f"BRAND IMPERSONATION DETECTED in {url}")
    
    # 4. SUSPICIOUS TLD DETECTION
    if is_suspicious_tld(url):
        detection_score += 0.6
        reasons.append("suspicious TLD")
        logger.info(f"SUSPICIOUS TLD in {url}")
    
    # 5. DOMAIN AGE CHECK
    domain_age = get_domain_age(url)
    if domain_age is not None and domain_age < 30:  # Less than 30 days old
        detection_score += 0.5
        reasons.append(f"new domain ({domain_age} days old)")
        logger.info(f"NEW DOMAIN: {domain_age} days old")
    
    # 6. ENTROPY ANALYSIS (random-looking domains)
    domain = urllib.parse.urlparse(url).netloc
    entropy = calculate_entropy(domain)
    if entropy > 4.0:  # High entropy suggests random generation
        detection_score += 0.4
        reasons.append(f"high entropy domain (entropy: {entropy:.2f})")
        logger.info(f"HIGH ENTROPY DOMAIN: {entropy:.2f}")
    
    # 7. HOMOGRAPH ATTACK DETECTION
    if has_homograph_chars(domain):
        detection_score += 0.7
        reasons.append("homograph attack characters detected")
        logger.info(f"HOMOGRAPH ATTACK DETECTED in {domain}")
    
    # Determine final result
    is_phishing = detection_score >= 0.8  # Lowered threshold for more aggressive detection
    final_confidence = min(detection_score, 0.95) if is_phishing else max(1 - detection_score, 0.6)
    
    logger.info(f"FINAL DETECTION SCORE: {detection_score}, IS PHISHING: {is_phishing}, CONFIDENCE: {final_confidence}")
    
    return is_phishing, final_confidence, reasons

# API endpoints
@app.post("/analyze")
async def analyze_input(
    url: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None)
):
    logger.info(f"Received request - URL: {url}, Text: {text}, File: {file}, Image: {image}")
    
    try:
        # Determine input type and extract content
        input_type, content, extracted_text = await _process_input(url, text, file, image)
        
        # Process based on input type
        if input_type == "url":
            result = await _analyze_url(content)
        elif input_type == "text":
            result = await _analyze_text(content)
        elif input_type == "file":
            result = await _analyze_file(content)
        elif input_type == "image":
            result = await _analyze_image(content, extracted_text)
        else:
            raise HTTPException(status_code=400, detail="No valid input provided")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")

async def _process_input(url, text, file, image):
    """Process and validate input parameters"""
    if url and url.strip():
        return "url", url.strip(), None
    elif text and text.strip():
        return "text", text.strip(), None
    elif file and file.filename:
        content = extract_text_from_file(file)
        return "file", content, None
    elif image and image.filename:
        image_bytes = await image.read()
        extracted_text = extract_text_from_image(image_bytes)
        return "image", image.filename, extracted_text
    else:
        raise HTTPException(status_code=400, detail="No input provided. Please provide a URL, text, file, or image.")

async def _analyze_url(content):
    """Analyze URL input"""
    logger.info(f"Processing URL: {content}")
    is_phishing, confidence, reasons = hybrid_url_detection(content)
    prediction = 1 if is_phishing else 0
    logger.info(f"URL analysis result - Prediction: {prediction}, Confidence: {confidence}")
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "input_type": "url",
        "detection_reasons": reasons if prediction == 1 else ["No threats detected"]
    }

async def _analyze_text(content):
    """Analyze text input"""
    logger.info(f"Processing text: {content[:100]}...")
    if is_likely_phishing_text_multilingual(content):
        prediction = 1
        confidence = calculate_phishing_confidence(content)
        logger.info(f"Text detected as phishing - Prediction: {prediction}, Confidence: {confidence}")
    else:
        prediction = 0
        confidence = get_safe_confidence(content)
        logger.info(f"Text appears safe - Prediction: {prediction}, Confidence: {confidence}")
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "input_type": "text"
    }

async def _analyze_file(content):
    """Analyze file input"""
    logger.info(f"Processing file content, Length: {len(content)}")
    urls_in_file = extract_urls(content)
    phishing_detected = False
    
    for u in urls_in_file:
        is_phishing, _, _ = hybrid_url_detection(u)
        if is_phishing:
            phishing_detected = True
            prediction, confidence = 1, 0.95
            logger.info(f"File contains phishing URL: {u}")
            break
    
    if not phishing_detected:
        if is_likely_phishing_text_multilingual(content):
            prediction = 1
            confidence = calculate_phishing_confidence(content)
            logger.info(f"File content detected as phishing - Prediction: {prediction}, Confidence: {confidence}")
        else:
            prediction = 0
            confidence = get_safe_confidence(content)
            logger.info(f"File content appears safe - Prediction: {prediction}, Confidence: {confidence}")
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "input_type": "file"
    }

async def _analyze_image(content, extracted_text):
    """Analyze image input"""
    logger.info(f"Processing image: {content}, Extracted text length: {len(extracted_text) if extracted_text else 0}")
    if is_likely_phishing_text_multilingual(extracted_text):
        prediction = 1
        confidence = calculate_phishing_confidence(extracted_text)
        logger.info(f"Image text detected as phishing - Prediction: {prediction}, Confidence: {confidence}")
    else:
        embeddings = get_bert_embeddings(extracted_text)
        prediction, confidence = get_prediction_with_confidence(TEXT_MODEL, embeddings)
        logger.info(f"Image analysis result - Prediction: {prediction}, Confidence: {confidence}")
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "input_type": "image",
        "extracted_text": extracted_text
    }

@app.get("/debug/{url:path}")
async def debug_url(url: str):
    """Debug endpoint to see detection details"""
    logger.info(f"Debug analysis for: {url}")
    
    # Test pattern matching
    url_lower = url.lower()
    patterns = [
        r"urgent.*update", r"update.*now", r"free.*gift", r"gift.*free",
        r"login.*verify", r"verify.*login", r"account.*compromised"
    ]
    
    pattern_matches = []
    for pattern in patterns:
        if re.search(pattern, url_lower):
            pattern_matches.append(pattern)
    
    # Test all detection components
    features = extract_url_features(url)
    ml_prediction, ml_confidence = get_prediction_with_confidence(URL_MODEL, features)
    is_phishing, final_confidence, reasons = hybrid_url_detection(url)
    
    return {
        "url": url,
        "pattern_matches": pattern_matches,
        "ml_prediction": ml_prediction,
        "ml_confidence": ml_confidence,
        "final_prediction": is_phishing,
        "final_confidence": final_confidence,
        "reasons": reasons,
        "suspicious_tld": is_suspicious_tld(url),
        "brand_impersonation": is_brand_impersonation(url),
        "domain_age_days": get_domain_age(url),
        "is_likely_phishing_url": is_likely_phishing_url(url)
    }

@app.post("/analyze_json")
async def analyze_input_json(request: AnalysisRequest):
    """Alternative endpoint that accepts JSON data"""
    logger.info(f"Received JSON request: {request}")
    
    if request.url:
        return await analyze_input(url=request.url)
    elif request.text:
        return await analyze_input(text=request.text)
    else:
        raise HTTPException(status_code=400, detail="No URL or text provided in JSON request")

@app.get("/")
def read_root():
    return {"message": "Multilingual Phishing Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)