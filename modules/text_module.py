from transformers import pipeline
from langdetect import detect
import logging

# Mapping languages to sentiment models
MODEL_MAP = {
    'en': "distilbert-base-uncased-finetuned-sst-2-english",  # English
    'hi': "papluca/xlm-roberta-base-finetuned-twitter-sentiment",  # Hindi fallback (XLM-R)
    'ar': "asafaya/bert-base-arabic-sentiment",  # Arabic
}

LABEL_MAP = {
    "distilbert-base-uncased-finetuned-sst-2-english": {
        "POSITIVE": "Positive",
        "NEGATIVE": "Negative"
    },
    "asafaya/bert-base-arabic-sentiment": {
        "POS": "Positive",
        "NEG": "Negative",
        "NEU": "Neutral"
    },
    "papluca/xlm-roberta-base-finetuned-twitter-sentiment": {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
}

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception as e:
        logging.warning(f"Language detection failed: {e}")
        return "en"  # Default to English

def predict_text(text: str):
    lang = detect_language(text)
    model_name = MODEL_MAP.get(lang, "papluca/xlm-roberta-base-finetuned-twitter-sentiment")  # Fallback

    classifier = pipeline("text-classification", model=model_name)
    result = classifier(text[:512])[0]  # Limit input to 512 tokens

    confidence = float(result['score'])
    label_key = result['label']
    label = LABEL_MAP[model_name].get(label_key, label_key)

    return confidence, [f"{label} (lang: {lang})"]
