# main.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "mumbly6/sautshieldsmodels"

tokenizer = AutoTokenizer.from_pretrained(mumbly6/sautshieldsmodels)
model = AutoModelForSequenceClassification.from_pretrained(mumbly6/sautshieldsmodels)

# --- Configuration and Global Variables ---
# Define model paths (assuming they are in a 'models' directory relative to main.py)
INTENT_MODEL_PATH = './models/alpha_intent_model'
HARM_MODEL_PATH = './models/alpha_harm_model'
EMOTION_MODEL_PATH = './models/alpha_emotion_model'

BASE_MODEL_NAME = "xlm-roberta-base" # The base model used for tokenization
MAX_LENGTH = 128

app = FastAPI()

# Global dictionaries to store loaded models, tokenizers, and encoders
models = {}
tokenizers = {}
encoders = {}

# Determine device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Utility Function: Language Detection ---
def detect_language(text: str) -> str:
    """Simple language detection: Kiswahili/Sheng vs English"""
    text_lower = text.lower()

    swahili_indicators = [
        'na', 'ya', 'wa', 'za', 'ni', 'ku', 'la', 'ma', 'ki', 'vi',
        'mimi', 'wewe', 'yeye', 'sisi', 'nyinyi', 'wao',
        'hapa', 'huko', 'hivyo', 'hii', 'hilo', 'hili',
        'kwa', 'kwenye', 'kutoka', 'hadi', 'lakini', 'au',
        'jambo', 'asante', 'karibu', 'pole', 'polepole',
        'sheng', 'msee', 'mzito', 'noma', 'poa', 'safi'
    ]

    sw_count = sum(1 for word in swahili_indicators if word in text_lower)

    if sw_count >= 2 or any(word in text_lower for word in ['sheng', 'msee', 'mzito']):
        return 'kiswahili'
    else:
        return 'english'

# --- FastAPI Startup Event Handler ---
@app.on_event("startup")
async def load_models_and_tokenizers():
    print("Loading models, tokenizers, and encoders...")
    global models, tokenizers, encoders, device

    # Load Intent Model
    if os.path.exists(INTENT_MODEL_PATH):
        models['intent'] = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_PATH).to(device)
        tokenizers['intent'] = AutoTokenizer.from_pretrained(INTENT_MODEL_PATH)
        with open(os.path.join(INTENT_MODEL_PATH, 'label_encoders.json'), 'r') as f:
            encoders['intent'] = json.load(f)['classes']
        print(f"Loaded Intent Model from {INTENT_MODEL_PATH}")
    else:
        print(f"[ERROR] Intent model not found at {INTENT_MODEL_PATH}")

    # Load Harm Model
    if os.path.exists(HARM_MODEL_PATH):
        models['harm'] = AutoModelForSequenceClassification.from_pretrained(HARM_MODEL_PATH).to(device)
        tokenizers['harm'] = AutoTokenizer.from_pretrained(HARM_MODEL_PATH)
        with open(os.path.join(HARM_MODEL_PATH, 'label_encoders.json'), 'r') as f:
            encoders['harm'] = json.load(f)['classes']
        print(f"Loaded Harm Model from {HARM_MODEL_PATH}")
    else:
        print(f"[ERROR] Harm model not found at {HARM_MODEL_PATH}")

    # Load Emotion Model
    if os.path.exists(EMOTION_MODEL_PATH):
        models['emotion'] = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH).to(device)
        tokenizers['emotion'] = AutoTokenizer.from_pretrained(EMOTION_MODEL_PATH)
        with open(os.path.join(EMOTION_MODEL_PATH, 'label_encoders.json'), 'r') as f:
            encoders['emotion'] = json.load(f)['classes']
        print(f"Loaded Emotion Model from {EMOTION_MODEL_PATH}")
    else:
        print(f"[ERROR] Emotion model not found at {EMOTION_MODEL_PATH}")
    
    print("All models, tokenizers, and encoders loaded successfully.")

# --- Pydantic Request Model ---
class TextRequest(BaseModel):
    text: str

# --- FastAPI Endpoint: /analyze ---
@app.post("/analyze")
async def analyze_text(request: TextRequest):
    if not all(k in models for k in ['intent', 'harm', 'emotion']):
        return {"error": "Models not loaded. Please ensure startup event completed successfully."}

    input_text = request.text

    # 1. Language Detection
    detected_language = detect_language(input_text)

    # 2. Intent Prediction (Multi-label)
    intent_tokenizer = tokenizers['intent']
    intent_model = models['intent']
    intent_inputs = intent_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH).to(device)
    intent_model.eval()
    with torch.no_grad():
        intent_logits = intent_model(**intent_inputs).logits
    intent_probs = torch.sigmoid(intent_logits).cpu().numpy().flatten()

    # Filter intents by threshold (e.g., 0.5)
    active_intents = []
    for i, prob in enumerate(intent_probs):
        if prob > 0.5:
            active_intents.append(encoders['intent'][i])
    if not active_intents:
        active_intents = ['no_specific_intent']

    # 3. Harm Prediction (Single-label)
    harm_tokenizer = tokenizers['harm']
    harm_model = models['harm']
    harm_inputs = harm_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH).to(device)
    harm_model.eval()
    with torch.no_grad():
        harm_logits = harm_model(**harm_inputs).logits
    harm_pred_id = torch.argmax(harm_logits, dim=-1).item()
    harm_level = encoders['harm'][harm_pred_id]

    # 4. Emotion Prediction (Single-label)
    emotion_tokenizer = tokenizers['emotion']
    emotion_model = models['emotion']
    emotion_inputs = emotion_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH).to(device)
    emotion_model.eval()
    with torch.no_grad():
        emotion_logits = emotion_model(**emotion_inputs).logits
    emotion_pred_id = torch.argmax(emotion_logits, dim=-1).item()
    emotion = encoders['emotion'][emotion_pred_id]

    return {
        "input_text": input_text,
        "detected_language": detected_language,
        "intent": active_intents,
        "harm_level": harm_level,
        "emotion": emotion
    }

# Example of how to run the app using uvicorn (for local development/testing)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
