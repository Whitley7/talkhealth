#import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#BASE_DIR = os.path.dirname(__file__)

# Load hate model
#hate_model_path = os.path.join(BASE_DIR, "saved_models", "hate_model")
hate_model_path = "Whitley7/distilroberta-hate"
hate_tokenizer = AutoTokenizer.from_pretrained(hate_model_path)
hate_model = AutoModelForSequenceClassification.from_pretrained(hate_model_path).to(device)
hate_model.eval()

# Load irony model
#irony_model_path = os.path.join(BASE_DIR, "saved_models", "irony_model")
irony_model_path = "Whitley7/distilbert-sarcasm-detection"
irony_tokenizer = AutoTokenizer.from_pretrained(irony_model_path)
irony_model = AutoModelForSequenceClassification.from_pretrained(irony_model_path).to(device)
irony_model.eval()


# Custom thresholds for each hate label (determined by validation)
hate_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
hate_thresholds = [0.65, 0.45, 0.72, 0.5, 0.46, 0.5]

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()

    return text

def predict(text: str):
    text = preprocess(text)

    # Hate prediction (multi-label)
    hate_inputs = hate_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    hate_inputs = {k: v.to(device) for k, v in hate_inputs.items()}

    with torch.no_grad():
        hate_outputs = hate_model(**hate_inputs)
    hate_logits = hate_outputs.logits
    hate_probs = torch.sigmoid(hate_logits).cpu().numpy()[0]  # (num_labels,)

    hate_detected = []
    for idx, (prob, threshold) in enumerate(zip(hate_probs, hate_thresholds)):
        if prob > threshold:
            hate_detected.append(hate_labels[idx])

    # Irony prediction (binary)
    irony_inputs = irony_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    irony_inputs = {k: v.to(device) for k, v in irony_inputs.items()}

    with torch.no_grad():
        irony_outputs = irony_model(**irony_inputs)
    irony_logits = irony_outputs.logits
    irony_probs = torch.sigmoid(irony_logits) if irony_logits.shape[-1] == 1 else torch.softmax(irony_logits, dim=-1)
    irony_score = irony_probs[0][0].item() if irony_probs.shape[-1] == 1 else irony_probs[0][1].item()

    return {             
        "hates": [],
        "is_hate": bool(hate_detected),
        "is_ironic": irony_score > 0.4,
        "irony_score": round(irony_score, 4),
        "hate_scores": {label: round(float(prob), 4) for label, prob in zip(hate_labels, hate_probs)},
    }
