from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment(texts):
    scores = []
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        scores.append(probs.detach().numpy()[0])
    return np.mean(scores, axis=0)
