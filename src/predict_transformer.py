"""
predict_transformer.py
------------------------
Inference module for the fine-tuned Transformer saved by
train_transformer.py. Mirrors the interface of predict.py
(SentimentPredictor) so app.py can offer either model interchangeably.
"""

import os
import joblib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from train_transformer import light_clean

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "transformer")


class TransformerSentimentPredictor:
    def __init__(self, model_dir: str = MODEL_DIR, max_length: int = 128):
        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        self.max_length = max_length

    @torch.no_grad()
    def predict(self, text: str):
        clean = light_clean(text)
        inputs = self.tokenizer(
            clean,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # DistilBERT does not accept token_type_ids, although some tokenizers
        # may return that field. Remove it before forwarding the batch.
        inputs.pop("token_type_ids", None)
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
        proba_dict = {
            cls: float(p) for cls, p in zip(self.label_encoder.classes_, probs)
        }

        return {
            "sentiment": label,
            "confidence": confidence,
            "probabilities": proba_dict,
        }


if __name__ == "__main__":
    predictor = TransformerSentimentPredictor()
    samples = [
        "This product is absolutely fantastic, I love it so much!",
        "Terrible experience, the item broke after one day and support ignored me.",
        "It's okay, does the job but nothing special.",
    ]
    for s in samples:
        result = predictor.predict(s)
        print(f"\nText: {s}")
        print(f"Predicted: {result['sentiment']} (confidence: {result['confidence']:.2%})")
        print(f"Probabilities: {result['probabilities']}")
