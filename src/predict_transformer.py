"""Inference wrapper for the fine-tuned Transformer sentiment model."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformer_text import light_clean

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "transformer"


class TransformerSentimentPredictor:
    """Load a validated local Transformer model and predict sentiment."""

    def __init__(self, model_dir: str | os.PathLike[str] = MODEL_DIR, max_length: int = 128):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.max_length = max_length
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")
        self._validate_model_dir()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir), local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.label_encoder = joblib.load(self.model_dir / "label_encoder.pkl")

    def _validate_model_dir(self) -> None:
        required = ("config.json", "label_encoder.pkl")
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        weights = any((self.model_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
        tokenizer = any((self.model_dir / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt"))
        if missing or not weights or not tokenizer:
            details = list(missing)
            if not weights:
                details.append("model weights")
            if not tokenizer:
                details.append("tokenizer files")
            raise FileNotFoundError(
                f"Incomplete Transformer model directory: {self.model_dir}. Missing {', '.join(details)}"
            )

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        clean = light_clean(text)
        inputs = self.tokenizer(
            clean,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())
        label = self.label_encoder.inverse_transform([pred_idx])[0]

        return {
            "sentiment": label,
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                cls: float(prob)
                for cls, prob in zip(self.label_encoder.classes_, probs)
            },
        }
