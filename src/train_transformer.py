"""Fine-tune a pretrained Transformer for sentiment classification."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from data_preprocessing import load_data
from logging_config import get_logger
from transformer_text import light_clean

logger = get_logger(__name__)
RANDOM_STATE = 42
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "reviews.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "transformer"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "models" / "transformer_checkpoints"
DEFAULT_MODEL = "distilbert-base-uncased"


class ReviewDataset(Dataset):
    """Tokenized reviews plus integer labels for Hugging Face Trainer."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    """Compute weighted validation metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Fine-tune a Transformer for sentiment classification.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_length < 1:
        raise ValueError("--max-length must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    args.data = args.data.resolve()
    args.out_dir = args.out_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    start = time.time()
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("Using device: %s", device)
    logger.info("Loading data from %s", args.data)

    df = load_data(str(args.data))
    df["clean_review"] = df["review"].apply(light_clean)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["sentiment"])
    if len(label_encoder.classes_) < 2:
        raise ValueError("At least two sentiment classes are required.")

    train_text, test_text, train_y, test_y = train_test_split(
        df["clean_review"].tolist(), labels, test_size=0.2,
        random_state=RANDOM_STATE, stratify=labels,
    )
    train_text, val_text, train_y, val_y = train_test_split(
        train_text, train_y, test_size=0.1,
        random_state=RANDOM_STATE, stratify=train_y,
    )

    logger.info("Train: %d | Val: %d | Test: %d", len(train_text), len(val_text), len(test_text))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label_encoder.classes_),
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)},
    )

    def tokenize(texts):
        return tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=args.max_length, return_tensors="pt",
        )

    train_dataset = ReviewDataset(tokenize(train_text), train_y)
    val_dataset = ReviewDataset(tokenize(val_text), val_y)
    test_dataset = ReviewDataset(tokenize(test_text), test_y)

    training_args = TrainingArguments(
        output_dir=str(args.checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=[],
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    result = trainer.predict(test_dataset)
    predictions = np.argmax(result.predictions, axis=1)
    accuracy = accuracy_score(test_y, predictions)
    precision = precision_score(test_y, predictions, average="weighted", zero_division=0)
    recall = recall_score(test_y, predictions, average="weighted", zero_division=0)
    f1 = f1_score(test_y, predictions, average="weighted", zero_division=0)
    matrix = confusion_matrix(test_y, predictions)
    report = classification_report(test_y, predictions, target_names=label_encoder.classes_, zero_division=0)

    report_text = "\n".join([
        f"Model: {args.model}",
        f"Epochs: {args.epochs}",
        f"Max length: {args.max_length}",
        f"Accuracy: {accuracy:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1: {f1:.4f}",
        "Confusion Matrix:",
        str(matrix),
        "Classification Report:",
        report,
    ])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))
    joblib.dump(label_encoder, args.out_dir / "label_encoder.pkl")
    (args.out_dir / "evaluation_report.txt").write_text(report_text, encoding="utf-8")

    logger.info("Saved model artifacts to %s", args.out_dir)
    logger.info("Total time: %.1fs", time.time() - start)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Training run failed with an unhandled exception.")
        raise
