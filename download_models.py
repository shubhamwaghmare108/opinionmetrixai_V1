"""Download and validate the Transformer model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import gdown

BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "models"
TRANSFORMER_DIR = MODEL_ROOT / "transformer"
FOLDER_URL = "https://drive.google.com/drive/folders/1rAonN5_zXpeWWmPDwZzZXnjTtcldlMFr?usp=drive_link"

REQUIRED_FILES = ("config.json", "label_encoder.pkl")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.txt")


def _has_any(directory: Path, names: Iterable[str]) -> bool:
    return any((directory / name).is_file() for name in names)


def _candidate_model_dirs() -> list[Path]:
    candidates = [TRANSFORMER_DIR, MODEL_ROOT]
    if MODEL_ROOT.is_dir():
        candidates.extend(path for path in MODEL_ROOT.iterdir() if path.is_dir())
    return list(dict.fromkeys(candidates))


def find_model_dir() -> Path | None:
    """Return the first complete model directory, including nested gdown layouts."""
    for directory in _candidate_model_dirs():
        if not directory.is_dir():
            continue
        if not all((directory / name).is_file() for name in REQUIRED_FILES):
            continue
        if not _has_any(directory, WEIGHT_FILES):
            continue
        if not _has_any(directory, TOKENIZER_FILES):
            continue
        return directory
    return None


def models_are_ready() -> bool:
    return find_model_dir() is not None


def download_models() -> Path:
    """Download model artifacts once and return a validated directory."""
    existing = find_model_dir()
    if existing is not None:
        return existing

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download_folder(
            url=FOLDER_URL,
            output=str(MODEL_ROOT),
            quiet=False,
            use_cookies=False,
        )
    except Exception as exc:
        raise RuntimeError("Failed to download the Transformer model artifacts from Google Drive.") from exc

    downloaded = find_model_dir()
    if downloaded is None:
        raise FileNotFoundError(
            "Model download completed, but validation failed. Expected config.json, "
            "label_encoder.pkl, model weights (model.safetensors or pytorch_model.bin), "
            "and tokenizer files."
        )
    return downloaded
