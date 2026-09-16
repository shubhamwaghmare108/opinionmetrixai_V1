from pathlib import Path
from typing import Iterable

import gdown

BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "models"
TRANSFORMER_DIR = MODEL_ROOT / "transformer"
FOLDER_URL = "https://drive.google.com/drive/folders/1rAonN5_zXpeWWmPDwZzZXnjTtcldlMFr?usp=drive_link"

# Files required by Hugging Face AutoTokenizer/AutoModel plus the app's label mapping.
REQUIRED_FILES = (
    "config.json",
    "label_encoder.pkl",
)
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.txt")


def _has_any(directory: Path, names: Iterable[str]) -> bool:
    return any((directory / name).is_file() for name in names)


def _candidate_model_dirs() -> list[Path]:
    """Return possible model directories created by different gdown layouts."""
    candidates = [TRANSFORMER_DIR, MODEL_ROOT]
    if MODEL_ROOT.is_dir():
        candidates.extend(path for path in MODEL_ROOT.iterdir() if path.is_dir())
    return list(dict.fromkeys(candidates))


def find_model_dir() -> Path | None:
    """Find a complete transformer directory, regardless of download nesting."""
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
    """Return True only when all required model, tokenizer, and label files exist."""
    return find_model_dir() is not None


def download_models() -> Path:
    """Download model artifacts and return the validated model directory."""
    existing = find_model_dir()
    if existing is not None:
        return existing

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(
        url=FOLDER_URL,
        output=str(MODEL_ROOT),
        quiet=False,
        use_cookies=False,
    )

    downloaded = find_model_dir()
    if downloaded is None:
        expected = ", ".join(REQUIRED_FILES)
        raise FileNotFoundError(
            "Model download completed, but the downloaded folder is incomplete. "
            f"Expected {expected}, one model-weight file ({', '.join(WEIGHT_FILES)}), "
            f"and one tokenizer file ({', '.join(TOKENIZER_FILES)})."
        )
    return downloaded
