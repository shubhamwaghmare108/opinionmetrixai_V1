from pathlib import Path
import gdown

BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "models"
FOLDER_URL = "https://drive.google.com/drive/folders/1rAonN5_zXpeWWmPDwZzZXnjTtcldlMFr?usp=drive_link"


def models_are_ready() -> bool:
    """Return True only when the expected transformer artifacts exist."""
    required = (
        MODEL_ROOT / "transformer" / "config.json",
        MODEL_ROOT / "transformer" / "label_encoder.pkl",
    )
    return all(path.is_file() for path in required)


def download_models() -> None:
    """Download model artifacts when they are not already available."""
    if models_are_ready():
        print("Models already downloaded.")
        return

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading models to {MODEL_ROOT}...")
    gdown.download_folder(
        url=FOLDER_URL,
        output=str(MODEL_ROOT),
        quiet=False,
        use_cookies=False,
    )

    if not models_are_ready():
        raise FileNotFoundError(
            "Model download completed, but models/transformer does not contain "
            "config.json and label_encoder.pkl. Check the Google Drive folder layout."
        )
