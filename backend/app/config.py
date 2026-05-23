from __future__ import annotations

import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("SHELFLENS_DATA_DIR", BACKEND_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
ANNOTATED_DIR = DATA_DIR / "annotated"
REFERENCE_DIR = DATA_DIR / "references"
EXPORTS_DIR = DATA_DIR / "exports"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = DATA_DIR / "models"
DATABASE_URL = os.getenv(
    "SHELFLENS_DB_URL",
    f"sqlite:///{(DATA_DIR / 'shelflens.db').as_posix()}",
)
INFERENCE_MODE = os.getenv("SHELFLENS_INFERENCE_MODE", "mock").strip().lower()
CLIP_MODEL_NAME = os.getenv("SHELFLENS_CLIP_MODEL", "sentence-transformers/clip-ViT-B-32")
DETECTOR_MODEL_PATH = os.getenv("SHELFLENS_DETECTOR_MODEL_PATH", "").strip() or None
DEFAULT_MIN_DETECTION_CONFIDENCE = "0.05" if DETECTOR_MODEL_PATH else "0.35"
DEFAULT_MAX_DETECTIONS = "80" if DETECTOR_MODEL_PATH else "10"
MIN_DETECTION_CONFIDENCE = float(
    os.getenv("SHELFLENS_MIN_DETECTION_CONFIDENCE", DEFAULT_MIN_DETECTION_CONFIDENCE)
)
MAX_DETECTIONS = int(os.getenv("SHELFLENS_MAX_DETECTIONS", DEFAULT_MAX_DETECTIONS))
DEFAULT_MIN_RECOGNITION_MARGIN = "0.04" if DETECTOR_MODEL_PATH else "0.0"
MIN_RECOGNITION_MARGIN = float(
    os.getenv("SHELFLENS_MIN_RECOGNITION_MARGIN", DEFAULT_MIN_RECOGNITION_MARGIN)
)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ensure_storage_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
