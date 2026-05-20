from __future__ import annotations

import argparse
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import EXPORTS_DIR, MODELS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO detector from an exported ShelfLens dataset.")
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=None,
        help="Path to detection dataset.yaml. If omitted, latest export is used.",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO base model.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument(
        "--run-name",
        type=str,
        default="shelflens_mvp",
        help="Run name under models directory.",
    )
    return parser


def resolve_dataset_yaml(explicit_yaml: Path | None) -> Path:
    if explicit_yaml is not None:
        return explicit_yaml

    candidates = sorted(
        EXPORTS_DIR.glob("*/detection/dataset.yaml"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No exported detection dataset found under data/exports.")
    return candidates[0]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_yaml = resolve_dataset_yaml(args.dataset_yaml)
    if not dataset_yaml.exists():
        parser.error(f"Dataset YAML not found: {dataset_yaml}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install ultralytics first (`pip install ultralytics`)."
        ) from exc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(MODELS_DIR),
        name=args.run_name,
    )
    print(f"Training complete. Artifacts saved under: {MODELS_DIR / args.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
