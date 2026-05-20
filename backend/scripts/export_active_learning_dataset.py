from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.active_learning import ActiveLearningExportOptions, export_active_learning_dataset
from app.database import SessionLocal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export active-learning datasets from ShelfLens data.")
    parser.add_argument("--export-name", type=str, default=None, help="Optional export folder name.")
    parser.add_argument(
        "--corrected-only",
        action="store_true",
        help="Export only corrected detections.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence required for export.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument(
        "--no-recognition-crops",
        action="store_true",
        help="Skip recognition crop export.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    ratio_total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_total - 1.0) > 1e-6:
        parser.error("train-ratio + val-ratio + test-ratio must equal 1.0")

    with SessionLocal() as db:
        summary = export_active_learning_dataset(
            db=db,
            options=ActiveLearningExportOptions(
                export_name=args.export_name,
                corrected_only=args.corrected_only,
                min_confidence=args.min_confidence,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                include_recognition_crops=not args.no_recognition_crops,
            ),
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
