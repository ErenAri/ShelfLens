from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import EXPORTS_DIR, MODELS_DIR


METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report detector training metrics against a target.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Model run directory. If omitted, latest data/models run with results.csv is used.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Dataset export directory. If omitted, latest data/exports folder with manifest.json is used.",
    )
    parser.add_argument("--target", type=float, default=0.95, help="Target metric threshold.")
    return parser


def latest_child_with(parent: Path, marker: str) -> Path:
    candidates = sorted(
        [path for path in parent.iterdir() if path.is_dir() and (path / marker).exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No folder with {marker} found under {parent}.")
    return candidates[0]


def load_best_metrics(results_csv: Path) -> dict[str, tuple[float, str]]:
    with results_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No metric rows found in {results_csv}.")

    output: dict[str, tuple[float, str]] = {}
    for key in METRIC_KEYS:
        values: list[tuple[float, str]] = []
        for row in rows:
            raw = (row.get(key) or "").strip()
            if not raw:
                continue
            values.append((float(raw), row.get("epoch", "")))
        if values:
            output[key] = max(values, key=lambda item: item[0])
    return output


def load_manifest(export_dir: Path | None) -> dict:
    resolved = export_dir or latest_child_with(EXPORTS_DIR, "manifest.json")
    return json.loads((resolved / "manifest.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = args.run_dir or latest_child_with(MODELS_DIR, "results.csv")
    results_csv = run_dir / "results.csv"
    manifest = load_manifest(args.export_dir)
    metrics = load_best_metrics(results_csv)

    print(f"Run: {run_dir}")
    print(f"Dataset export: {manifest.get('export_path')}")
    print(f"Detection label mode: {manifest.get('detection_label_mode', 'unknown')}")
    print(
        "Dataset: "
        f"{manifest.get('detection', {}).get('image_count', 0)} images, "
        f"{manifest.get('detection', {}).get('annotation_count', 0)} boxes, "
        f"splits={manifest.get('detection', {}).get('split_counts', {})}"
    )
    print(f"Corrected detections exported: {manifest.get('corrected_detections_exported', 0)}")
    for warning in manifest.get("quality_warnings", []):
        print(f"WARNING: {warning}")

    print("Best metrics:")
    for key in METRIC_KEYS:
        value, epoch = metrics.get(key, (0.0, "n/a"))
        status = "PASS" if value >= args.target else "FAIL"
        print(f"  {key}: {value:.5f} at epoch {epoch} ({status} vs target {args.target:.2f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
