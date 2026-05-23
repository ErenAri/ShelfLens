from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


@dataclass
class ScanItem:
    split: str
    listed_path: str
    resolved_path: str
    issue: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan SKU-110K image lists for unreadable JPEGs and emit *_clean.txt split files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to SKU-110K dataset root (contains train.txt/val.txt/test.txt).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split names to scan.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        required=True,
        help="Output path for corruption scan report JSON.",
    )
    return parser.parse_args()


def _iter_split_lines(path: Path) -> Iterable[str]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            yield line


def _is_decodable_image(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing_file"

    if path.suffix.lower() in {".jpg", ".jpeg"}:
        try:
            data = path.read_bytes()
            if len(data) < 4:
                return False, "too_small"
            if not (data[0] == 0xFF and data[1] == 0xD8):
                return False, "missing_jpeg_soi"
            if not (data[-2] == 0xFF and data[-1] == 0xD9):
                return False, "missing_jpeg_eoi"
        except OSError:
            return False, "unreadable_bytes"

    try:
        with Image.open(path) as im:
            im.verify()
    except Exception:
        return False, "pil_verify_failed"

    try:
        buf = path.read_bytes()
        arr = cv2.imdecode(
            np.frombuffer(buf, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if arr is None:
            return False, "opencv_decode_failed"
    except Exception:
        return False, "opencv_decode_failed"

    return True, "ok"


def main() -> int:
    args = _parse_args()
    dataset_root = args.dataset_root.resolve()
    split_names = [part.strip() for part in args.splits.split(",") if part.strip()]
    bad_items: list[ScanItem] = []
    summary: dict[str, dict[str, int]] = {}

    for split in split_names:
        split_file = dataset_root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        kept: list[str] = []
        total = 0
        bad = 0
        for listed in _iter_split_lines(split_file):
            total += 1
            listed_norm = listed.replace("\\", "/")
            resolved = (dataset_root / listed_norm).resolve()
            ok, issue = _is_decodable_image(resolved)
            if ok:
                kept.append(listed_norm)
                continue
            bad += 1
            bad_items.append(
                ScanItem(
                    split=split,
                    listed_path=listed_norm,
                    resolved_path=str(resolved),
                    issue=issue,
                )
            )

        clean_file = dataset_root / f"{split}_clean.txt"
        clean_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        summary[split] = {
            "total": total,
            "kept": len(kept),
            "bad": bad,
        }

    payload = {
        "dataset_root": str(dataset_root),
        "summary": summary,
        "bad_items": [asdict(item) for item in bad_items],
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))
    print(f"Report saved: {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
