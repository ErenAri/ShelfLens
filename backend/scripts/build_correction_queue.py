from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import sys

from PIL import Image


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import (  # noqa: E402
    CLIP_MODEL_NAME,
    DETECTOR_MODEL_PATH,
    MAX_DETECTIONS,
    MIN_DETECTION_CONFIDENCE,
    MIN_RECOGNITION_MARGIN,
    REFERENCE_DIR,
    ensure_storage_dirs,
)
from app.database import Base, SessionLocal, engine  # noqa: E402
from app.inference import CatalogItem, RealInferenceEngine  # noqa: E402
from app.models import ImageRecord, Product  # noqa: E402


@dataclass
class QueueItem:
    queue_id: str
    image_id: str
    filename: str
    image_path: str
    crop_path: str
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    suggested_sku: str | None
    suggested_product_name: str | None
    detection_confidence: float
    recognition_confidence: float
    confidence: float
    status: str
    review_priority: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run real inference over existing uploads and produce a human "
            "correction queue with crops, CSV, and JSON outputs."
        )
    )
    parser.add_argument("--model", type=str, default=DETECTOR_MODEL_PATH, help="YOLO detector .pt path.")
    parser.add_argument("--clip-model", type=str, default=CLIP_MODEL_NAME, help="SentenceTransformer CLIP model.")
    parser.add_argument("--min-detection-confidence", type=float, default=MIN_DETECTION_CONFIDENCE)
    parser.add_argument("--max-detections", type=int, default=MAX_DETECTIONS)
    parser.add_argument("--min-recognition-margin", type=float, default=MIN_RECOGNITION_MARGIN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to data/correction_queue/<timestamp>.",
    )
    parser.add_argument(
        "--dedupe-image-by-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip duplicate upload files with identical bytes.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max image count for quick smoke tests.")
    return parser


def _reference_images_for_sku(sku: str) -> tuple[str, ...]:
    directory = REFERENCE_DIR / sku
    if not directory.exists():
        return ()
    return tuple(str(path) for path in sorted(directory.glob("*")) if path.is_file())


def _build_catalog() -> list[CatalogItem]:
    with SessionLocal() as db:
        products = db.query(Product).filter(Product.is_active.is_(True)).order_by(Product.sku.asc()).all()
        return [
            CatalogItem(
                sku=product.sku,
                name=product.name,
                reference_images=_reference_images_for_sku(product.sku),
            )
            for product in products
        ]


def _load_images(limit: int | None, dedupe_by_hash: bool) -> list[ImageRecord]:
    seen_hashes: set[str] = set()
    selected: list[ImageRecord] = []
    with SessionLocal() as db:
        query = db.query(ImageRecord).order_by(ImageRecord.created_at.asc())
        for image in query.all():
            path = Path(image.stored_path)
            if not path.exists():
                continue
            if dedupe_by_hash:
                digest = sha256(path.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
            selected.append(image)
            if limit is not None and len(selected) >= limit:
                break
    return selected


def _priority(status: str, confidence: float, recognition_confidence: float) -> str:
    if status == "unknown_product":
        return "high"
    if status == "low_confidence":
        return "high" if recognition_confidence >= 0.70 else "medium"
    if confidence < 0.80:
        return "medium"
    return "low"


def _save_crop(source_path: Path, crop_path: Path, bbox: tuple[int, int, int, int]) -> None:
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path).convert("RGB") as image:
        crop = image.crop(bbox)
        crop.save(crop_path, format="JPEG", quality=92)


def _write_csv(path: Path, items: list[QueueItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(QueueItem.__dataclass_fields__.keys()))
        writer.writeheader()
        for item in items:
            writer.writerow(asdict(item))


def _write_html(path: Path, items: list[QueueItem], output_dir: Path) -> None:
    rows = []
    for item in items:
        crop_relative = Path(item.crop_path).resolve().relative_to(output_dir.resolve()).as_posix()
        rows.append(
            "<tr>"
            f"<td><img src='{crop_relative}' alt='{item.queue_id}'></td>"
            f"<td>{item.queue_id}</td>"
            f"<td>{item.filename}</td>"
            f"<td>{item.suggested_sku or ''}</td>"
            f"<td>{item.suggested_product_name or ''}</td>"
            f"<td>{item.status}</td>"
            f"<td>{item.confidence:.2f}</td>"
            f"<td>{item.recognition_confidence:.2f}</td>"
            f"<td>{item.review_priority}</td>"
            f"<td>{item.bbox_x1},{item.bbox_y1},{item.bbox_x2},{item.bbox_y2}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ShelfLens Correction Queue</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #17201d; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d9e1de; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f8f7; position: sticky; top: 0; }}
    img {{ width: 120px; max-height: 120px; object-fit: contain; background: #f6f6f6; }}
  </style>
</head>
<body>
  <h1>ShelfLens Correction Queue</h1>
  <p>{len(items)} detections need review or confirmation.</p>
  <table>
    <thead>
      <tr>
        <th>Crop</th><th>Queue ID</th><th>Image</th><th>Suggested SKU</th>
        <th>Suggested Product</th><th>Status</th><th>Confidence</th>
        <th>Recognition</th><th>Priority</th><th>BBox</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if not args.model:
        raise SystemExit("Detector model is required. Pass --model or set SHELFLENS_DETECTOR_MODEL_PATH.")

    ensure_storage_dirs()
    Base.metadata.create_all(bind=engine)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (BACKEND_DIR / "data" / "correction_queue" / timestamp)
    crops_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = _build_catalog()
    images = _load_images(args.limit, args.dedupe_image_by_hash)
    inference = RealInferenceEngine(
        clip_model_name=args.clip_model,
        detector_model_path=args.model,
        min_detection_confidence=args.min_detection_confidence,
        max_detections=args.max_detections,
        min_recognition_margin=args.min_recognition_margin,
    )

    items: list[QueueItem] = []
    for image in images:
        source_path = Path(image.stored_path)
        candidates = inference.infer(source_path, image.id, catalog)
        for index, candidate in enumerate(candidates, start=1):
            queue_id = f"{image.id}_{index:03d}"
            crop_path = crops_dir / f"{queue_id}.jpg"
            _save_crop(source_path, crop_path, candidate.bbox)
            items.append(
                QueueItem(
                    queue_id=queue_id,
                    image_id=image.id,
                    filename=image.filename,
                    image_path=str(source_path),
                    crop_path=str(crop_path),
                    bbox_x1=candidate.bbox[0],
                    bbox_y1=candidate.bbox[1],
                    bbox_x2=candidate.bbox[2],
                    bbox_y2=candidate.bbox[3],
                    suggested_sku=candidate.sku,
                    suggested_product_name=candidate.product_name,
                    detection_confidence=round(candidate.detection_confidence, 4),
                    recognition_confidence=round(candidate.recognition_confidence, 4),
                    confidence=round(candidate.confidence, 4),
                    status=candidate.status,
                    review_priority=_priority(
                        candidate.status,
                        candidate.confidence,
                        candidate.recognition_confidence,
                    ),
                )
            )

    items.sort(key=lambda item: (item.review_priority != "high", -item.recognition_confidence, item.filename))
    csv_path = output_dir / "correction_queue.csv"
    json_path = output_dir / "correction_queue.json"
    html_path = output_dir / "index.html"
    _write_csv(csv_path, items)
    json_path.write_text(json.dumps([asdict(item) for item in items], indent=2), encoding="utf-8")
    _write_html(html_path, items, output_dir)

    summary = {
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "html_path": str(html_path),
        "image_count": len(images),
        "queue_item_count": len(items),
        "status_counts": {
            status: sum(1 for item in items if item.status == status)
            for status in sorted({item.status for item in items})
        },
        "priority_counts": {
            priority: sum(1 for item in items if item.review_priority == priority)
            for priority in ["high", "medium", "low"]
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
