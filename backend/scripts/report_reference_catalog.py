from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import ALLOWED_EXTENSIONS, REFERENCE_DIR, ensure_storage_dirs
from app.database import Base, SessionLocal, engine
from app.models import Product


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report SKU reference-image coverage.")
    parser.add_argument("--target-per-sku", type=int, default=5, help="Desired minimum reference images per SKU.")
    return parser


def _reference_count(sku: str) -> int:
    directory = REFERENCE_DIR / sku
    if not directory.exists():
        return 0
    return len([item for item in directory.iterdir() if item.is_file() and item.suffix.lower() in ALLOWED_EXTENSIONS])


def build_report(target_per_sku: int) -> dict:
    ensure_storage_dirs()
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        products = db.query(Product).order_by(Product.sku.asc()).all()
        items = []
        for product in products:
            count = _reference_count(product.sku)
            items.append(
                {
                    "sku": product.sku,
                    "name": product.name,
                    "reference_image_count": count,
                    "target_per_sku": target_per_sku,
                    "status": "ready" if count >= target_per_sku else "needs_more_references",
                    "missing_count": max(0, target_per_sku - count),
                }
            )

    total_references = sum(item["reference_image_count"] for item in items)
    ready_skus = sum(1 for item in items if item["status"] == "ready")
    return {
        "target_per_sku": target_per_sku,
        "sku_count": len(items),
        "ready_sku_count": ready_skus,
        "total_reference_images": total_references,
        "items": items,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print(json.dumps(build_report(args.target_per_sku), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
