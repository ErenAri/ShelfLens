from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _catalog_import import (
    CatalogImportStats,
    download_image_bytes,
    file_name_from_url,
    open_session,
    read_local_image_bytes,
    save_reference_image,
    upsert_product,
)


TEMPLATE = """sku,name,category,image_path,image_url
bev_001,Coca-Cola 330ml,beverages,C:\\path\\to\\coke_front.jpg,
bev_002,Pepsi 330ml,beverages,,https://example.com/pepsi_front.jpg
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import ShelfLens products and reference images from a CSV file. "
            "Use this for your own SKU photos or product image URLs."
        )
    )
    parser.add_argument("--csv", type=Path, help="CSV with sku,name,category,image_path,image_url columns.")
    parser.add_argument(
        "--write-template",
        type=Path,
        default=None,
        help="Write a starter CSV template to this path and exit.",
    )
    parser.add_argument(
        "--no-seed-products",
        action="store_true",
        help="Do not seed the default 20 beverage products before importing.",
    )
    return parser


def _resolve_local_path(csv_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (csv_path.parent / candidate).resolve()


def import_csv(csv_path: Path, seed_default_products: bool) -> CatalogImportStats:
    stats = CatalogImportStats()
    db = open_session(seed_default_products=seed_default_products)
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"sku", "name"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

            for row_number, row in enumerate(reader, start=2):
                sku = (row.get("sku") or "").strip()
                name = (row.get("name") or "").strip()
                category = (row.get("category") or "beverages").strip() or "beverages"
                image_path = (row.get("image_path") or "").strip()
                image_url = (row.get("image_url") or "").strip()
                if not sku or not name:
                    stats.errors.append(f"Row {row_number}: sku and name are required.")
                    continue

                _, created = upsert_product(db, sku, name, category)
                if created:
                    stats.products_created += 1
                else:
                    stats.products_updated += 1

                sources: list[tuple[str, str]] = []
                if image_path:
                    sources.append(("path", image_path))
                if image_url:
                    sources.append(("url", image_url))

                for source_type, source_value in sources:
                    try:
                        if source_type == "path":
                            resolved_path = _resolve_local_path(csv_path, source_value)
                            raw = read_local_image_bytes(resolved_path)
                            filename_hint = resolved_path.name
                        else:
                            raw = download_image_bytes(source_value)
                            filename_hint = file_name_from_url(source_value)
                        _, saved = save_reference_image(sku, raw, filename_hint)
                    except Exception as exc:  # noqa: BLE001 - keep importing remaining rows.
                        stats.errors.append(f"Row {row_number} ({sku}): {exc}")
                        continue

                    if saved:
                        stats.reference_images_saved += 1
                    else:
                        stats.reference_images_skipped += 1
    finally:
        db.close()
    return stats


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.write_template is not None:
        args.write_template.parent.mkdir(parents=True, exist_ok=True)
        args.write_template.write_text(TEMPLATE, encoding="utf-8")
        print(json.dumps({"template_path": str(args.write_template)}, indent=2))
        return 0

    if args.csv is None:
        parser.error("--csv is required unless --write-template is used.")

    stats = import_csv(args.csv, seed_default_products=not args.no_seed_products)
    print(json.dumps(stats.__dict__, indent=2))
    return 0 if not stats.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
