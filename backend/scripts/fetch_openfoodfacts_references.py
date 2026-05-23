from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from _catalog_import import (
    CatalogImportStats,
    USER_AGENT,
    download_image_bytes,
    file_name_from_url,
    open_session,
    save_reference_image,
    upsert_product,
)


OPENFOODFACTS_FIELDS = "code,product_name,image_front_url,image_url"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch product names and front images from Open Food Facts by barcode, "
            "then save them as ShelfLens SKU reference images."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV with barcode plus optional sku,name,category columns.",
    )
    parser.add_argument(
        "--barcode",
        action="append",
        default=[],
        help="Barcode to fetch. Can be repeated. SKU defaults to off_<barcode>.",
    )
    parser.add_argument(
        "--country",
        default="world",
        help="Open Food Facts host prefix, for example world, us, tr, or fr.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch metadata but do not write products or images.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.5,
        help="Delay between Open Food Facts requests to avoid rate limits.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for transient 429/503 responses.",
    )
    parser.add_argument(
        "--no-seed-products",
        action="store_true",
        help="Do not seed the default 20 beverage products before importing.",
    )
    return parser


def fetch_product(barcode: str, country: str, retries: int = 3, sleep_seconds: float = 1.5) -> dict:
    host = "world" if country.strip() == "" else country.strip()
    encoded_barcode = quote(barcode.strip())
    url = (
        f"https://{host}.openfoodfacts.org/api/v2/product/{encoded_barcode}.json"
        f"?fields={OPENFOODFACTS_FIELDS}"
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise
            time.sleep(sleep_seconds * (attempt + 1))
    else:
        raise RuntimeError(f"Failed to fetch barcode: {barcode}") from last_error

    if payload.get("status") != 1:
        raise ValueError(f"Barcode not found in Open Food Facts: {barcode}")
    return payload.get("product") or {}


def _rows_from_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "barcode" not in (reader.fieldnames or []):
            raise ValueError("CSV missing required column: barcode")
        return [dict(row) for row in reader]


def _rows_from_barcodes(barcodes: list[str]) -> list[dict[str, str]]:
    return [{"barcode": barcode} for barcode in barcodes]


def fetch_and_import(
    rows: list[dict[str, str]],
    country: str,
    dry_run: bool,
    seed_default_products: bool,
    sleep_seconds: float,
    retries: int,
) -> dict:
    stats = CatalogImportStats()
    fetched: list[dict[str, str | bool | None]] = []
    db = None if dry_run else open_session(seed_default_products=seed_default_products)
    try:
        for row_number, row in enumerate(rows, start=1):
            barcode = (row.get("barcode") or "").strip()
            if not barcode:
                stats.errors.append(f"Row {row_number}: barcode is required.")
                continue

            product_payload: dict = {}
            image_url = (row.get("image_url") or "").strip()
            if not image_url:
                try:
                    product_payload = fetch_product(
                        barcode,
                        country,
                        retries=retries,
                        sleep_seconds=sleep_seconds,
                    )
                except Exception as exc:  # noqa: BLE001 - keep fetching remaining rows.
                    stats.errors.append(f"Row {row_number} ({barcode}): {exc}")
                    continue
                image_url = (
                    product_payload.get("image_front_url")
                    or product_payload.get("image_url")
                    or ""
                )

            sku = (row.get("sku") or f"off_{barcode}").strip()
            name = (
                row.get("name")
                or product_payload.get("product_name")
                or sku
            ).strip()
            category = (row.get("category") or "beverages").strip() or "beverages"
            fetched.append(
                {
                    "barcode": barcode,
                    "sku": sku,
                    "name": name,
                    "category": category,
                    "image_url": image_url,
                    "written": not dry_run,
                }
            )

            if dry_run:
                continue

            assert db is not None
            _, created = upsert_product(db, sku, name, category)
            if created:
                stats.products_created += 1
            else:
                stats.products_updated += 1

            if not image_url:
                stats.errors.append(f"Row {row_number} ({barcode}): no product image URL.")
                continue

            try:
                raw = download_image_bytes(image_url)
                _, saved = save_reference_image(sku, raw, file_name_from_url(image_url))
            except Exception as exc:  # noqa: BLE001 - keep importing remaining rows.
                stats.errors.append(f"Row {row_number} ({barcode}): {exc}")
                continue

            if saved:
                stats.reference_images_saved += 1
            else:
                stats.reference_images_skipped += 1
            time.sleep(sleep_seconds)
    finally:
        if db is not None:
            db.close()

    return {"fetched": fetched, **stats.__dict__}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    if args.csv is not None:
        rows.extend(_rows_from_csv(args.csv))
    rows.extend(_rows_from_barcodes(args.barcode))
    if not rows:
        parser.error("Provide --csv or at least one --barcode.")

    result = fetch_and_import(
        rows,
        country=args.country,
        dry_run=args.dry_run,
        seed_default_products=not args.no_seed_products,
        sleep_seconds=args.sleep_seconds,
        retries=args.retries,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
