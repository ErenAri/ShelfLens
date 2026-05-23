from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from _catalog_import import (
    CatalogImportStats,
    USER_AGENT,
    download_image_bytes,
    file_name_from_url,
    save_reference_image,
)


PRODUCT_CODE_PATTERN = re.compile(r"/images/products/((?:\d{3}/)*\d+?)/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download extra Open Food Facts front-image variants for ShelfLens "
            "reference SKUs. This uses only selected front images, not nutrition "
            "or ingredient panels."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/sources/beverage_references.csv"),
        help="CSV with sku and image_url columns. Optional barcode column is also supported.",
    )
    parser.add_argument(
        "--country",
        default="world",
        help="Open Food Facts host prefix, for example world, us, tr, or fr.",
    )
    parser.add_argument(
        "--max-per-sku",
        type=int,
        default=6,
        help="Maximum front-image variants to keep per SKU.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Delay between product requests to avoid rate limits.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for transient 429/503 responses.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report candidate URLs without writing files.")
    return parser


def _barcode_from_url(image_url: str) -> str | None:
    match = PRODUCT_CODE_PATTERN.search(image_url)
    if not match:
        return None
    return match.group(1).replace("/", "")


def _fetch_product(barcode: str, country: str, retries: int, sleep_seconds: float) -> dict:
    host = country.strip() or "world"
    encoded = quote(barcode.strip())
    url = (
        f"https://{host}.openfoodfacts.net/api/v2/product/{encoded}.json"
        "?fields=code,product_name,selected_images,image_front_url,image_url"
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") != 1:
                raise ValueError(f"Barcode not found in Open Food Facts: {barcode}")
            return payload.get("product") or {}
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise
            time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError(f"Failed to fetch barcode: {barcode}") from last_error


def _front_variant_urls(product: dict) -> list[str]:
    urls: list[str] = []
    primary_url = product.get("image_front_url") or product.get("image_url")
    if isinstance(primary_url, str) and primary_url:
        urls.append(primary_url)

    selected_images = product.get("selected_images") or {}
    front = selected_images.get("front") if isinstance(selected_images, dict) else {}
    display = front.get("display") if isinstance(front, dict) else {}
    if isinstance(display, dict):
        preferred_languages = ["en", "tr", "fr", "de", "es", "it", "pt", "nl", "pl", "ru"]
        for language in preferred_languages:
            value = display.get(language)
            if isinstance(value, str) and value:
                urls.append(value)
        for value in display.values():
            if isinstance(value, str) and value:
                urls.append(value)

    deduped: list[str] = []
    seen = set()
    for url in urls:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def _rows_from_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sku" not in (reader.fieldnames or []):
            raise ValueError("CSV missing required column: sku")
        return [dict(row) for row in reader]


def import_variants(
    csv_path: Path,
    country: str,
    max_per_sku: int,
    sleep_seconds: float,
    retries: int,
    dry_run: bool,
) -> dict:
    stats = CatalogImportStats()
    summary_rows: list[dict[str, object]] = []

    for row_number, row in enumerate(_rows_from_csv(csv_path), start=2):
        sku = (row.get("sku") or "").strip()
        image_url = (row.get("image_url") or "").strip()
        barcode = (row.get("barcode") or "").strip() or _barcode_from_url(image_url)
        if not sku or not barcode:
            stats.errors.append(f"Row {row_number}: sku and barcode/image_url are required.")
            continue

        try:
            product = _fetch_product(barcode, country, retries, sleep_seconds)
            candidates = _front_variant_urls(product)[:max_per_sku]
        except Exception as exc:  # noqa: BLE001 - keep importing remaining rows.
            stats.errors.append(f"Row {row_number} ({sku}): {exc}")
            continue

        saved_count = 0
        skipped_count = 0
        for candidate_url in candidates:
            try:
                if dry_run:
                    skipped_count += 1
                    continue
                raw = download_image_bytes(candidate_url)
                _, saved = save_reference_image(sku, raw, file_name_from_url(candidate_url))
            except Exception as exc:  # noqa: BLE001 - keep importing remaining candidates.
                stats.errors.append(f"Row {row_number} ({sku}): {exc}")
                continue

            if saved:
                saved_count += 1
                stats.reference_images_saved += 1
            else:
                skipped_count += 1
                stats.reference_images_skipped += 1

        summary_rows.append(
            {
                "sku": sku,
                "barcode": barcode,
                "candidate_count": len(candidates),
                "saved_count": saved_count,
                "skipped_count": skipped_count,
                "candidate_urls": candidates,
            }
        )
        time.sleep(sleep_seconds)

    return {"items": summary_rows, **stats.__dict__}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = import_variants(
        csv_path=args.csv,
        country=args.country,
        max_per_sku=args.max_per_sku,
        sleep_seconds=args.sleep_seconds,
        retries=args.retries,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
