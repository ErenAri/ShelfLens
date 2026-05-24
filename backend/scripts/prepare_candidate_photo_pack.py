from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from PIL import Image


USER_AGENT = "ShelfLensMVP/0.1 (local dataset candidate review)"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class TargetProduct:
    sku: str
    name: str
    queries: tuple[str, ...]


TARGET_PRODUCTS: tuple[TargetProduct, ...] = (
    TargetProduct("bev_coca_cola_original", "Coca-Cola Original", ("Coca-Cola original 330ml", "Coca-Cola classic can")),
    TargetProduct("bev_coca_cola_zero", "Coca-Cola Zero", ("Coca-Cola Zero Sugar 330ml", "Coca-Cola Zero can")),
    TargetProduct("bev_pepsi", "Pepsi", ("Pepsi 330ml can", "Pepsi cola bottle")),
    TargetProduct("bev_pepsi_max", "Pepsi Max", ("Pepsi Max 330ml", "Pepsi Max can")),
    TargetProduct("bev_fanta_orange", "Fanta Orange", ("Fanta Orange 330ml", "Fanta orange bottle")),
    TargetProduct("bev_sprite", "Sprite", ("Sprite 330ml can", "Sprite lemon lime bottle")),
    TargetProduct("bev_lipton_ice_tea", "Lipton Ice Tea", ("Lipton Ice Tea peach 500ml", "Lipton Ice Tea lemon")),
    TargetProduct("bev_fuse_tea", "Fuse Tea", ("Fuse Tea peach", "Fuze Tea peach")),
    TargetProduct("bev_red_bull", "Red Bull", ("Red Bull Energy Drink 250ml", "Red Bull can")),
    TargetProduct("bev_monster", "Monster Energy", ("Monster Energy drink can", "Monster energy original")),
    TargetProduct("bev_cappy", "Cappy Juice", ("Cappy juice", "Cappy orange juice")),
    TargetProduct("bev_dimes", "Dimes Juice", ("Dimes juice", "Dimes fruit juice")),
)


COMMONS_CATEGORIES: tuple[str, ...] = (
    "Category:Soft drinks aisles in supermarkets",
    "Category:Retail displays of soft drinks",
)


COMMONS_TITLE_ALLOWLIST: tuple[str, ...] = (
    "soft drink",
    "softdrink",
    "soda",
    "coca-cola",
    "energy drinks",
    "supermarket",
    "mattoni",
    "ifri",
)


def request_json(url: str, retries: int = 2, sleep_seconds: float = 1.0) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError("request failed") from last_error


def download_bytes(url: str, retries: int = 2, sleep_seconds: float = 1.0) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Referer": "https://commons.wikimedia.org/",
                },
            )
            with urlopen(request, timeout=45) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError("download failed") from last_error


def slugify(value: str, fallback: str = "image") -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered or fallback


def valid_image(path: Path) -> tuple[bool, tuple[int, int] | None]:
    try:
        with Image.open(path) as image:
            size = image.size
            image.verify()
        return True, size
    except Exception:
        return False, None


def save_image(raw: bytes, path: Path) -> tuple[bool, tuple[int, int] | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    ok, size = valid_image(path)
    if not ok:
        path.unlink(missing_ok=True)
    return ok, size


def openfoodfacts_candidates(
    target: TargetProduct,
    per_sku: int,
    errors: list[str] | None = None,
) -> list[dict]:
    candidates: list[dict] = []
    seen_urls: set[str] = set()
    for query in target.queries:
        params = urlencode(
            {
                "search_terms": query,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": per_sku * 3,
                "fields": "code,product_name,brands,image_front_url,image_url,countries_tags",
            }
        )
        url = f"https://world.openfoodfacts.org/cgi/search.pl?{params}"
        try:
            payload = request_json(url)
        except Exception as exc:  # noqa: BLE001 - keep building the review pack.
            if errors is not None:
                errors.append(f"{target.sku}: Open Food Facts search failed for '{query}': {exc}")
            continue
        for product in payload.get("products", []):
            image_url = product.get("image_front_url") or product.get("image_url")
            if not image_url or image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            candidates.append(
                {
                    "kind": "product",
                    "sku": target.sku,
                    "target_name": target.name,
                    "query": query,
                    "source": "Open Food Facts",
                    "source_url": image_url,
                    "product_name": product.get("product_name") or "",
                    "brands": product.get("brands") or "",
                    "code": product.get("code") or "",
                }
            )
            if len(candidates) >= per_sku:
                return candidates
        time.sleep(0.5)
    return candidates


def commons_file_titles(limit: int, errors: list[str] | None = None) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for category in COMMONS_CATEGORIES:
        params = urlencode(
            {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmtype": "file",
                "cmlimit": 50,
                "format": "json",
            }
        )
        try:
            payload = request_json(f"https://commons.wikimedia.org/w/api.php?{params}")
        except Exception as exc:  # noqa: BLE001
            if errors is not None:
                errors.append(f"{category}: Wikimedia category query failed: {exc}")
            continue
        for item in payload.get("query", {}).get("categorymembers", []):
            title = item.get("title", "")
            lowered = title.lower()
            if title in seen:
                continue
            if not any(token in lowered for token in COMMONS_TITLE_ALLOWLIST):
                continue
            seen.add(title)
            titles.append(title)
            if len(titles) >= limit:
                return titles
        time.sleep(0.5)
    return titles


def commons_candidates(limit: int, errors: list[str] | None = None) -> list[dict]:
    titles = commons_file_titles(limit * 2, errors=errors)
    candidates: list[dict] = []
    for title in titles:
        params = urlencode(
            {
                "action": "query",
                "prop": "imageinfo",
                "iiprop": "url",
                "iiurlwidth": 1280,
                "titles": title,
                "format": "json",
            }
        )
        try:
            payload = request_json(f"https://commons.wikimedia.org/w/api.php?{params}")
        except Exception as exc:  # noqa: BLE001
            if errors is not None:
                errors.append(f"{title}: Wikimedia imageinfo query failed: {exc}")
            continue
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            imageinfo = (page.get("imageinfo") or [{}])[0]
            image_url = imageinfo.get("thumburl") or imageinfo.get("url")
            if not image_url:
                continue
            candidates.append(
                {
                    "kind": "shelf",
                    "source": "Wikimedia Commons",
                    "source_title": title,
                    "source_url": imageinfo.get("descriptionurl") or image_url,
                    "image_url": image_url,
                }
            )
            if len(candidates) >= limit:
                return candidates
        time.sleep(0.5)
    return candidates


def write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    if not rows:
        manifest_path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare candidate beverage photos for human review.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/candidate_photo_pack"),
        help="Directory to create/update.",
    )
    parser.add_argument("--per-sku", type=int, default=8)
    parser.add_argument("--shelf-count", type=int, default=15)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    if args.clear and output_dir.exists():
        for path in sorted(output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    product_dir = output_dir / "product_candidates"
    shelf_dir = output_dir / "shelf_candidates"
    rows: list[dict] = []
    errors: list[str] = []

    for target in TARGET_PRODUCTS:
        for index, candidate in enumerate(
            openfoodfacts_candidates(target, args.per_sku, errors=errors),
            start=1,
        ):
            ext = Path(candidate["source_url"].split("?", 1)[0]).suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                ext = ".jpg"
            file_path = product_dir / target.sku / f"{index:02d}_{slugify(candidate.get('product_name') or target.name)}{ext}"
            try:
                raw = download_bytes(candidate["source_url"])
                ok, size = save_image(raw, file_path)
                if not ok:
                    errors.append(f"Invalid product image: {candidate['source_url']}")
                    continue
                rows.append({**candidate, "local_path": str(file_path), "width": size[0], "height": size[1]})
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{target.sku}: {candidate['source_url']}: {exc}")
            time.sleep(0.3)

    for index, candidate in enumerate(commons_candidates(args.shelf_count, errors=errors), start=1):
        ext = Path(candidate["image_url"].split("?", 1)[0]).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            ext = ".jpg"
        file_path = shelf_dir / f"{index:02d}_{slugify(candidate.get('source_title', 'shelf'))}{ext}"
        try:
            raw = download_bytes(candidate["image_url"])
            ok, size = save_image(raw, file_path)
            if not ok:
                errors.append(f"Invalid shelf image: {candidate['image_url']}")
                continue
            rows.append({**candidate, "local_path": str(file_path), "width": size[0], "height": size[1]})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"shelf: {candidate['image_url']}: {exc}")
        time.sleep(0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "manifest.csv", rows)
    (output_dir / "manifest.json").write_text(
        json.dumps({"items": rows, "errors": errors}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "output_dir": str(output_dir),
        "product_images": sum(1 for row in rows if row.get("kind") == "product"),
        "shelf_images": sum(1 for row in rows if row.get("kind") == "shelf"),
        "errors": len(errors),
        "manifest": str(output_dir / "manifest.csv"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
