from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def normalized(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronize a human-reviewed candidate photo pack after files were deleted."
    )
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=Path("data/candidate_photo_pack"),
    )
    parser.add_argument(
        "--approved-dir-name",
        default="approved_import",
        help="Subdirectory for a copied clean import set.",
    )
    parser.add_argument(
        "--copy-approved",
        action="store_true",
        help="Copy existing reviewed files into approved_import/product_references and shelf_images.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pack_dir = args.pack_dir
    product_dir = pack_dir / "product_candidates"
    shelf_dir = pack_dir / "shelf_candidates"
    manifest_rows = read_manifest(pack_dir / "manifest.csv")

    existing_files = {
        normalized(path): path
        for path in pack_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and args.approved_dir_name not in path.parts
        and not path.name.endswith("_contact_sheet.jpg")
    }
    existing_by_name = {path.name: path for path in existing_files.values()}

    synced_rows: list[dict[str, object]] = []
    matched_paths: set[Path] = set()
    for row in manifest_rows:
        local_path = row.get("local_path") or ""
        candidates: list[Path] = []
        if local_path.strip():
            candidates.extend(
                [
                    Path(local_path),
                    pack_dir.parent / local_path,
                    pack_dir / local_path,
                ]
            )
            by_name = existing_by_name.get(Path(local_path).name)
            if by_name is not None:
                candidates.append(by_name)
        found = next((path for path in candidates if path.is_file()), None)
        if found is None:
            continue
        resolved = found.resolve()
        matched_paths.add(resolved)
        synced_rows.append({**row, "local_path": normalized(resolved)})

    for path in sorted(existing_files.values()):
        if path.resolve() in matched_paths:
            continue
        if product_dir in path.parents:
            sku = path.parent.name
            synced_rows.append(
                {
                    "kind": "product",
                    "sku": sku,
                    "target_name": sku,
                    "source": "local_reviewed_copy",
                    "source_url": "",
                    "local_path": normalized(path.resolve()),
                }
            )
        elif shelf_dir in path.parents:
            synced_rows.append(
                {
                    "kind": "shelf",
                    "source": "local_reviewed_copy",
                    "source_url": "",
                    "source_title": path.name,
                    "local_path": normalized(path.resolve()),
                }
            )

    product_rows = [row for row in synced_rows if row.get("kind") == "product"]
    shelf_rows = [row for row in synced_rows if row.get("kind") == "shelf"]
    product_counts = Counter(str(row.get("sku") or "unknown") for row in product_rows)

    synced_csv = pack_dir / "manifest_synced.csv"
    import_csv = pack_dir / "approved_reference_import.csv"
    shelf_csv = pack_dir / "approved_shelf_images.csv"
    write_csv(synced_csv, synced_rows)
    write_csv(
        import_csv,
        [
            {
                "sku": row.get("sku"),
                "name": row.get("target_name") or row.get("product_name") or row.get("sku"),
                "category": "beverages",
                "image_path": row.get("local_path"),
                "image_url": "",
            }
            for row in product_rows
        ],
    )
    write_csv(shelf_csv, shelf_rows)

    approved_dir = pack_dir / args.approved_dir_name
    if args.copy_approved:
        if approved_dir.exists():
            shutil.rmtree(approved_dir)
        for row in product_rows:
            src = Path(str(row["local_path"]))
            sku = str(row.get("sku") or "unknown")
            dst = approved_dir / "product_references" / sku / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        for row in shelf_rows:
            src = Path(str(row["local_path"]))
            dst = approved_dir / "shelf_images" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    summary = {
        "pack_dir": str(pack_dir),
        "product_images": len(product_rows),
        "shelf_images": len(shelf_rows),
        "product_counts": dict(sorted(product_counts.items())),
        "synced_manifest": str(synced_csv),
        "approved_reference_import": str(import_csv),
        "approved_shelf_images": str(shelf_csv),
        "approved_dir": str(approved_dir) if args.copy_approved else None,
        "low_count_skus": {
            sku: count for sku, count in sorted(product_counts.items()) if count < 5
        },
    }
    (pack_dir / "sync_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
