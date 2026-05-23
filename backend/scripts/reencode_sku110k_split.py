from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageFile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backup and re-encode images listed in a SKU-110K split file."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to SKU-110K dataset root.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Split list file (e.g. val.txt).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for re-encoding.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset_root = args.dataset_root.resolve()
    split_file = args.split_file.resolve()
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    lines = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_root = dataset_root / "image_backups" / f"{split_file.stem}-preclean-{stamp}"
    backup_root.mkdir(parents=True, exist_ok=True)

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    processed = 0
    failed = 0
    for listed in lines:
        image_path = (dataset_root / listed.replace("\\", "/")).resolve()
        if not image_path.exists():
            failed += 1
            continue
        rel = image_path.relative_to(dataset_root)
        backup_path = backup_root / rel
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_bytes(image_path.read_bytes())

        try:
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                rgb.save(
                    image_path,
                    format="JPEG",
                    quality=int(args.jpeg_quality),
                    optimize=True,
                    progressive=False,
                )
            processed += 1
        except Exception:
            failed += 1

    print(f"backup_dir={backup_root}")
    print(f"processed={processed}")
    print(f"failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
