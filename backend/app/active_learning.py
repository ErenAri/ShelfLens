from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from typing import Any

from PIL import Image
from sqlalchemy.orm import Session

from .config import EXPORTS_DIR
from .models import Detection, ImageRecord, Product


EXPORT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,64}$")


@dataclass(frozen=True)
class ActiveLearningExportOptions:
    export_name: str | None
    corrected_only: bool
    min_confidence: float
    train_ratio: float
    val_ratio: float
    test_ratio: float
    include_recognition_crops: bool
    detection_label_mode: str = "product"


@dataclass(frozen=True)
class LabeledDetection:
    detection_id: int
    image_id: str
    image_path: Path
    width: int
    height: int
    bbox: tuple[int, int, int, int]
    sku: str | None
    product_name: str | None
    confidence: float
    status: str


def export_active_learning_dataset(
    db: Session,
    options: ActiveLearningExportOptions,
) -> dict[str, Any]:
    export_name = _resolve_export_name(options.export_name)
    export_root = EXPORTS_DIR / export_name
    if export_root.exists():
        raise FileExistsError(f"Export folder already exists: {export_name}")
    export_root.mkdir(parents=True, exist_ok=False)

    rows = (
        db.query(Detection, ImageRecord, Product)
        .join(ImageRecord, Detection.image_id == ImageRecord.id)
        .outerjoin(Product, Detection.product_id == Product.id)
        .order_by(Detection.id.asc())
        .all()
    )

    total_scanned = len(rows)
    skipped_no_sku = 0
    skipped_missing_image = 0
    skipped_invalid_box = 0
    corrected_exported = 0
    exported: list[LabeledDetection] = []

    for detection, image, product in rows:
        if options.corrected_only and detection.status != "corrected":
            continue
        if detection.confidence < options.min_confidence:
            continue

        sku = (detection.override_sku or (product.sku if product else None) or "").strip() or None
        name = (
            detection.override_product_name
            or (product.name if product else None)
            or sku
        )
        product_name = name.strip() if name else None
        if options.detection_label_mode == "sku" and not sku:
            skipped_no_sku += 1
            continue

        source_path = Path(image.stored_path)
        if not source_path.exists():
            skipped_missing_image += 1
            continue

        bbox = _sanitize_bbox(
            (
                detection.bbox_x1,
                detection.bbox_y1,
                detection.bbox_x2,
                detection.bbox_y2,
            ),
            image.width,
            image.height,
        )
        if bbox is None:
            skipped_invalid_box += 1
            continue

        exported.append(
            LabeledDetection(
                detection_id=detection.id,
                image_id=detection.image_id,
                image_path=source_path,
                width=image.width,
                height=image.height,
                bbox=bbox,
                sku=sku,
                product_name=product_name,
                confidence=detection.confidence,
                status=detection.status,
            )
        )
        if detection.status == "corrected":
            corrected_exported += 1

    split_by_image = _build_image_split_map(
        image_ids={item.image_id for item in exported},
        train_ratio=options.train_ratio,
        val_ratio=options.val_ratio,
    )
    if options.detection_label_mode == "product":
        class_names = ["product"] if exported else []
    else:
        class_names = sorted({item.sku for item in exported if item.sku})
    class_map = {sku: idx for idx, sku in enumerate(class_names)}

    detection_summary = _export_detection_dataset(
        export_root=export_root,
        items=exported,
        class_map=class_map,
        class_names=class_names,
        split_by_image=split_by_image,
    )
    recognition_summary = _export_recognition_dataset(
        export_root=export_root,
        items=exported,
        split_by_image=split_by_image,
        enabled=options.include_recognition_crops,
    )

    generated_at = datetime.now(UTC)
    manifest_path = export_root / "manifest.json"
    manifest: dict[str, Any] = {
        "export_name": export_name,
        "export_path": str(export_root),
        "generated_at": generated_at.isoformat(),
        "corrected_only": options.corrected_only,
        "min_confidence": options.min_confidence,
        "train_ratio": options.train_ratio,
        "val_ratio": options.val_ratio,
        "test_ratio": options.test_ratio,
        "include_recognition_crops": options.include_recognition_crops,
        "detection_label_mode": options.detection_label_mode,
        "total_detections_scanned": total_scanned,
        "total_detections_exported": len(exported),
        "corrected_detections_exported": corrected_exported,
        "skipped_no_sku": skipped_no_sku,
        "skipped_missing_image": skipped_missing_image,
        "skipped_invalid_box": skipped_invalid_box,
        "class_names": class_names,
        "quality_warnings": _build_quality_warnings(exported, split_by_image, corrected_exported),
        "detection": detection_summary,
        "recognition": recognition_summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)

    return manifest


def _resolve_export_name(export_name: str | None) -> str:
    if export_name:
        candidate = export_name.strip()
        if not EXPORT_NAME_PATTERN.fullmatch(candidate):
            raise ValueError(
                "export_name must be 3-64 chars and contain only letters, numbers, '_' or '-'."
            )
        return candidate
    return datetime.now(UTC).strftime("export_%Y%m%d_%H%M%S")


def _build_image_split_map(
    image_ids: set[str],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    ordered = sorted(
        image_ids,
        key=lambda image_id: sha256(image_id.encode("utf-8")).hexdigest(),
    )
    total = len(ordered)
    if total == 0:
        return {}

    train_count = max(1, int(round(total * train_ratio)))
    val_count = int(round(total * val_ratio))
    test_count = total - train_count - val_count

    if total >= 3:
        if val_count == 0:
            val_count = 1
            train_count -= 1
        if test_count == 0:
            test_count = 1
            train_count -= 1
    elif total == 2 and val_count == 0:
        val_count = 1
        train_count = 1
        test_count = 0

    if train_count < 1:
        train_count = 1
    while train_count + val_count + test_count > total:
        if train_count > 1:
            train_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            test_count -= 1
    while train_count + val_count + test_count < total:
        train_count += 1

    split_map: dict[str, str] = {}
    for index, image_id in enumerate(ordered):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        split_map[image_id] = split
    return split_map


def _build_quality_warnings(
    items: list[LabeledDetection],
    split_by_image: dict[str, str],
    corrected_exported: int,
) -> list[str]:
    warnings: list[str] = []
    image_count = len({item.image_id for item in items})
    if image_count < 100:
        warnings.append(
            "Detector dataset is small; collect at least 300-500 real shelf/product images for reliable training."
        )
    if corrected_exported == 0:
        warnings.append(
            "No corrected detections were exported; use human-corrected boxes for production-quality metrics."
        )
    split_counts = {
        "train": sum(1 for split in split_by_image.values() if split == "train"),
        "val": sum(1 for split in split_by_image.values() if split == "val"),
        "test": sum(1 for split in split_by_image.values() if split == "test"),
    }
    if image_count >= 3 and split_counts["test"] == 0:
        warnings.append("Test split is empty; hold out real test images before claiming accuracy.")
    return warnings


def _sanitize_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if width <= 1 or height <= 1:
        return None

    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 2, x1))
    y1 = max(0, min(height - 2, y1))
    x2 = max(x1 + 1, min(width - 1, x2))
    y2 = max(y1 + 1, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _to_yolo_xywh(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = x1 + box_width / 2.0
    center_y = y1 + box_height / 2.0
    return (
        max(0.0, min(1.0, center_x / width)),
        max(0.0, min(1.0, center_y / height)),
        max(0.0, min(1.0, box_width / width)),
        max(0.0, min(1.0, box_height / height)),
    )


def _export_detection_dataset(
    export_root: Path,
    items: list[LabeledDetection],
    class_map: dict[str, int],
    class_names: list[str],
    split_by_image: dict[str, str],
) -> dict[str, Any]:
    if not items:
        return {
            "enabled": False,
            "root_path": None,
            "image_count": 0,
            "annotation_count": 0,
            "split_counts": {"train": 0, "val": 0, "test": 0},
            "dataset_yaml_path": None,
        }

    root = export_root / "detection"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    labels_by_image: dict[str, list[str]] = defaultdict(list)
    source_by_image: dict[str, Path] = {}
    split_annotation_counts = {"train": 0, "val": 0, "test": 0}

    for item in items:
        class_key = "product" if "product" in class_map else item.sku
        if class_key is None:
            continue
        class_index = class_map[class_key]
        x, y, w, h = _to_yolo_xywh(item.bbox, item.width, item.height)
        labels_by_image[item.image_id].append(f"{class_index} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        source_by_image[item.image_id] = item.image_path
        split = split_by_image[item.image_id]
        split_annotation_counts[split] += 1

    split_image_counts = {"train": 0, "val": 0, "test": 0}
    for image_id, label_lines in labels_by_image.items():
        split = split_by_image[image_id]
        source = source_by_image[image_id]
        suffix = source.suffix.lower() if source.suffix else ".jpg"
        dest_image = root / "images" / split / f"{image_id}{suffix}"
        shutil.copy2(source, dest_image)
        split_image_counts[split] += 1

        label_path = root / "labels" / split / f"{image_id}.txt"
        label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    dataset_yaml_path = root / "dataset.yaml"
    names_lines = [f"  {index}: {name}" for index, name in enumerate(class_names)]
    dataset_yaml_content = "\n".join(
        [
            f"path: {root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            *names_lines,
            "",
        ]
    )
    dataset_yaml_path.write_text(dataset_yaml_content, encoding="utf-8")

    return {
        "enabled": True,
        "root_path": str(root),
        "image_count": sum(split_image_counts.values()),
        "annotation_count": len(items),
        "split_counts": split_image_counts,
        "annotation_split_counts": split_annotation_counts,
        "dataset_yaml_path": str(dataset_yaml_path),
    }


def _export_recognition_dataset(
    export_root: Path,
    items: list[LabeledDetection],
    split_by_image: dict[str, str],
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "root_path": None,
            "image_count": 0,
            "annotation_count": 0,
            "split_counts": {"train": 0, "val": 0, "test": 0},
        }
    if not items:
        return {
            "enabled": True,
            "root_path": str(export_root / "recognition"),
            "image_count": 0,
            "annotation_count": 0,
            "split_counts": {"train": 0, "val": 0, "test": 0},
        }

    root = export_root / "recognition"
    root.mkdir(parents=True, exist_ok=True)
    split_crop_counts = {"train": 0, "val": 0, "test": 0}
    images_by_source: dict[str, list[LabeledDetection]] = defaultdict(list)
    for item in items:
        if not item.sku:
            continue
        images_by_source[item.image_id].append(item)
    exportable_count = sum(len(detection_items) for detection_items in images_by_source.values())

    for image_id, detection_items in images_by_source.items():
        source = detection_items[0].image_path
        with Image.open(source).convert("RGB") as image:
            for item in detection_items:
                split = split_by_image[image_id]
                sku_dir = root / split / item.sku
                sku_dir.mkdir(parents=True, exist_ok=True)
                output_path = sku_dir / f"{item.image_id}_{item.detection_id}.jpg"
                crop = image.crop(item.bbox)
                crop.save(output_path, format="JPEG", quality=92)
                split_crop_counts[split] += 1

    return {
        "enabled": True,
        "root_path": str(root),
        "image_count": sum(split_crop_counts.values()),
        "annotation_count": exportable_count,
        "split_counts": split_crop_counts,
    }
