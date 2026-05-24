from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class RenderDetection:
    bbox: tuple[int, int, int, int]
    product_name: str | None
    confidence: float
    status: str


STATUS_COLORS = {
    "recognized": "#0f766e",
    "low_confidence": "#b45309",
    "unknown_product": "#b91c1c",
    "corrected": "#1d4ed8",
}

STATUS_LABELS = {
    "low_confidence": "Needs review",
    "unknown_product": "Unknown product",
}


def annotate_image(
    image_path: Path,
    output_path: Path,
    detections: Iterable[RenderDetection],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)

        for item in detections:
            color = STATUS_COLORS.get(item.status, "#0f172a")
            x1, y1, x2, y2 = item.bbox
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

            label_name = STATUS_LABELS.get(item.status, item.product_name or "Unknown")
            label = f"{label_name} ({item.confidence:.2f})"
            text_bbox = draw.textbbox((x1, y1), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            pad = 4
            top = max(0, y1 - text_height - pad * 2)
            draw.rectangle(
                [(x1, top), (x1 + text_width + pad * 2, top + text_height + pad * 2)],
                fill=color,
            )
            draw.text((x1 + pad, top + pad), label, fill="#ffffff")

        image.save(output_path, format="JPEG", quality=90)
