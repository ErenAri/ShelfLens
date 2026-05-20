from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import logging
from pathlib import Path
import random
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogItem:
    sku: str
    name: str
    reference_images: tuple[str, ...] = ()


@dataclass(frozen=True)
class DetectionCandidate:
    bbox: tuple[int, int, int, int]
    sku: str | None
    product_name: str | None
    detection_confidence: float
    recognition_confidence: float
    confidence: float
    status: str


class InferenceEngine(Protocol):
    def infer(
        self,
        image_path: Path,
        image_id: str,
        catalog: Sequence[CatalogItem],
    ) -> list[DetectionCandidate]:
        ...


class MockInferenceEngine:
    """Deterministic mock inference for the MVP pipeline."""

    def infer(
        self,
        image_path: Path,
        image_id: str,
        catalog: Sequence[CatalogItem],
    ) -> list[DetectionCandidate]:
        with Image.open(image_path) as img:
            width, height = img.size

        seed_input = f"{image_id}:{width}:{height}".encode("utf-8")
        seed = int(sha256(seed_input).hexdigest()[:16], 16)
        rng = random.Random(seed)
        detection_count = rng.randint(2, 6)
        output: list[DetectionCandidate] = []

        for _ in range(detection_count):
            bbox = self._make_bbox(rng, width, height)
            detection_confidence = round(rng.uniform(0.45, 0.98), 2)
            recognition_confidence = round(rng.uniform(0.45, 0.98), 2)
            confidence = round((detection_confidence + recognition_confidence) / 2.0, 2)

            sku = None
            product_name = None
            if catalog:
                picked = catalog[rng.randrange(0, len(catalog))]
                sku = picked.sku
                product_name = picked.name

            status = "recognized"
            if confidence < 0.50:
                status = "unknown_product"
                sku = None
                product_name = None
            elif confidence < 0.60:
                status = "low_confidence"

            output.append(
                DetectionCandidate(
                    bbox=bbox,
                    sku=sku,
                    product_name=product_name,
                    detection_confidence=detection_confidence,
                    recognition_confidence=recognition_confidence,
                    confidence=confidence,
                    status=status,
                )
            )

        return output

    @staticmethod
    def _make_bbox(rng: random.Random, width: int, height: int) -> tuple[int, int, int, int]:
        min_w = max(40, width // 8)
        max_w = max(min_w + 1, width // 3)
        min_h = max(40, height // 6)
        max_h = max(min_h + 1, height // 2)

        box_w = rng.randint(min_w, max_w)
        box_h = rng.randint(min_h, max_h)
        x1 = rng.randint(0, max(0, width - box_w - 1))
        y1 = rng.randint(0, max(0, height - box_h - 1))
        x2 = min(width - 1, x1 + box_w)
        y2 = min(height - 1, y1 + box_h)
        return (x1, y1, x2, y2)


class RealInferenceEngine:
    """Detection + embedding recognition pipeline for small SKU catalogs."""

    BEVERAGE_CLASS_NAMES = {"bottle", "cup", "wine glass"}

    def __init__(
        self,
        clip_model_name: str,
        fallback: InferenceEngine | None = None,
        min_detection_confidence: float = 0.35,
        max_detections: int = 10,
    ) -> None:
        self.clip_model_name = clip_model_name
        self.fallback = fallback or MockInferenceEngine()
        self.min_detection_confidence = min_detection_confidence
        self.max_detections = max_detections

        self._detector: Any | None = None
        self._detector_categories: list[str] | None = None
        self._embedder: Any | None = None

    def infer(
        self,
        image_path: Path,
        image_id: str,
        catalog: Sequence[CatalogItem],
    ) -> list[DetectionCandidate]:
        try:
            detections = self._detect_regions(image_path)
            if not detections:
                return self.fallback.infer(image_path, image_id, catalog)

            prototypes = self._build_catalog_prototypes(catalog)
            if not prototypes:
                return self.fallback.infer(image_path, image_id, catalog)

            output: list[DetectionCandidate] = []
            with Image.open(image_path).convert("RGB") as image:
                for bbox, detection_score in detections:
                    crop = image.crop(bbox)
                    matched_sku, matched_name, recognition_score = self._match_catalog(crop, prototypes)
                    confidence = round((detection_score + recognition_score) / 2.0, 2)
                    status = "recognized"
                    if confidence < 0.50:
                        status = "unknown_product"
                        matched_sku = None
                        matched_name = None
                    elif confidence < 0.60:
                        status = "low_confidence"

                    output.append(
                        DetectionCandidate(
                            bbox=bbox,
                            sku=matched_sku,
                            product_name=matched_name,
                            detection_confidence=round(detection_score, 2),
                            recognition_confidence=round(recognition_score, 2),
                            confidence=confidence,
                            status=status,
                        )
                    )

            return output if output else self.fallback.infer(image_path, image_id, catalog)
        except Exception as exc:
            logger.warning("Real inference failed, using mock fallback: %s", exc)
            return self.fallback.infer(image_path, image_id, catalog)

    def _detect_regions(self, image_path: Path) -> list[tuple[tuple[int, int, int, int], float]]:
        detector, categories = self._load_detector()
        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size
            input_tensor = self._pil_to_tensor(image)
            prediction = detector([input_tensor])[0]

        boxes = prediction["boxes"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        filtered: list[tuple[tuple[int, int, int, int], float]] = []

        for box, label, score in zip(boxes, labels, scores):
            score_value = float(score)
            if score_value < self.min_detection_confidence:
                continue

            class_name = categories[int(label)] if int(label) < len(categories) else ""
            is_beverage = class_name in self.BEVERAGE_CLASS_NAMES
            if is_beverage:
                filtered.append((self._sanitize_bbox(box, width, height), score_value))

        # If beverage-only filtering finds nothing, keep top detections as fallback regions.
        if not filtered:
            for box, score in zip(boxes[: self.max_detections], scores[: self.max_detections]):
                score_value = float(score)
                if score_value < self.min_detection_confidence:
                    continue
                filtered.append((self._sanitize_bbox(box, width, height), score_value))

        return filtered[: self.max_detections]

    def _build_catalog_prototypes(self, catalog: Sequence[CatalogItem]) -> list[tuple[str, str, np.ndarray]]:
        if not catalog:
            return []

        embedder = self._load_embedder()
        text_prompts = [f"a product photo of {item.name}" for item in catalog]
        text_vectors = self._encode_with_normalization(embedder, text_prompts)

        prototypes: list[tuple[str, str, np.ndarray]] = []
        for index, item in enumerate(catalog):
            vectors = [text_vectors[index]]

            ref_paths = [Path(path) for path in item.reference_images if Path(path).exists()]
            if ref_paths:
                ref_images = [Image.open(path).convert("RGB") for path in ref_paths]
                try:
                    ref_vectors = self._encode_with_normalization(embedder, ref_images)
                    vectors.extend(ref_vectors)
                finally:
                    for image in ref_images:
                        image.close()

            mean_vector = self._normalize(np.mean(np.stack(vectors), axis=0))
            prototypes.append((item.sku, item.name, mean_vector))

        return prototypes

    def _match_catalog(
        self,
        crop: Image.Image,
        prototypes: list[tuple[str, str, np.ndarray]],
    ) -> tuple[str | None, str | None, float]:
        embedder = self._load_embedder()
        crop_vector = self._encode_with_normalization(embedder, [crop])[0]

        best_sku: str | None = None
        best_name: str | None = None
        best_similarity = -1.0
        for sku, name, vector in prototypes:
            similarity = float(np.dot(crop_vector, vector))
            if similarity > best_similarity:
                best_similarity = similarity
                best_sku = sku
                best_name = name

        # Map cosine similarity [-1, 1] to [0, 1].
        recognition_confidence = max(0.0, min(1.0, (best_similarity + 1.0) / 2.0))
        return best_sku, best_name, recognition_confidence

    def _load_detector(self) -> tuple[Any, list[str]]:
        if self._detector is not None and self._detector_categories is not None:
            return self._detector, self._detector_categories

        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        detector.eval()
        detector.to(torch.device("cpu"))

        categories = list(weights.meta.get("categories", []))
        self._detector = detector
        self._detector_categories = categories
        return detector, categories

    def _load_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder

        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.clip_model_name)
        return self._embedder

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> Any:
        from torchvision.transforms.functional import to_tensor

        return to_tensor(image)

    @staticmethod
    def _sanitize_bbox(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box.tolist()]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width - 1, x2))
        y2 = max(y1 + 1, min(height - 1, y2))
        return (x1, y1, x2, y2)

    def _encode_with_normalization(self, embedder: Any, payload: Sequence[Any]) -> list[np.ndarray]:
        try:
            encoded = embedder.encode(
                list(payload),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except TypeError:
            # For older sentence-transformers versions without normalize_embeddings.
            encoded = embedder.encode(
                list(payload),
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = np.expand_dims(matrix, axis=0)
        return [self._normalize(vector) for vector in matrix]

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return vector
        return vector / norm


def create_inference_engine(mode: str, clip_model_name: str) -> InferenceEngine:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "real":
        return RealInferenceEngine(clip_model_name=clip_model_name, fallback=MockInferenceEngine())
    return MockInferenceEngine()

