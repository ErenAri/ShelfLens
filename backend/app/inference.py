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


@dataclass(frozen=True)
class DetectionRegion:
    bbox: tuple[int, int, int, int]
    detection_confidence: float
    sku_hint: str | None = None
    recognition_hint: float | None = None


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

    backend = "mock_deterministic"
    detector_model_path: str | None = None

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
    """Detection + recognition pipeline for small SKU catalogs."""

    BEVERAGE_CLASS_NAMES = {"bottle", "cup", "wine glass"}
    GENERIC_DETECTOR_CLASS_NAMES = {"object", "product", "retail_product", "item"}

    def __init__(
        self,
        clip_model_name: str,
        detector_model_path: str | None = None,
        fallback: InferenceEngine | None = None,
        min_detection_confidence: float = 0.35,
        max_detections: int = 10,
        min_recognition_margin: float = 0.0,
        min_recognition_confidence: float = 0.60,
    ) -> None:
        self.clip_model_name = clip_model_name
        self.fallback = fallback or MockInferenceEngine()
        self.min_detection_confidence = min_detection_confidence
        self.max_detections = max_detections
        self.min_recognition_margin = min_recognition_margin
        self.min_recognition_confidence = min_recognition_confidence

        self.detector_model_path = detector_model_path
        self.backend = "ultralytics_yolo" if detector_model_path else "torchvision_fasterrcnn"

        self._detector: Any | None = None
        self._detector_categories: list[str] | None = None
        self._yolo_detector: Any | None = None
        self._custom_detector_unavailable = False
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
                if self.detector_model_path:
                    return []
                return self.fallback.infer(image_path, image_id, catalog)

            catalog_by_sku = {item.sku: item.name for item in catalog}
            prototypes: list[tuple[str, str, np.ndarray]] | None = None

            output: list[DetectionCandidate] = []
            with Image.open(image_path).convert("RGB") as image:
                for region in detections:
                    matched_sku: str | None = None
                    matched_name: str | None = None
                    recognition_score = 0.0
                    recognition_margin = 1.0

                    if region.sku_hint and region.sku_hint in catalog_by_sku:
                        matched_sku = region.sku_hint
                        matched_name = catalog_by_sku[region.sku_hint]
                        recognition_score = region.recognition_hint or region.detection_confidence
                    elif catalog:
                        if prototypes is None:
                            prototypes = self._build_catalog_prototypes(catalog)
                        if prototypes:
                            crop = image.crop(region.bbox)
                            matched_sku, matched_name, recognition_score, recognition_margin = self._match_catalog(
                                crop,
                                prototypes,
                            )
                        elif region.sku_hint:
                            matched_sku = region.sku_hint
                            matched_name = catalog_by_sku.get(region.sku_hint) or region.sku_hint
                            recognition_score = region.recognition_hint or region.detection_confidence

                    has_match = bool(matched_sku and matched_name)
                    confidence = self._combine_confidence(
                        region.detection_confidence,
                        recognition_score,
                        has_match,
                    )
                    status = self._classify_status(
                        confidence,
                        recognition_score,
                        recognition_margin,
                        has_match,
                    )
                    if status == "unknown_product":
                        matched_sku = None
                        matched_name = None

                    output.append(
                        DetectionCandidate(
                            bbox=region.bbox,
                            sku=matched_sku,
                            product_name=matched_name,
                            detection_confidence=round(region.detection_confidence, 2),
                            recognition_confidence=round(recognition_score, 2),
                            confidence=confidence,
                            status=status,
                        )
                    )

            if output:
                return output
            if self.detector_model_path:
                return []
            return self.fallback.infer(image_path, image_id, catalog)
        except Exception as exc:
            if self.detector_model_path:
                logger.warning("Real inference failed with custom detector: %s", exc)
                return []
            logger.warning("Real inference failed, using mock fallback: %s", exc)
            return self.fallback.infer(image_path, image_id, catalog)

    def _detect_regions(self, image_path: Path) -> list[DetectionRegion]:
        if self.detector_model_path:
            return self._detect_regions_with_custom_yolo(image_path)

        return self._detect_regions_with_torchvision(image_path)

    def _detect_regions_with_custom_yolo(self, image_path: Path) -> list[DetectionRegion]:
        if self._custom_detector_unavailable:
            return []

        model_path = Path(self.detector_model_path or "")
        if not model_path.exists():
            logger.warning("Configured detector model not found: %s", model_path)
            self._custom_detector_unavailable = True
            return []

        detector = self._load_custom_detector()
        if detector is None:
            self._custom_detector_unavailable = True
            return []

        try:
            results = detector.predict(source=str(image_path), conf=self.min_detection_confidence, verbose=False)
        except Exception as exc:
            logger.warning("Custom detector inference failed: %s", exc)
            return []

        if not results:
            return []

        first = results[0]
        boxes = first.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        names = first.names if hasattr(first, "names") else {}

        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size

        regions: list[DetectionRegion] = []
        for box, score, class_id in zip(xyxy, scores, class_ids):
            score_value = float(score)
            if score_value < self.min_detection_confidence:
                continue

            bbox = self._sanitize_bbox(box, width, height)
            class_name = names.get(int(class_id)) if isinstance(names, dict) else None
            sku_hint = class_name.strip() if isinstance(class_name, str) and class_name.strip() else None
            if sku_hint and sku_hint.lower() in self.GENERIC_DETECTOR_CLASS_NAMES:
                sku_hint = None
            regions.append(
                DetectionRegion(
                    bbox=bbox,
                    detection_confidence=score_value,
                    sku_hint=sku_hint,
                    recognition_hint=score_value,
                )
            )

        return regions[: self.max_detections]

    def _detect_regions_with_torchvision(self, image_path: Path) -> list[DetectionRegion]:
        detector, categories = self._load_detector()
        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size
            input_tensor = self._pil_to_tensor(image)
            prediction = detector([input_tensor])[0]

        boxes = prediction["boxes"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        filtered: list[DetectionRegion] = []

        for box, label, score in zip(boxes, labels, scores):
            score_value = float(score)
            if score_value < self.min_detection_confidence:
                continue

            class_name = categories[int(label)] if int(label) < len(categories) else ""
            is_beverage = class_name in self.BEVERAGE_CLASS_NAMES
            if is_beverage:
                filtered.append(
                    DetectionRegion(
                        bbox=self._sanitize_bbox(box, width, height),
                        detection_confidence=score_value,
                    )
                )

        if not filtered:
            for box, score in zip(boxes[: self.max_detections], scores[: self.max_detections]):
                score_value = float(score)
                if score_value < self.min_detection_confidence:
                    continue
                filtered.append(
                    DetectionRegion(
                        bbox=self._sanitize_bbox(box, width, height),
                        detection_confidence=score_value,
                    )
                )

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
    ) -> tuple[str | None, str | None, float, float]:
        embedder = self._load_embedder()
        crop_vector = self._encode_with_normalization(embedder, [crop])[0]

        ranked = sorted(
            ((float(np.dot(crop_vector, vector)), sku, name) for sku, name, vector in prototypes),
            reverse=True,
        )
        if not ranked:
            return None, None, 0.0, 0.0

        best_similarity, best_sku, best_name = ranked[0]
        second_similarity = ranked[1][0] if len(ranked) > 1 else -1.0
        recognition_margin = max(0.0, best_similarity - second_similarity)

        recognition_confidence = max(0.0, min(1.0, (best_similarity + 1.0) / 2.0))
        return best_sku, best_name, recognition_confidence, recognition_margin

    def _combine_confidence(
        self,
        detection_confidence: float,
        recognition_confidence: float,
        has_match: bool,
    ) -> float:
        if self.detector_model_path and has_match:
            return round(recognition_confidence, 2)
        return round((detection_confidence + recognition_confidence) / 2.0, 2)

    def _classify_status(
        self,
        confidence: float,
        recognition_confidence: float,
        recognition_margin: float,
        has_match: bool,
    ) -> str:
        if self.detector_model_path:
            if not has_match or recognition_confidence < 0.55:
                return "unknown_product"
            if (
                recognition_confidence < self.min_recognition_confidence
                or recognition_margin < self.min_recognition_margin
            ):
                return "low_confidence"
            return "recognized"

        if confidence < 0.50:
            return "unknown_product"
        if confidence < 0.60:
            return "low_confidence"
        return "recognized"

    def _load_custom_detector(self) -> Any | None:
        if self._yolo_detector is not None:
            return self._yolo_detector

        model_path = Path(self.detector_model_path or "")
        if not model_path.exists():
            return None

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning(
                "ultralytics not installed; install it to use SHELFLENS_DETECTOR_MODEL_PATH."
            )
            return None

        self._yolo_detector = YOLO(str(model_path))
        return self._yolo_detector

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


def create_inference_engine(
    mode: str,
    clip_model_name: str,
    detector_model_path: str | None = None,
    min_detection_confidence: float = 0.35,
    max_detections: int = 10,
    min_recognition_margin: float = 0.0,
    min_recognition_confidence: float = 0.60,
) -> InferenceEngine:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "real":
        return RealInferenceEngine(
            clip_model_name=clip_model_name,
            detector_model_path=detector_model_path,
            fallback=MockInferenceEngine(),
            min_detection_confidence=min_detection_confidence,
            max_detections=max_detections,
            min_recognition_margin=min_recognition_margin,
            min_recognition_confidence=min_recognition_confidence,
        )
    return MockInferenceEngine()
