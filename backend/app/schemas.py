from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DetectionStatus(StrEnum):
    RECOGNIZED = "recognized"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN_PRODUCT = "unknown_product"
    CORRECTED = "corrected"


class ProductCreate(BaseModel):
    sku: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    category: str = Field(default="beverages", min_length=1, max_length=128)
    is_active: bool = True


class ProductOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    sku: str
    name: str
    category: str
    is_active: bool
    created_at: datetime
    reference_image_count: int = 0


class ProductReferenceImageOut(BaseModel):
    sku: str
    file_name: str
    file_path: str
    image_url: str


class SaveDetectionReferencePayload(BaseModel):
    sku: str | None = Field(default=None, min_length=1, max_length=64)
    product_name: str | None = Field(default=None, min_length=1, max_length=255)


class ActiveLearningExportRequest(BaseModel):
    export_name: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{3,64}$",
    )
    corrected_only: bool = False
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    test_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    include_recognition_crops: bool = True
    detection_label_mode: Literal["product", "sku"] = "product"


class ActiveLearningExportSectionOut(BaseModel):
    enabled: bool
    root_path: str | None
    image_count: int
    annotation_count: int
    split_counts: dict[str, int]
    annotation_split_counts: dict[str, int] | None = None
    dataset_yaml_path: str | None = None


class ActiveLearningExportOut(BaseModel):
    export_name: str
    export_path: str
    generated_at: str
    manifest_path: str
    corrected_only: bool
    min_confidence: float
    train_ratio: float
    val_ratio: float
    test_ratio: float
    include_recognition_crops: bool
    detection_label_mode: Literal["product", "sku"]
    total_detections_scanned: int
    total_detections_exported: int
    corrected_detections_exported: int
    skipped_no_sku: int
    skipped_missing_image: int
    skipped_invalid_box: int
    class_names: list[str]
    quality_warnings: list[str]
    detection: ActiveLearningExportSectionOut
    recognition: ActiveLearningExportSectionOut


class DetectionPatch(BaseModel):
    sku: str | None = Field(default=None, min_length=1, max_length=64)
    product_name: str | None = Field(default=None, min_length=1, max_length=255)
    confidence_override: float | None = Field(default=None, ge=0.0, le=1.0)


class DetectionOut(BaseModel):
    id: int
    image_id: str
    bbox: list[int]
    sku: str | None
    product_name: str | None
    detection_confidence: float
    recognition_confidence: float
    confidence: float
    status: DetectionStatus
    created_at: datetime
    updated_at: datetime


class ImageSummaryOut(BaseModel):
    id: str
    filename: str
    status: str
    width: int
    height: int
    created_at: datetime
    annotated_url: str | None


class ImageDetailOut(ImageSummaryOut):
    stored_path: str
    annotated_path: str | None


class ImageUploadOut(ImageDetailOut):
    detections: list[DetectionOut]


class ImageListOut(BaseModel):
    items: list[ImageSummaryOut]
    total: int
    limit: int
    offset: int
