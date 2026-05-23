from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import shutil
import sys
import uuid

from fastapi.testclient import TestClient
from PIL import Image
import pytest


TEST_DIR = Path(__file__).resolve().parents[1] / "test_runtime"
os.environ["SHELFLENS_DATA_DIR"] = str(TEST_DIR)
os.environ["SHELFLENS_DB_URL"] = f"sqlite:///{(TEST_DIR / 'shelflens_test.db').as_posix()}"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import engine  # noqa: E402
from app.main import app  # noqa: E402


def make_png_bytes(width: int = 640, height: int = 360) -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", (width, height), color=(230, 230, 230))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def client() -> TestClient:
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    with TestClient(app) as test_client:
        yield test_client

    engine.dispose()
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR, ignore_errors=True)


def test_seed_products_exist(client: TestClient) -> None:
    response = client.get("/api/products")
    assert response.status_code == 200
    products = response.json()
    assert len(products) == 20
    assert products[0]["sku"] == "bev_001"
    assert products[0]["reference_image_count"] == 0


def test_inference_status_endpoint(client: TestClient) -> None:
    response = client.get("/api/system/inference")
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] in {"mock", "real"}
    assert "engine" in payload
    assert "backend" in payload
    assert "detector_model_path" in payload


def test_upload_rejects_invalid_file_type(client: TestClient) -> None:
    response = client.post(
        "/api/images",
        files={"file": ("bad.txt", b"invalid", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only .jpg, .jpeg, and .png" in response.json()["detail"]


def test_upload_creates_records_and_annotation(client: TestClient) -> None:
    response = client.post(
        "/api/images",
        files={"file": ("shelf.png", make_png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["width"] == 640
    assert payload["height"] == 360
    assert 2 <= len(payload["detections"]) <= 6

    for item in payload["detections"]:
        assert len(item["bbox"]) == 4
        assert 0.0 <= item["detection_confidence"] <= 1.0
        assert 0.0 <= item["recognition_confidence"] <= 1.0
        assert 0.0 <= item["confidence"] <= 1.0

        if item["confidence"] < 0.5:
            assert item["status"] == "unknown_product"
        elif item["confidence"] < 0.6:
            assert item["status"] == "low_confidence"

    image_id = payload["id"]
    results = client.get(f"/api/images/{image_id}/results")
    assert results.status_code == 200
    assert len(results.json()) == len(payload["detections"])

    annotated = client.get(f"/api/images/{image_id}/annotated")
    assert annotated.status_code == 200
    assert annotated.headers["content-type"].startswith("image/")


def test_detection_patch_persists_correction(client: TestClient) -> None:
    upload = client.post(
        "/api/images",
        files={"file": ("shelf.png", make_png_bytes(720, 480), "image/png")},
    )
    assert upload.status_code == 200
    detection_id = upload.json()["detections"][0]["id"]
    image_id = upload.json()["id"]

    patch = client.patch(
        f"/api/detections/{detection_id}",
        json={
            "sku": "bev_001",
            "product_name": "Corrected Beverage",
            "confidence_override": 0.88,
        },
    )
    assert patch.status_code == 200
    patched = patch.json()
    assert patched["status"] == "corrected"
    assert patched["sku"] == "bev_001"
    assert patched["product_name"] == "Corrected Beverage"
    assert patched["confidence"] == 0.88

    results = client.get(f"/api/images/{image_id}/results")
    assert results.status_code == 200
    updated = [item for item in results.json() if item["id"] == detection_id][0]
    assert updated["status"] == "corrected"
    assert updated["product_name"] == "Corrected Beverage"


def test_reference_image_upload_and_list(client: TestClient) -> None:
    upload = client.post(
        "/api/products/bev_001/reference-images",
        files={"file": ("reference.png", make_png_bytes(240, 240), "image/png")},
    )
    assert upload.status_code == 201
    payload = upload.json()
    assert payload["sku"] == "bev_001"
    assert payload["image_url"].endswith(payload["file_name"])

    listed = client.get("/api/products/bev_001/reference-images")
    assert listed.status_code == 200
    images = listed.json()
    assert len(images) == 1
    assert images[0]["file_name"] == payload["file_name"]

    products = client.get("/api/products")
    assert products.status_code == 200
    bev = [item for item in products.json() if item["sku"] == "bev_001"][0]
    assert bev["reference_image_count"] == 1


def test_save_detection_as_reference(client: TestClient) -> None:
    upload = client.post(
        "/api/images",
        files={"file": ("shelf.png", make_png_bytes(720, 480), "image/png")},
    )
    assert upload.status_code == 200
    first_detection = upload.json()["detections"][0]

    # Ensure SKU is known for the save-reference action.
    if not first_detection["sku"]:
        patched = client.patch(
            f"/api/detections/{first_detection['id']}",
            json={"sku": "bev_001", "product_name": "Cola Classic 330ml"},
        )
        assert patched.status_code == 200
        first_detection = patched.json()

    save = client.post(f"/api/detections/{first_detection['id']}/save-reference")
    assert save.status_code == 201
    saved = save.json()
    assert saved["sku"] == first_detection["sku"]

    listed = client.get(f"/api/products/{saved['sku']}/reference-images")
    assert listed.status_code == 200
    references = listed.json()
    assert any(item["file_name"] == saved["file_name"] for item in references)


def test_active_learning_export_creates_files(client: TestClient) -> None:
    upload = client.post(
        "/api/images",
        files={"file": ("shelf.png", make_png_bytes(720, 480), "image/png")},
    )
    assert upload.status_code == 200
    first_detection = upload.json()["detections"][0]

    patch = client.patch(
        f"/api/detections/{first_detection['id']}",
        json={
            "sku": "bev_001",
            "product_name": "Cola Classic 330ml",
            "confidence_override": 0.95,
        },
    )
    assert patch.status_code == 200

    export_name = f"active_learning_{uuid.uuid4().hex[:8]}"
    exported = client.post(
        "/api/system/active-learning/export",
        json={
            "export_name": export_name,
            "corrected_only": True,
            "min_confidence": 0.0,
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1,
            "include_recognition_crops": True,
        },
    )
    assert exported.status_code == 200
    payload = exported.json()
    assert payload["export_name"] == export_name
    assert payload["detection_label_mode"] == "product"
    assert payload["class_names"] == ["product"]
    assert payload["total_detections_exported"] >= 1
    assert payload["detection"]["enabled"] is True
    assert payload["recognition"]["enabled"] is True

    export_path = Path(payload["export_path"])
    manifest_path = Path(payload["manifest_path"])
    dataset_yaml_path = Path(payload["detection"]["dataset_yaml_path"])
    assert export_path.exists()
    assert manifest_path.exists()
    assert dataset_yaml_path.exists()
    assert payload["recognition"]["image_count"] >= 1


def test_active_learning_export_rejects_invalid_split_ratio(client: TestClient) -> None:
    exported = client.post(
        "/api/system/active-learning/export",
        json={
            "train_ratio": 0.8,
            "val_ratio": 0.3,
            "test_ratio": 0.1,
        },
    )
    assert exported.status_code == 400
    assert "must equal 1.0" in exported.json()["detail"]
