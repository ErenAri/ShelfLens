# ShelfLens MVP

Simple uploaded-image product recognition system for a small beverage SKU catalog.

## Stack

- Backend: FastAPI + SQLAlchemy + SQLite + Pillow
- Frontend: React + TypeScript + Vite
- Inference: pluggable engine (`mock` default, optional `real` detection + CLIP recognition)

## Environment Examples

- Backend: `backend/.env.example`
- Frontend: `frontend/.env.example`

## Run Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Optional real-inference dependencies:

```powershell
cd backend
pip install -r requirements-ml.txt
```

Set backend mode:

```powershell
$env:SHELFLENS_INFERENCE_MODE="real"
```

Use a trained YOLO detector artifact (optional):

```powershell
$env:SHELFLENS_DETECTOR_MODEL_PATH="C:\Users\erena\Desktop\ShelfLens\backend\data\models\shelflens_mvp\weights\best.pt"
$env:SHELFLENS_MIN_DETECTION_CONFIDENCE="0.05"
$env:SHELFLENS_MAX_DETECTIONS="80"
$env:SHELFLENS_MIN_RECOGNITION_MARGIN="0.04"
```

## Run Frontend

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

## Tests

Backend:

```powershell
cd backend
pytest
```

Frontend:

```powershell
cd frontend
npm run test
```

## Active-Learning Export + Training

Export product-detector and recognition datasets from current detections. The default detector export is now single-class `product`; SKU identity stays in the recognition crop dataset.

Full external source workflow: `DATA_SOURCES.md`.

Pretrain a generic product detector from SKU-110K:

```powershell
cd backend
python scripts/train_detector_yolo.py --dataset-yaml SKU-110K.yaml --datasets-dir .\data\external --model yolov8s.pt --epochs 80 --imgsz 640 --batch 8 --device 0 --workers 4 --run-name sku110k_product_detector
```

```powershell
cd backend
python scripts/export_active_learning_dataset.py --export-name product_detector_01 --min-confidence 0.0 --detection-label-mode product
```

Train a detector from the product export:

```powershell
cd backend
python scripts/train_detector_yolo.py --dataset-yaml .\data\exports\product_detector_01\detection\dataset.yaml --epochs 60 --imgsz 640 --run-name product_detector_01
```

Report metrics against the 95% target:

```powershell
cd backend
python scripts/report_model_metrics.py --run-dir .\data\models\product_detector_01 --export-dir .\data\exports\product_detector_01 --target 0.95
```

Import SKU reference images from a local/image-url CSV:

```powershell
cd backend
python scripts/import_reference_catalog.py --write-template .\data\sources\beverage_references.csv
python scripts/import_reference_catalog.py --csv .\data\sources\beverage_references.csv
```

Add extra Open Food Facts front-image variants and check reference coverage:

```powershell
cd backend
python scripts/import_openfoodfacts_variants.py --csv .\data\sources\beverage_references.csv --max-per-sku 5
python scripts/report_reference_catalog.py --target-per-sku 5
```

Prepare and import the reviewed beverage candidate pack:

```powershell
cd backend
python scripts/sync_candidate_photo_pack.py --pack-dir .\data\candidate_photo_pack --copy-approved
python scripts/import_reference_catalog.py --csv .\data\candidate_photo_pack\approved_reference_import.csv --no-seed-products
```

The committed candidate pack is for human review and bootstrap references. Low-confidence shelf predictions should be corrected in the dashboard before exporting active-learning data.

Create a correction queue from existing uploads without writing new labels:

```powershell
cd backend
python scripts/build_correction_queue.py --model .\data\models\sku110k_product_detector_tune1_clean_20260523_1458-2\weights\best.pt --min-detection-confidence 0.05 --max-detections 80 --min-recognition-margin 0.04 --output-dir .\data\correction_queue\current_uploads
```

Bootstrap references from Open Food Facts barcodes:

```powershell
cd backend
python scripts/fetch_openfoodfacts_references.py --csv .\data\sources\openfoodfacts_barcodes.csv --country world
```

## API Surface

- `POST /api/images`
- `GET /api/images`
- `GET /api/images/{image_id}`
- `GET /api/images/{image_id}/results`
- `GET /api/images/{image_id}/annotated`
- `GET /api/products`
- `POST /api/products`
- `GET /api/system/inference`
- `POST /api/system/active-learning/export`
- `GET /api/products/{sku}/reference-images`
- `GET /api/products/{sku}/reference-images/{file_name}`
- `POST /api/products/{sku}/reference-images`
- `POST /api/detections/{detection_id}/save-reference`
- `PATCH /api/detections/{detection_id}`
