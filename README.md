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

Optional detector training dependency:

```powershell
cd backend
pip install ultralytics
```

Set backend mode:

```powershell
$env:SHELFLENS_INFERENCE_MODE="real"
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

Export YOLO/recognition datasets from current detections:

```powershell
cd backend
python scripts/export_active_learning_dataset.py --min-confidence 0.6
```

Train a YOLO detector from latest export:

```powershell
cd backend
python scripts/train_detector_yolo.py --epochs 40 --imgsz 640 --run-name shelflens_mvp
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
