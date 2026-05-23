from __future__ import annotations

from pathlib import Path
import uuid

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .annotations import RenderDetection, annotate_image
from .active_learning import ActiveLearningExportOptions, export_active_learning_dataset
from .config import (
    ALLOWED_EXTENSIONS,
    ANNOTATED_DIR,
    CLIP_MODEL_NAME,
    DETECTOR_MODEL_PATH,
    INFERENCE_MODE,
    REFERENCE_DIR,
    UPLOAD_DIR,
    ensure_storage_dirs,
)
from .database import Base, SessionLocal, engine
from .inference import CatalogItem, create_inference_engine
from .models import Detection, ImageRecord, Product
from .schemas import (
    ActiveLearningExportOut,
    ActiveLearningExportRequest,
    DetectionOut,
    DetectionPatch,
    ImageDetailOut,
    ImageListOut,
    ImageSummaryOut,
    ImageUploadOut,
    ProductCreate,
    ProductReferenceImageOut,
    ProductOut,
    SaveDetectionReferencePayload,
)
from .seed import seed_products


app = FastAPI(title="ShelfLens API", version="0.1.0")
inference_engine = create_inference_engine(INFERENCE_MODE, CLIP_MODEL_NAME, DETECTOR_MODEL_PATH)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    ensure_storage_dirs()
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_products(db)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _resolve_detection_name(item: Detection) -> tuple[str | None, str | None]:
    sku = item.override_sku or (item.product.sku if item.product else None)
    product_name = item.override_product_name or (item.product.name if item.product else None)
    return sku, product_name


def _to_detection_out(item: Detection) -> DetectionOut:
    sku, product_name = _resolve_detection_name(item)
    return DetectionOut(
        id=item.id,
        image_id=item.image_id,
        bbox=[item.bbox_x1, item.bbox_y1, item.bbox_x2, item.bbox_y2],
        sku=sku,
        product_name=product_name,
        detection_confidence=round(item.detection_confidence, 2),
        recognition_confidence=round(item.recognition_confidence, 2),
        confidence=round(item.confidence, 2),
        status=item.status,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def _to_image_summary(item: ImageRecord) -> ImageSummaryOut:
    annotated_url = f"/api/images/{item.id}/annotated" if item.annotated_path else None
    return ImageSummaryOut(
        id=item.id,
        filename=item.filename,
        status=item.status,
        width=item.width,
        height=item.height,
        created_at=item.created_at,
        annotated_url=annotated_url,
    )


def _to_image_detail(item: ImageRecord) -> ImageDetailOut:
    return ImageDetailOut(
        **_to_image_summary(item).model_dump(),
        stored_path=item.stored_path,
        annotated_path=item.annotated_path,
    )


def _reference_dir_for_sku(sku: str) -> Path:
    return REFERENCE_DIR / sku


def _list_reference_images_for_sku(sku: str) -> list[Path]:
    directory = _reference_dir_for_sku(sku)
    if not directory.exists():
        return []
    files = [path for path in directory.iterdir() if path.suffix.lower() in ALLOWED_EXTENSIONS]
    return sorted(files, key=lambda item: item.name)


def _to_product_out(item: Product) -> ProductOut:
    return ProductOut(
        id=item.id,
        sku=item.sku,
        name=item.name,
        category=item.category,
        is_active=item.is_active,
        created_at=item.created_at,
        reference_image_count=len(_list_reference_images_for_sku(item.sku)),
    )


def _get_or_create_product_by_sku(db: Session, sku: str, product_name: str | None = None) -> Product:
    product = db.query(Product).filter(Product.sku == sku).first()
    if product is not None:
        if product_name and product.name != product_name:
            product.name = product_name
            db.add(product)
            db.commit()
            db.refresh(product)
        return product

    product = Product(
        sku=sku,
        name=(product_name or sku).strip(),
        category="beverages",
        is_active=True,
    )
    db.add(product)
    db.commit()
    db.refresh(product)
    return product


def _render_annotated(image: ImageRecord, db: Session) -> None:
    detections = (
        db.query(Detection)
        .filter(Detection.image_id == image.id)
        .order_by(Detection.id.asc())
        .all()
    )
    render_items = []
    for det in detections:
        _, name = _resolve_detection_name(det)
        render_items.append(
            RenderDetection(
                bbox=(det.bbox_x1, det.bbox_y1, det.bbox_x2, det.bbox_y2),
                product_name=name,
                confidence=det.confidence,
                status=det.status,
            )
        )

    output_path = ANNOTATED_DIR / f"{image.id}.jpg"
    annotate_image(Path(image.stored_path), output_path, render_items)
    image.annotated_path = str(output_path)
    image.status = "completed"
    db.add(image)
    db.commit()


@app.get("/api/products", response_model=list[ProductOut])
def list_products(db: Session = Depends(get_db)) -> list[ProductOut]:
    products = db.query(Product).order_by(Product.sku.asc()).all()
    return [_to_product_out(item) for item in products]


@app.post("/api/products", response_model=ProductOut, status_code=201)
def create_product(payload: ProductCreate, db: Session = Depends(get_db)) -> ProductOut:
    product = Product(
        sku=payload.sku.strip(),
        name=payload.name.strip(),
        category=payload.category.strip(),
        is_active=payload.is_active,
    )
    db.add(product)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="SKU already exists.") from exc
    db.refresh(product)
    return _to_product_out(product)


@app.get("/api/system/inference")
def get_inference_status() -> dict[str, str]:
    return {
        "mode": INFERENCE_MODE,
        "engine": inference_engine.__class__.__name__,
        "clip_model": CLIP_MODEL_NAME,
        "backend": str(getattr(inference_engine, "backend", "unknown")),
        "detector_model_path": str(getattr(inference_engine, "detector_model_path", "") or ""),
    }


@app.post("/api/system/active-learning/export", response_model=ActiveLearningExportOut)
def export_active_learning(
    payload: ActiveLearningExportRequest | None = None,
    db: Session = Depends(get_db),
) -> ActiveLearningExportOut:
    request = payload or ActiveLearningExportRequest()
    ratio_total = request.train_ratio + request.val_ratio + request.test_ratio
    if abs(ratio_total - 1.0) > 1e-6:
        raise HTTPException(
            status_code=400,
            detail="train_ratio + val_ratio + test_ratio must equal 1.0.",
        )
    if request.val_ratio == 0.0 and request.test_ratio == 0.0:
        raise HTTPException(
            status_code=400,
            detail="At least one of val_ratio or test_ratio must be > 0.",
        )

    try:
        summary = export_active_learning_dataset(
            db,
            ActiveLearningExportOptions(
                export_name=request.export_name,
                corrected_only=request.corrected_only,
                min_confidence=request.min_confidence,
                train_ratio=request.train_ratio,
                val_ratio=request.val_ratio,
                test_ratio=request.test_ratio,
                include_recognition_crops=request.include_recognition_crops,
                detection_label_mode=request.detection_label_mode,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return ActiveLearningExportOut(**summary)


@app.get("/api/products/{sku}/reference-images", response_model=list[ProductReferenceImageOut])
def list_reference_images(sku: str, db: Session = Depends(get_db)) -> list[ProductReferenceImageOut]:
    product = db.query(Product).filter(Product.sku == sku).first()
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found.")

    files = _list_reference_images_for_sku(sku)
    return [
        ProductReferenceImageOut(
            sku=sku,
            file_name=path.name,
            file_path=str(path),
            image_url=f"/api/products/{sku}/reference-images/{path.name}",
        )
        for path in files
    ]


@app.get("/api/products/{sku}/reference-images/{file_name}")
def get_reference_image(
    sku: str,
    file_name: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    product = db.query(Product).filter(Product.sku == sku).first()
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found.")

    path = _reference_dir_for_sku(sku) / file_name
    if not path.exists() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Reference image not found.")
    return FileResponse(path)


@app.post("/api/products/{sku}/reference-images", response_model=ProductReferenceImageOut, status_code=201)
async def upload_reference_image(
    sku: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> ProductReferenceImageOut:
    product = db.query(Product).filter(Product.sku == sku).first()
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, and .png are allowed.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    directory = _reference_dir_for_sku(sku)
    directory.mkdir(parents=True, exist_ok=True)
    image_name = f"{uuid.uuid4().hex}{suffix}"
    path = directory / image_name
    path.write_bytes(raw)

    try:
        with Image.open(path):
            pass
    except (UnidentifiedImageError, OSError) as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    return ProductReferenceImageOut(
        sku=sku,
        file_name=image_name,
        file_path=str(path),
        image_url=f"/api/products/{sku}/reference-images/{image_name}",
    )


@app.post("/api/images", response_model=ImageUploadOut)
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)) -> ImageUploadOut:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, and .png are allowed.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image_id = uuid.uuid4().hex
    stored_path = UPLOAD_DIR / f"{image_id}{suffix}"
    stored_path.write_bytes(raw)

    try:
        with Image.open(stored_path) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    record = ImageRecord(
        id=image_id,
        filename=file.filename or stored_path.name,
        stored_path=str(stored_path),
        annotated_path=None,
        width=width,
        height=height,
        status="processing",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    products = db.query(Product).filter(Product.is_active.is_(True)).all()
    catalog = [
        CatalogItem(
            sku=item.sku,
            name=item.name,
            reference_images=tuple(str(path) for path in _list_reference_images_for_sku(item.sku)),
        )
        for item in products
    ]
    product_by_sku = {item.sku: item for item in products}

    candidates = inference_engine.infer(stored_path, image_id, catalog)
    for candidate in candidates:
        linked_product = product_by_sku.get(candidate.sku) if candidate.sku else None
        db.add(
            Detection(
                image_id=image_id,
                product_id=linked_product.id if linked_product else None,
                bbox_x1=candidate.bbox[0],
                bbox_y1=candidate.bbox[1],
                bbox_x2=candidate.bbox[2],
                bbox_y2=candidate.bbox[3],
                detection_confidence=candidate.detection_confidence,
                recognition_confidence=candidate.recognition_confidence,
                confidence=candidate.confidence,
                status=candidate.status,
                override_sku=candidate.sku,
                override_product_name=candidate.product_name,
            )
        )
    db.commit()

    _render_annotated(record, db)
    db.refresh(record)

    detections = (
        db.query(Detection)
        .filter(Detection.image_id == image_id)
        .order_by(Detection.id.asc())
        .all()
    )

    return ImageUploadOut(
        **_to_image_detail(record).model_dump(),
        detections=[_to_detection_out(item) for item in detections],
    )


@app.get("/api/images", response_model=ImageListOut)
def list_images(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> ImageListOut:
    query = db.query(ImageRecord)
    total = query.count()
    records = query.order_by(ImageRecord.created_at.desc()).offset(offset).limit(limit).all()
    return ImageListOut(
        items=[_to_image_summary(item) for item in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/api/images/{image_id}", response_model=ImageDetailOut)
def get_image(image_id: str, db: Session = Depends(get_db)) -> ImageDetailOut:
    image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    return _to_image_detail(image)


@app.get("/api/images/{image_id}/results", response_model=list[DetectionOut])
def get_image_results(image_id: str, db: Session = Depends(get_db)) -> list[DetectionOut]:
    image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    detections = (
        db.query(Detection)
        .filter(Detection.image_id == image_id)
        .order_by(Detection.id.asc())
        .all()
    )
    return [_to_detection_out(item) for item in detections]


@app.get("/api/images/{image_id}/annotated")
def get_annotated_image(image_id: str, db: Session = Depends(get_db)) -> FileResponse:
    image = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if image is None or not image.annotated_path:
        raise HTTPException(status_code=404, detail="Annotated image not found.")

    path = Path(image.annotated_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Annotated image file missing.")
    return FileResponse(path)


@app.patch("/api/detections/{detection_id}", response_model=DetectionOut)
def patch_detection(
    detection_id: int,
    payload: DetectionPatch,
    db: Session = Depends(get_db),
) -> DetectionOut:
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if detection is None:
        raise HTTPException(status_code=404, detail="Detection not found.")

    if payload.sku is None and payload.product_name is None and payload.confidence_override is None:
        raise HTTPException(status_code=400, detail="No correction payload provided.")

    if payload.sku is not None:
        sku = payload.sku.strip()
        product = _get_or_create_product_by_sku(db, sku, payload.product_name)
        detection.product_id = product.id
        detection.override_sku = product.sku

    if payload.product_name is not None:
        detection.override_product_name = payload.product_name.strip()

    if payload.confidence_override is not None:
        detection.confidence = round(payload.confidence_override, 2)

    detection.status = "corrected"
    db.add(detection)
    db.commit()
    db.refresh(detection)

    image = db.query(ImageRecord).filter(ImageRecord.id == detection.image_id).first()
    if image is not None:
        _render_annotated(image, db)

    return _to_detection_out(detection)


@app.post(
    "/api/detections/{detection_id}/save-reference",
    response_model=ProductReferenceImageOut,
    status_code=201,
)
def save_detection_as_reference(
    detection_id: int,
    payload: SaveDetectionReferencePayload | None = None,
    db: Session = Depends(get_db),
) -> ProductReferenceImageOut:
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if detection is None:
        raise HTTPException(status_code=404, detail="Detection not found.")

    image = db.query(ImageRecord).filter(ImageRecord.id == detection.image_id).first()
    if image is None:
        raise HTTPException(status_code=404, detail="Source image not found.")

    default_sku, default_name = _resolve_detection_name(detection)
    requested_sku = payload.sku.strip() if payload and payload.sku else default_sku
    requested_name = payload.product_name.strip() if payload and payload.product_name else default_name
    if not requested_sku:
        raise HTTPException(status_code=400, detail="Detection has no SKU. Provide one in request body.")

    product = _get_or_create_product_by_sku(db, requested_sku, requested_name)
    detection.product_id = product.id
    detection.override_sku = product.sku
    if requested_name:
        detection.override_product_name = requested_name
    detection.status = "corrected"
    db.add(detection)
    db.commit()
    db.refresh(detection)

    x1, y1, x2, y2 = detection.bbox_x1, detection.bbox_y1, detection.bbox_x2, detection.bbox_y2
    source_path = Path(image.stored_path)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source image file missing.")

    directory = _reference_dir_for_sku(product.sku)
    directory.mkdir(parents=True, exist_ok=True)
    file_name = f"{uuid.uuid4().hex}.jpg"
    output_path = directory / file_name
    with Image.open(source_path).convert("RGB") as source:
        crop = source.crop((x1, y1, x2, y2))
        crop.save(output_path, format="JPEG", quality=92)

    _render_annotated(image, db)

    return ProductReferenceImageOut(
        sku=product.sku,
        file_name=file_name,
        file_path=str(output_path),
        image_url=f"/api/products/{product.sku}/reference-images/{file_name}",
    )
