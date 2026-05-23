from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import ALLOWED_EXTENSIONS, REFERENCE_DIR, ensure_storage_dirs
from app.database import Base, SessionLocal, engine
from app.models import Product
from app.seed import seed_products


USER_AGENT = "ShelfLens-MVP/0.1 (+local development)"


@dataclass
class CatalogImportStats:
    products_created: int = 0
    products_updated: int = 0
    reference_images_saved: int = 0
    reference_images_skipped: int = 0
    errors: list[str] = field(default_factory=list)


def open_session(seed_default_products: bool = True) -> Session:
    ensure_storage_dirs()
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    if seed_default_products:
        seed_products(db)
    return db


def upsert_product(
    db: Session,
    sku: str,
    name: str,
    category: str = "beverages",
) -> tuple[Product, bool]:
    normalized_sku = sku.strip()
    normalized_name = name.strip() or normalized_sku
    normalized_category = category.strip() or "beverages"
    product = db.query(Product).filter(Product.sku == normalized_sku).first()
    if product is None:
        product = Product(
            sku=normalized_sku,
            name=normalized_name,
            category=normalized_category,
            is_active=True,
        )
        db.add(product)
        db.commit()
        db.refresh(product)
        return product, True

    changed = False
    if product.name != normalized_name:
        product.name = normalized_name
        changed = True
    if product.category != normalized_category:
        product.category = normalized_category
        changed = True
    if not product.is_active:
        product.is_active = True
        changed = True
    if changed:
        db.add(product)
        db.commit()
        db.refresh(product)
    return product, False


def read_local_image_bytes(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"Image path not found: {path}")
    return path.read_bytes()


def download_image_bytes(url: str, timeout_seconds: int = 30) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
    except URLError as exc:
        raise RuntimeError(f"Failed to download image: {url}") from exc


def file_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or "reference.jpg"


def save_reference_image(
    sku: str,
    raw: bytes,
    filename_hint: str,
) -> tuple[Path, bool]:
    suffix = Path(filename_hint).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".jpg"

    digest = sha256(raw).hexdigest()[:16]
    output_dir = REFERENCE_DIR / sku
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{digest}{suffix}"
    if output_path.exists():
        return output_path, False

    output_path.write_bytes(raw)
    try:
        with Image.open(output_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"Invalid image for SKU {sku}: {filename_hint}") from exc

    return output_path, True
