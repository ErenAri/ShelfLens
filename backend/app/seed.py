from __future__ import annotations

from sqlalchemy.orm import Session

from .models import Product


SEED_BEVERAGES = [
    "Cola Classic 330ml",
    "Cola Zero 330ml",
    "Orange Soda 330ml",
    "Lemon-Lime Soda 330ml",
    "Sparkling Water Lime 500ml",
    "Sparkling Water Plain 500ml",
    "Iced Tea Lemon 500ml",
    "Iced Tea Peach 500ml",
    "Energy Drink Original 250ml",
    "Energy Drink Sugar Free 250ml",
    "Sports Drink Blue 500ml",
    "Sports Drink Orange 500ml",
    "Cold Brew Coffee 250ml",
    "Latte Coffee 250ml",
    "Fruit Juice Apple 1L",
    "Fruit Juice Orange 1L",
    "Mineral Water 500ml",
    "Mineral Water 1.5L",
    "Chocolate Milk 250ml",
    "Protein Shake Vanilla 330ml",
]


def seed_products(db: Session) -> None:
    existing = db.query(Product).count()
    if existing > 0:
        return

    for index, name in enumerate(SEED_BEVERAGES, start=1):
        db.add(
            Product(
                sku=f"bev_{index:03d}",
                name=name,
                category="beverages",
                is_active=True,
            )
        )

    db.commit()

