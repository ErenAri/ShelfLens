from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import EXPORTS_DIR, EXTERNAL_DATA_DIR, MODELS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a YOLO detector from an exported ShelfLens dataset or an "
            "Ultralytics dataset alias such as SKU-110K.yaml."
        )
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default=None,
        help=(
            "Path to detection dataset.yaml. If omitted, latest ShelfLens export is used. "
            "You can also pass an Ultralytics alias such as SKU-110K.yaml."
        ),
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO base model.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size. Omit for Ultralytics default.")
    parser.add_argument("--device", type=str, default=None, help="Device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--workers", type=int, default=None, help="Data loader workers.")
    parser.add_argument("--patience", type=int, default=None, help="Early-stopping patience.")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=EXTERNAL_DATA_DIR,
        help="Where Ultralytics should download public datasets such as SKU-110K.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="shelflens_mvp",
        help="Run name under models directory.",
    )
    return parser


def _looks_like_ultralytics_alias(value: str) -> bool:
    normalized = value.strip()
    return (
        normalized.lower().endswith((".yaml", ".yml"))
        and "/" not in normalized
        and "\\" not in normalized
        and not Path(normalized).exists()
    )


def resolve_dataset_yaml(explicit_yaml: str | None) -> Path | str:
    if explicit_yaml is not None:
        explicit_value = explicit_yaml.strip()
        if _looks_like_ultralytics_alias(explicit_value):
            return explicit_value
        return Path(explicit_value)

    candidates = sorted(
        EXPORTS_DIR.glob("*/detection/dataset.yaml"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No exported detection dataset found under data/exports. "
            "Upload images first (to create detections), then run "
            "'python scripts/export_active_learning_dataset.py --min-confidence 0.0'."
        )
    return candidates[0]


def normalize_dataset_yaml_for_ultralytics(dataset_yaml: Path) -> Path:
    try:
        import yaml
    except ImportError:
        return dataset_yaml

    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return dataset_yaml

    changed = False
    yaml_parent = dataset_yaml.parent.resolve()
    dataset_root_raw = data.get("path")

    if not dataset_root_raw:
        data["path"] = yaml_parent.as_posix()
        changed = True
    else:
        dataset_root_path = Path(str(dataset_root_raw))
        if not dataset_root_path.is_absolute():
            data["path"] = (yaml_parent / dataset_root_path).resolve().as_posix()
            changed = True

    for key, default_value in (
        ("train", "images/train"),
        ("val", "images/val"),
        ("test", "images/test"),
    ):
        if key not in data:
            data[key] = default_value
            changed = True

    val_relative = Path(str(data["val"]))
    root_path = Path(str(data["path"]))
    resolved_val = val_relative if val_relative.is_absolute() else (root_path / val_relative).resolve()
    fallback_val = (yaml_parent / "images" / "val").resolve()
    if not resolved_val.exists() and fallback_val.exists():
        data["path"] = yaml_parent.as_posix()
        changed = True

    if not changed:
        return dataset_yaml

    fixed_yaml = dataset_yaml.parent / "dataset.ultralytics.yaml"
    fixed_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return fixed_yaml


def configure_ultralytics_datasets_dir(datasets_dir: Path) -> Path:
    resolved = datasets_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import settings

        current_value = str(settings.get("datasets_dir", ""))
        if Path(current_value).resolve() != resolved:
            settings.update({"datasets_dir": str(resolved)})
    except Exception as exc:  # noqa: BLE001 - explicit local YAML training can still continue.
        print(f"Warning: could not update Ultralytics datasets_dir: {exc}")
    return resolved


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_yaml = resolve_dataset_yaml(args.dataset_yaml)
    if isinstance(dataset_yaml, Path) and not dataset_yaml.exists():
        parser.error(f"Dataset YAML not found: {dataset_yaml}")
    if isinstance(dataset_yaml, Path):
        dataset_yaml = normalize_dataset_yaml_for_ultralytics(dataset_yaml)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install ultralytics first (`pip install ultralytics`)."
        ) from exc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    datasets_dir = configure_ultralytics_datasets_dir(args.datasets_dir)
    model = YOLO(args.model)
    train_args: dict[str, Any] = {
        "data": str(dataset_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "project": str(MODELS_DIR),
        "name": args.run_name,
    }
    for key in ("batch", "device", "workers", "patience"):
        value = getattr(args, key)
        if value is not None:
            train_args[key] = value

    print(f"Training detector with dataset: {dataset_yaml}")
    model.train(**train_args)
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", MODELS_DIR / args.run_name)
    print(f"Training complete. Artifacts saved under: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
