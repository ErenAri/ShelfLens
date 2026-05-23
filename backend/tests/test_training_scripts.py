from __future__ import annotations

from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from scripts.train_detector_yolo import resolve_dataset_yaml  # noqa: E402


def test_train_script_accepts_ultralytics_dataset_alias() -> None:
    assert resolve_dataset_yaml("SKU-110K.yaml") == "SKU-110K.yaml"


def test_train_script_resolves_explicit_local_path() -> None:
    path = resolve_dataset_yaml(r".\data\exports\example\detection\dataset.yaml")
    assert isinstance(path, Path)
    assert path.as_posix().endswith("data/exports/example/detection/dataset.yaml")
