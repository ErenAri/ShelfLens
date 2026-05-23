# ShelfLens Data Source Plan

This project should use public data for generic product-box detection and your own
SKU photos for product recognition. Public datasets are not enough for 95% SKU
accuracy on a real store catalog because exact packaging, country variants, and
shelf conditions differ.

## Recommended Sources

| Source | Use | Notes |
| --- | --- | --- |
| SKU-110K | Generic shelf/product detector pretraining | Large grocery shelf detection dataset in Ultralytics format. Best first source for finding product boxes. |
| Your own shelf uploads | Detector fine-tuning | Correct boxes in ShelfLens, export corrected data, and fine-tune the detector. |
| Your own SKU reference photos | Recognition | Take 10-20 front/side/package-condition images per SKU. This is the main path to 95% SKU recognition. |
| Open Food Facts | Bootstrap reference images and product names by barcode | Useful for beverage/snack metadata, but images are inconsistent and should not replace your own references. |
| RPC / Products-10K | Research-only recognition experiments | Useful to test retrieval models. Do not assume their licenses or labels match your commercial SKU catalog. |
| D-FINE / RT-DETR | Future detector replacement | Strong open-source detectors. Keep the API stable so the detector can be swapped later. |

## Stage 1: Pretrain Generic Product Detector

Install ML dependencies:

```powershell
cd C:\Users\erena\Desktop\ShelfLens\backend
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ml.txt
```

Train on SKU-110K through the Ultralytics dataset alias:

```powershell
python scripts\train_detector_yolo.py --dataset-yaml SKU-110K.yaml --datasets-dir .\data\external --model yolov8s.pt --epochs 80 --imgsz 640 --batch 8 --device 0 --workers 4 --run-name sku110k_product_detector
```

On an RTX 2060, reduce `--batch` to `4` if CUDA memory is not enough.
If Ultralytics reports a stale global path such as `C:\projects\yolo-trt\datasets`,
the `--datasets-dir .\data\external` argument forces downloads into this project.

## Stage 2: Fine-Tune On ShelfLens Corrections

Correct detections in the app first. Then export only corrected boxes:

```powershell
cd C:\Users\erena\Desktop\ShelfLens\backend
python scripts\export_active_learning_dataset.py --export-name shelf_corrected_01 --corrected-only --min-confidence 0.0 --detection-label-mode product
```

Fine-tune from the SKU-110K-pretrained detector:

```powershell
python scripts\train_detector_yolo.py --dataset-yaml .\data\exports\shelf_corrected_01\detection\dataset.yaml --model .\data\models\sku110k_product_detector\weights\best.pt --epochs 60 --imgsz 640 --batch 8 --device 0 --workers 4 --run-name shelflens_product_detector_01
```

Evaluate:

```powershell
python scripts\report_model_metrics.py --run-dir .\data\models\shelflens_product_detector_01 --export-dir .\data\exports\shelf_corrected_01 --target 0.95
```

## Stage 3: Import SKU Reference Images

Create a CSV template:

```powershell
python scripts\import_reference_catalog.py --write-template .\data\sources\beverage_references.csv
```

Edit the CSV. Each row can create/update a product and optionally attach a local
image path or image URL:

```csv
sku,name,category,image_path,image_url
bev_001,Coca-Cola 330ml,beverages,C:\Users\erena\Pictures\sku_refs\coke_front.jpg,
bev_002,Pepsi 330ml,beverages,,https://example.com/pepsi_front.jpg
```

Import:

```powershell
python scripts\import_reference_catalog.py --csv .\data\sources\beverage_references.csv
```

Add extra front-image variants from the same Open Food Facts product pages:

```powershell
python scripts\import_openfoodfacts_variants.py --csv .\data\sources\beverage_references.csv --max-per-sku 5
python scripts\report_reference_catalog.py --target-per-sku 5
```

Only selected front images are imported. Ingredient, nutrition, and packaging
panels are intentionally excluded because they hurt front-package recognition.

## Stage 4: Bootstrap From Open Food Facts

If you have barcodes, create:

```csv
barcode,sku,name,category
5449000000996,bev_001,Coca-Cola 330ml,beverages
```

Fetch metadata and front images:

```powershell
python scripts\fetch_openfoodfacts_references.py --csv .\data\sources\openfoodfacts_barcodes.csv --country world
```

Preview without writing:

```powershell
python scripts\fetch_openfoodfacts_references.py --barcode 5449000000996 --dry-run
```

## Stage 5: Run The App With The Real Detector

```powershell
cd C:\Users\erena\Desktop\ShelfLens\backend
$env:SHELFLENS_INFERENCE_MODE="real"
$env:SHELFLENS_DETECTOR_MODEL_PATH="C:\Users\erena\Desktop\ShelfLens\backend\data\models\shelflens_product_detector_01\weights\best.pt"
$env:SHELFLENS_MIN_DETECTION_CONFIDENCE="0.05"
$env:SHELFLENS_MAX_DETECTIONS="80"
$env:SHELFLENS_MIN_RECOGNITION_MARGIN="0.04"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

For the current best generic SKU-110K detector, use:

```powershell
$env:SHELFLENS_DETECTOR_MODEL_PATH="C:\Users\erena\Desktop\ShelfLens\backend\data\models\sku110k_product_detector_tune1_clean_20260523_1458-2\weights\best.pt"
```

## Source Links

- SKU-110K Ultralytics docs: https://docs.ultralytics.com/datasets/detect/sku-110k/
- Open Food Facts API: https://openfoodfacts.github.io/openfoodfacts-server/api/
- RPC dataset mirror: https://huggingface.co/datasets/benjamintli/retail-product-checkout
- Products-10K: https://products-10k.github.io/
- D-FINE: https://github.com/Peterande/D-FINE
- RT-DETR: https://github.com/lyuwenyu/RT-DETR
