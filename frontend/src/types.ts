export type DetectionStatus =
  | 'recognized'
  | 'low_confidence'
  | 'unknown_product'
  | 'corrected'

export interface Product {
  id: number
  sku: string
  name: string
  category: string
  is_active: boolean
  created_at: string
  reference_image_count: number
}

export interface Detection {
  id: number
  image_id: string
  bbox: [number, number, number, number]
  sku: string | null
  product_name: string | null
  detection_confidence: number
  recognition_confidence: number
  confidence: number
  status: DetectionStatus
  created_at: string
  updated_at: string
}

export interface ImageSummary {
  id: string
  filename: string
  status: string
  width: number
  height: number
  created_at: string
  annotated_url: string | null
}

export interface ImageDetail extends ImageSummary {
  stored_path: string
  annotated_path: string | null
}

export interface ImageUpload extends ImageDetail {
  detections: Detection[]
}

export interface ImageListResponse {
  items: ImageSummary[]
  total: number
  limit: number
  offset: number
}

export interface DetectionPatchPayload {
  sku?: string
  product_name?: string
  confidence_override?: number
}

export interface InferenceStatus {
  mode: string
  engine: string
  clip_model: string
}

export interface ProductReferenceImage {
  sku: string
  file_name: string
  file_path: string
  image_url: string
}

export interface SaveDetectionReferencePayload {
  sku?: string
  product_name?: string
}

export interface ActiveLearningExportRequest {
  export_name?: string
  corrected_only?: boolean
  min_confidence?: number
  train_ratio?: number
  val_ratio?: number
  test_ratio?: number
  include_recognition_crops?: boolean
}

export interface ActiveLearningExportSection {
  enabled: boolean
  root_path: string | null
  image_count: number
  annotation_count: number
  split_counts: Record<string, number>
  annotation_split_counts?: Record<string, number> | null
  dataset_yaml_path?: string | null
}

export interface ActiveLearningExportResult {
  export_name: string
  export_path: string
  generated_at: string
  manifest_path: string
  corrected_only: boolean
  min_confidence: number
  train_ratio: number
  val_ratio: number
  test_ratio: number
  include_recognition_crops: boolean
  total_detections_scanned: number
  total_detections_exported: number
  corrected_detections_exported: number
  skipped_no_sku: number
  skipped_missing_image: number
  skipped_invalid_box: number
  class_names: string[]
  detection: ActiveLearningExportSection
  recognition: ActiveLearningExportSection
}
