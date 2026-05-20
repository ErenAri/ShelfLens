import type {
  ActiveLearningExportRequest,
  ActiveLearningExportResult,
  Detection,
  DetectionPatchPayload,
  InferenceStatus,
  ImageDetail,
  ImageListResponse,
  ImageUpload,
  Product,
  ProductReferenceImage,
  SaveDetectionReferencePayload,
} from './types'

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

function toUrl(path: string): string {
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path
  }
  return `${API_BASE_URL}${path}`
}

export function toAbsoluteUrl(path: string | null): string {
  return path ? toUrl(path) : ''
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(toUrl(path), init)
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) {
        message = body.detail
      }
    } catch {
      // Ignore response parsing errors for non-JSON payloads.
    }
    throw new Error(message)
  }
  return (await response.json()) as T
}

export function listProducts(): Promise<Product[]> {
  return request<Product[]>('/api/products')
}

export function getInferenceStatus(): Promise<InferenceStatus> {
  return request<InferenceStatus>('/api/system/inference')
}

export function createProduct(payload: {
  sku: string
  name: string
  category?: string
  is_active?: boolean
}): Promise<Product> {
  return request<Product>('/api/products', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function listImages(limit = 20, offset = 0): Promise<ImageListResponse> {
  return request<ImageListResponse>(`/api/images?limit=${limit}&offset=${offset}`)
}

export function getImage(imageId: string): Promise<ImageDetail> {
  return request<ImageDetail>(`/api/images/${imageId}`)
}

export function getImageResults(imageId: string): Promise<Detection[]> {
  return request<Detection[]>(`/api/images/${imageId}/results`)
}

export function uploadImage(file: File): Promise<ImageUpload> {
  const formData = new FormData()
  formData.append('file', file)
  return request<ImageUpload>('/api/images', {
    method: 'POST',
    body: formData,
  })
}

export function patchDetection(
  detectionId: number,
  payload: DetectionPatchPayload,
): Promise<Detection> {
  return request<Detection>(`/api/detections/${detectionId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function listReferenceImages(sku: string): Promise<ProductReferenceImage[]> {
  return request<ProductReferenceImage[]>(`/api/products/${encodeURIComponent(sku)}/reference-images`)
}

export function uploadReferenceImage(sku: string, file: File): Promise<ProductReferenceImage> {
  const formData = new FormData()
  formData.append('file', file)
  return request<ProductReferenceImage>(
    `/api/products/${encodeURIComponent(sku)}/reference-images`,
    {
      method: 'POST',
      body: formData,
    },
  )
}

export function saveDetectionAsReference(
  detectionId: number,
  payload?: SaveDetectionReferencePayload,
): Promise<ProductReferenceImage> {
  return request<ProductReferenceImage>(
    `/api/detections/${detectionId}/save-reference`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload ?? {}),
    },
  )
}

export function exportActiveLearningDataset(
  payload?: ActiveLearningExportRequest,
): Promise<ActiveLearningExportResult> {
  return request<ActiveLearningExportResult>('/api/system/active-learning/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
  })
}
