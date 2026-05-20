import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const productsPayload = [
  {
    id: 1,
    sku: 'bev_001',
    name: 'Cola Classic 330ml',
    category: 'beverages',
    is_active: true,
    created_at: '2026-05-20T12:00:00Z',
    reference_image_count: 0,
  },
]

const emptyHistoryPayload = {
  items: [],
  total: 0,
  limit: 20,
  offset: 0,
}

const uploadPayload = {
  id: 'image-1',
  filename: 'shelf.png',
  status: 'completed',
  width: 640,
  height: 360,
  created_at: '2026-05-20T12:00:00Z',
  annotated_url: '/api/images/image-1/annotated',
  stored_path: 'stored/path',
  annotated_path: 'annotated/path',
  detections: [
    {
      id: 11,
      image_id: 'image-1',
      bbox: [10, 12, 120, 200],
      sku: 'bev_001',
      product_name: 'Cola Classic 330ml',
      detection_confidence: 0.88,
      recognition_confidence: 0.84,
      confidence: 0.86,
      status: 'recognized',
      created_at: '2026-05-20T12:00:00Z',
      updated_at: '2026-05-20T12:00:00Z',
    },
  ],
}

const inferencePayload = {
  mode: 'mock',
  engine: 'MockInferenceEngine',
  clip_model: 'sentence-transformers/clip-ViT-B-32',
}

const exportPayload = {
  export_name: 'export_20260520_120000',
  export_path: 'C:/tmp/export_20260520_120000',
  generated_at: '2026-05-20T12:00:00Z',
  manifest_path: 'C:/tmp/export_20260520_120000/manifest.json',
  corrected_only: false,
  min_confidence: 0.6,
  train_ratio: 0.7,
  val_ratio: 0.2,
  test_ratio: 0.1,
  include_recognition_crops: true,
  total_detections_scanned: 5,
  total_detections_exported: 4,
  corrected_detections_exported: 1,
  skipped_no_sku: 1,
  skipped_missing_image: 0,
  skipped_invalid_box: 0,
  class_names: ['bev_001'],
  detection: {
    enabled: true,
    root_path: 'C:/tmp/export_20260520_120000/detection',
    image_count: 2,
    annotation_count: 4,
    split_counts: { train: 1, val: 1, test: 0 },
    annotation_split_counts: { train: 2, val: 2, test: 0 },
    dataset_yaml_path: 'C:/tmp/export_20260520_120000/detection/dataset.yaml',
  },
  recognition: {
    enabled: true,
    root_path: 'C:/tmp/export_20260520_120000/recognition',
    image_count: 4,
    annotation_count: 4,
    split_counts: { train: 2, val: 2, test: 0 },
    annotation_split_counts: null,
    dataset_yaml_path: null,
  },
}

describe('App', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('renders annotated image and detections after successful upload', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(productsPayload))
      .mockResolvedValueOnce(jsonResponse(emptyHistoryPayload))
      .mockResolvedValueOnce(jsonResponse(inferencePayload))
      .mockResolvedValueOnce(jsonResponse(uploadPayload))
      .mockResolvedValueOnce(jsonResponse(emptyHistoryPayload))

    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<App />)

    const input = screen.getByLabelText(/upload image/i)
    const file = new File(['image-bytes'], 'shelf.png', { type: 'image/png' })
    await user.upload(input, file)
    await user.click(screen.getByRole('button', { name: /run recognition/i }))

    expect(await screen.findByRole('img', { name: /annotated upload/i })).toBeInTheDocument()
    expect(screen.getByText('recognized')).toBeInTheDocument()
  })

  it('shows error message when upload fails', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(productsPayload))
      .mockResolvedValueOnce(jsonResponse(emptyHistoryPayload))
      .mockResolvedValueOnce(jsonResponse(inferencePayload))
      .mockResolvedValueOnce(jsonResponse({ detail: 'Upload failed hard.' }, 500))

    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<App />)

    const input = screen.getByLabelText(/upload image/i)
    const file = new File(['image-bytes'], 'shelf.png', { type: 'image/png' })
    await user.upload(input, file)
    await user.click(screen.getByRole('button', { name: /run recognition/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Upload failed hard.')
  })

  it('applies correction and updates row state without reload', async () => {
    const correctedDetection = {
      ...uploadPayload.detections[0],
      sku: 'bev_002',
      product_name: 'Orange Soda 330ml',
      confidence: 0.91,
      status: 'corrected',
    }

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(productsPayload))
      .mockResolvedValueOnce(jsonResponse(emptyHistoryPayload))
      .mockResolvedValueOnce(jsonResponse(inferencePayload))
      .mockResolvedValueOnce(jsonResponse(uploadPayload))
      .mockResolvedValueOnce(jsonResponse(emptyHistoryPayload))
      .mockResolvedValueOnce(jsonResponse(correctedDetection))
      .mockResolvedValueOnce(jsonResponse(emptyHistoryPayload))

    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<App />)

    const input = screen.getByLabelText(/upload image/i)
    await user.upload(input, new File(['bytes'], 'shelf.png', { type: 'image/png' }))
    await user.click(screen.getByRole('button', { name: /run recognition/i }))

    const skuField = await screen.findByLabelText('SKU 11')
    const productField = screen.getByLabelText('Product 11')
    const confidenceField = screen.getByLabelText('Confidence 11')

    await user.clear(skuField)
    await user.type(skuField, 'bev_002')
    await user.clear(productField)
    await user.type(productField, 'Orange Soda 330ml')
    await user.clear(confidenceField)
    await user.type(confidenceField, '0.91')
    await user.click(screen.getByRole('button', { name: /apply correction/i }))

    expect(await screen.findByText('Orange Soda 330ml')).toBeInTheDocument()
    expect(screen.getByText('corrected')).toBeInTheDocument()

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/api/detections/11'),
        expect.objectContaining({ method: 'PATCH' }),
      )
    })
  })

  it('saves detection crop as reference image', async () => {
    const saveReferencePayload = {
      sku: 'bev_001',
      file_name: 'saved.jpg',
      file_path: 'dummy/path',
      image_url: '/api/products/bev_001/reference-images/saved.jpg',
    }
    const refreshedProducts = [{ ...productsPayload[0], reference_image_count: 1 }]

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(productsPayload))
      .mockResolvedValueOnce(jsonResponse(emptyHistoryPayload))
      .mockResolvedValueOnce(jsonResponse(inferencePayload))
      .mockResolvedValueOnce(jsonResponse(uploadPayload))
      .mockResolvedValueOnce(jsonResponse(emptyHistoryPayload))
      .mockResolvedValueOnce(jsonResponse(saveReferencePayload))
      .mockResolvedValueOnce(jsonResponse(refreshedProducts))
      .mockResolvedValueOnce(jsonResponse(uploadPayload.detections))

    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<App />)
    const input = screen.getByLabelText(/upload image/i)
    await user.upload(input, new File(['bytes'], 'shelf.png', { type: 'image/png' }))
    await user.click(screen.getByRole('button', { name: /run recognition/i }))

    await user.click(screen.getByRole('button', { name: /save as reference/i }))

    expect(await screen.findByText(/saved reference from detection/i)).toBeInTheDocument()
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/api/detections/11/save-reference'),
        expect.objectContaining({ method: 'POST' }),
      )
    })
  })

  it('exports active-learning dataset', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(productsPayload))
      .mockResolvedValueOnce(jsonResponse(emptyHistoryPayload))
      .mockResolvedValueOnce(jsonResponse(inferencePayload))
      .mockResolvedValueOnce(jsonResponse(exportPayload))

    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()

    render(<App />)
    await user.click(screen.getByRole('button', { name: /export active-learning dataset/i }))

    expect(await screen.findByText(/export export_20260520_120000 ready/i)).toBeInTheDocument()
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/api/system/active-learning/export'),
        expect.objectContaining({ method: 'POST' }),
      )
    })
  })
})
