import { useEffect, useMemo, useState } from 'react'
import {
  exportActiveLearningDataset,
  getInferenceStatus,
  getImage,
  getImageResults,
  listImages,
  listProducts,
  patchDetection,
  saveDetectionAsReference,
  uploadReferenceImage,
  toAbsoluteUrl,
  uploadImage,
} from './api'
import './App.css'
import type {
  ActiveLearningExportResult,
  Detection,
  DetectionPatchPayload,
  ImageDetail,
  ImageSummary,
  InferenceStatus,
  Product,
} from './types'

type DraftMap = Record<
  number,
  {
    sku: string
    productName: string
    confidence: string
  }
>

function App() {
  const [products, setProducts] = useState<Product[]>([])
  const [history, setHistory] = useState<ImageSummary[]>([])
  const [selectedImage, setSelectedImage] = useState<ImageDetail | null>(null)
  const [detections, setDetections] = useState<Detection[]>([])
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedReferenceSku, setSelectedReferenceSku] = useState<string>('')
  const [referenceFile, setReferenceFile] = useState<File | null>(null)
  const [uploadingReference, setUploadingReference] = useState(false)
  const [referenceMessage, setReferenceMessage] = useState<string>('')
  const [uploading, setUploading] = useState(false)
  const [exportingDataset, setExportingDataset] = useState(false)
  const [exportMessage, setExportMessage] = useState<string>('')
  const [lastExport, setLastExport] = useState<ActiveLearningExportResult | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [inferenceStatus, setInferenceStatus] = useState<InferenceStatus | null>(null)
  const [drafts, setDrafts] = useState<DraftMap>({})

  useEffect(() => {
    async function bootstrap() {
      try {
        const [productList, imageList, status] = await Promise.all([
          listProducts(),
          listImages(),
          getInferenceStatus(),
        ])
        setProducts(productList)
        setHistory(imageList.items)
        setInferenceStatus(status)
        if (productList.length > 0) {
          setSelectedReferenceSku(productList[0].sku)
        }
      } catch (error) {
        setErrorMessage(error instanceof Error ? error.message : 'Failed to load dashboard data.')
      }
    }

    void bootstrap()
  }, [])

  function buildDefaultDraft(detection: Detection): DraftMap[number] {
    return {
      sku: detection.sku ?? '',
      productName: detection.product_name ?? '',
      confidence: '',
    }
  }

  function getDraft(detection: Detection): DraftMap[number] {
    return drafts[detection.id] ?? buildDefaultDraft(detection)
  }

  async function refreshHistory() {
    const list = await listImages()
    setHistory(list.items)
  }

  async function refreshProducts() {
    const productList = await listProducts()
    setProducts(productList)
    if (!productList.find((item) => item.sku === selectedReferenceSku)) {
      setSelectedReferenceSku(productList[0]?.sku ?? '')
    }
  }

  async function onUpload() {
    if (!selectedFile) {
      return
    }

    setErrorMessage(null)
    setUploading(true)
    try {
      const payload = await uploadImage(selectedFile)
      setSelectedImage(payload)
      setDetections(payload.detections)
      setDrafts({})
      await refreshHistory()
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Upload failed.')
    } finally {
      setUploading(false)
    }
  }

  async function onUploadReferenceImage() {
    if (!selectedReferenceSku || !referenceFile) {
      return
    }

    setReferenceMessage('')
    setUploadingReference(true)
    try {
      await uploadReferenceImage(selectedReferenceSku, referenceFile)
      await refreshProducts()
      setReferenceMessage(`Reference image saved for ${selectedReferenceSku}.`)
      setReferenceFile(null)
    } catch (error) {
      setReferenceMessage(error instanceof Error ? error.message : 'Failed to upload reference image.')
    } finally {
      setUploadingReference(false)
    }
  }

  async function onSelectHistory(imageId: string) {
    setErrorMessage(null)
    try {
      const [image, imageDetections] = await Promise.all([
        getImage(imageId),
        getImageResults(imageId),
      ])
      setSelectedImage(image)
      setDetections(imageDetections)
      setDrafts({})
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Failed to load image results.')
    }
  }

  async function onExportDataset() {
    setErrorMessage(null)
    setExportMessage('')
    setExportingDataset(true)
    try {
      const result = await exportActiveLearningDataset({
        corrected_only: false,
        min_confidence: 0.6,
        train_ratio: 0.7,
        val_ratio: 0.2,
        test_ratio: 0.1,
        include_recognition_crops: true,
        detection_label_mode: 'product',
      })
      setLastExport(result)
      setExportMessage(
        `Export ${result.export_name} ready (${result.total_detections_exported} labeled detections).`,
      )
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : 'Failed to export active-learning dataset.',
      )
    } finally {
      setExportingDataset(false)
    }
  }

  function onDraftChange(detectionId: number, key: keyof DraftMap[number], value: string) {
    setDrafts((current) => ({
      ...current,
      [detectionId]: {
        ...(current[detectionId] ?? { sku: '', productName: '', confidence: '' }),
        [key]: value,
      },
    }))
  }

  async function onApplyCorrection(detection: Detection) {
    const draft = getDraft(detection)
    const payload: DetectionPatchPayload = {}

    if (draft.sku.trim() && draft.sku.trim() !== (detection.sku ?? '')) {
      payload.sku = draft.sku.trim()
    }
    if (
      draft.productName.trim() &&
      draft.productName.trim() !== (detection.product_name ?? '')
    ) {
      payload.product_name = draft.productName.trim()
    }
    if (draft.confidence.trim()) {
      const numeric = Number.parseFloat(draft.confidence)
      if (!Number.isNaN(numeric)) {
        payload.confidence_override = numeric
      }
    }

    if (Object.keys(payload).length === 0) {
      return
    }

    setErrorMessage(null)
    try {
      const corrected = await patchDetection(detection.id, payload)
      setDetections((current) =>
        current.map((item) => (item.id === corrected.id ? corrected : item)),
      )
      setDrafts((current) => {
        const next = { ...current }
        delete next[detection.id]
        return next
      })
      await refreshHistory()
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Failed to save correction.')
    }
  }

  async function onSaveDetectionReference(detection: Detection) {
    setErrorMessage(null)
    const draft = getDraft(detection)
    try {
      const saved = await saveDetectionAsReference(detection.id, {
        sku: draft.sku.trim() || undefined,
        product_name: draft.productName.trim() || undefined,
      })
      setReferenceMessage(`Saved reference from detection for ${saved.sku}.`)
      await refreshProducts()
      if (selectedImage) {
        const updatedDetections = await getImageResults(selectedImage.id)
        setDetections(updatedDetections)
      }
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : 'Failed to save detection as reference image.',
      )
    }
  }

  const annotatedImageUrl = useMemo(() => {
    return selectedImage?.annotated_url ? toAbsoluteUrl(selectedImage.annotated_url) : ''
  }, [selectedImage])

  return (
    <div className="app-shell">
      <header className="top-strip">
        <h1>ShelfLens MVP</h1>
        <p>Uploaded-image product recognition for beverage shelves.</p>
      </header>

      <main className="layout-grid">
        <section className="panel">
          <h2>Upload Image</h2>
          <label htmlFor="image-upload" className="field-label">
            Upload image
          </label>
          <input
            id="image-upload"
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            aria-label="Upload image"
            onChange={(event) => {
              setSelectedFile(event.target.files?.[0] ?? null)
            }}
          />
          <button type="button" onClick={() => void onUpload()} disabled={!selectedFile || uploading}>
            {uploading ? 'Processing...' : 'Run Recognition'}
          </button>
          {errorMessage ? (
            <p className="error-text" role="alert">
              {errorMessage}
            </p>
          ) : null}
        </section>

        <section className="panel">
          <h2>Catalog</h2>
          <p className="muted-text">{products.length} SKUs loaded</p>
          {inferenceStatus ? (
            <div className="muted-text">
              <p>
                Inference: <strong>{inferenceStatus.mode}</strong> ({inferenceStatus.engine})
              </p>
              {inferenceStatus.backend ? (
                <p>
                  Backend: <strong>{inferenceStatus.backend}</strong>
                </p>
              ) : null}
              {inferenceStatus.detector_model_path ? (
                <p>Detector model: {inferenceStatus.detector_model_path}</p>
              ) : null}
            </div>
          ) : null}
          <div className="reference-uploader">
            <label htmlFor="reference-sku">Reference SKU</label>
            <select
              id="reference-sku"
              value={selectedReferenceSku}
              onChange={(event) => setSelectedReferenceSku(event.target.value)}
            >
              {products.map((item) => (
                <option key={item.id} value={item.sku}>
                  {item.sku}
                </option>
              ))}
            </select>
            <input
              id="reference-upload"
              type="file"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              aria-label="Upload reference image"
              onChange={(event) => setReferenceFile(event.target.files?.[0] ?? null)}
            />
            <button
              type="button"
              disabled={!selectedReferenceSku || !referenceFile || uploadingReference}
              onClick={() => void onUploadReferenceImage()}
            >
              {uploadingReference ? 'Saving reference...' : 'Add Reference Image'}
            </button>
            {referenceMessage ? <p className="muted-text">{referenceMessage}</p> : null}
          </div>
          <ul className="catalog-list">
            {products.slice(0, 20).map((item) => (
              <li key={item.id}>
                <strong>{item.sku}</strong>
                <span>
                  {item.name} ({item.reference_image_count})
                </span>
              </li>
            ))}
          </ul>
        </section>
      </main>

      <section className="panel wide-panel">
        <h2>Result Preview</h2>
        {selectedImage && annotatedImageUrl ? (
          <img
            src={annotatedImageUrl}
            alt="Annotated upload"
            className="annotated-image"
          />
        ) : (
          <p className="muted-text">Upload an image to view annotated output.</p>
        )}
      </section>

      <section className="panel wide-panel">
        <h2>Detections</h2>
        {detections.length === 0 ? (
          <p className="muted-text">No detections yet.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Product</th>
                <th>SKU</th>
                <th>Confidence</th>
                <th>Status</th>
                <th>Correction</th>
              </tr>
            </thead>
            <tbody>
              {detections.map((detection) => {
                const draft = getDraft(detection)
                return (
                  <tr key={detection.id}>
                    <td>{detection.product_name ?? 'Unknown product'}</td>
                    <td>{detection.sku ?? '-'}</td>
                    <td>{detection.confidence.toFixed(2)}</td>
                    <td>
                      <span className={`status-badge status-${detection.status}`}>
                        {detection.status}
                      </span>
                    </td>
                    <td>
                      <div className="correction-grid">
                        <input
                          aria-label={`SKU ${detection.id}`}
                          placeholder="SKU"
                          value={draft.sku}
                          onChange={(event) =>
                            onDraftChange(detection.id, 'sku', event.target.value)
                          }
                        />
                        <input
                          aria-label={`Product ${detection.id}`}
                          placeholder="Product name"
                          value={draft.productName}
                          onChange={(event) =>
                            onDraftChange(detection.id, 'productName', event.target.value)
                          }
                        />
                        <input
                          aria-label={`Confidence ${detection.id}`}
                          placeholder="Confidence (0-1)"
                          value={draft.confidence}
                          onChange={(event) =>
                            onDraftChange(detection.id, 'confidence', event.target.value)
                          }
                        />
                        <button
                          type="button"
                          onClick={() => void onApplyCorrection(detection)}
                        >
                          Apply correction
                        </button>
                        <button
                          type="button"
                          onClick={() => void onSaveDetectionReference(detection)}
                        >
                          Save as reference
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </section>

      <section className="panel wide-panel">
        <h2>Upload History</h2>
        {history.length === 0 ? (
          <p className="muted-text">No uploads yet.</p>
        ) : (
          <ul className="history-list">
            {history.map((item) => (
              <li key={item.id}>
                <button type="button" onClick={() => void onSelectHistory(item.id)}>
                  <span>{item.filename}</span>
                  <span>{item.status}</span>
                  <span>{new Date(item.created_at).toLocaleString()}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel wide-panel">
        <h2>Dataset Export</h2>
        <button
          type="button"
          onClick={() => void onExportDataset()}
          disabled={exportingDataset}
        >
          {exportingDataset ? 'Exporting...' : 'Export Active-Learning Dataset'}
        </button>
        {exportMessage ? <p className="muted-text">{exportMessage}</p> : null}
        {lastExport ? (
          <div className="export-summary">
            <p>
              <strong>Path:</strong> {lastExport.export_path}
            </p>
            <p>
              <strong>Detection labels:</strong> {lastExport.detection.annotation_count}
            </p>
            <p>
              <strong>Recognition crops:</strong> {lastExport.recognition.image_count}
            </p>
          </div>
        ) : (
          <p className="muted-text">Generate train/val/test files from current detections.</p>
        )}
      </section>
    </div>
  )
}

export default App
