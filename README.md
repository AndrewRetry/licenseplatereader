# licenseplatereader v3

Singapore license plate detection microservice — **Python / FastAPI / OpenCV / EasyOCR**.

---

## Detection pipeline

```
POST /api/plate/detect  ←  raw image frame
        │
        ▼
  OpenCV plate detector
  ┌─────────────────────────────────────┐
  │  1. Bilateral filter (edge-safe)    │
  │  2. Canny edge detection            │
  │  3. Morphological dilate            │
  │  4. findContours                    │
  │  5. Filter: aspect ratio 2.0–7.5    │
  │     & area 0.2–25% of frame         │
  │  6. Position bonus (lower frame)    │
  └────────────────┬────────────────────┘
                   │  candidates ranked by score
                   │  (fallback: Haar cascade)
                   │  (fallback: horizontal strips)
                   ▼
  EasyOCR  (CRAFT detector + CRNN recogniser)
  ┌─────────────────────────────────────┐
  │  3 preprocessing variants per crop  │
  │  thresh / inverted / raw gray       │
  └────────────────┬────────────────────┘
                   ▼
  SG plate validation + checksum
        │
        ▼
  Ranked candidates → best result
```

---

## Stack

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Async HTTP server |
| `opencv-python-headless` | Plate region detection |
| `easyocr` | Neural OCR (no Tesseract) |
| `numpy` + `Pillow` | Image utilities |

---

## Setup

```bash
pip install -r requirements.txt
cp .env .env.local   # optional overrides
python server.py
```

Docker:
```bash
docker build -t licenseplatereader .
docker run -p 3001:3001 licenseplatereader
```

---

### GET /api/plate/health
```json
{ "status": "ok", "version": "3.0.0", "engine": "opencv+easyocr" }
```

### POST /api/plate/detect
```bash
curl -X POST http://localhost:3001/api/plate/detect \
  -F "image=@/path/to/car.jpg"
```

```json
{
  "requestId": "550e8400-...",
  "success": true,
  "elapsedMs": 620,
  "best": {
    "plate": "SBA1234L",
    "formatted": "SBA 1234 L",
    "prefix": "SBA",
    "digits": "1234",
    "checksum": "L",
    "checksumValid": true,
    "format": "private_car",
    "confidence": "high",
    "ocrConfidence": 91.3,
    "method": "contour",
    "score": 106.3
  },
  "candidates": [],
  "meta": { "regionsInspected": 4, "ocrAttempts": 4 }
}
```

### POST /api/plate/detect/batch
Up to 10 images per call.

### POST /api/plate/validate
```bash
curl -X POST http://localhost:3001/api/plate/validate \
  -H "Content-Type: application/json" \
  -d '{ "plate": "SBA1234L" }'
```

---

## Integration with Drive-Thru system

```
Car approaches gantry
  POST /api/plate/detect  ← frame from camera
  best.plate              → match against order.customer_plate
  confidence=high         → auto-grant entry
  confidence=medium       → grant + log for review
  confidence=low          → flag for staff override
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3001` | HTTP port |
| `LOG_LEVEL` | `INFO` | `DEBUG` for verbose |
| `MAX_FILE_SIZE_MB` | `10` | Upload size cap |
