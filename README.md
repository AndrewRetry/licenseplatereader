# licenseplatereader

Singapore licence plate detection microservice — Node.js / Express / ESM.

Detects plates under varied real-world conditions: night, glare, washed-out, dirty, angled.  
Returns ranked candidates with OCR confidence, checksum validation, and vehicle format.

---

## Stack

| Package | Purpose |
|---|---|
| `express` v5 | HTTP server |
| `tesseract.js` v5 | OCR engine (WASM, no binary deps) |
| `sharp` | Image preprocessing pipelines |
| `multer` | Multipart upload handling |
| `uuid` | Request IDs |

No Python, no OpenCV, no native binaries — runs anywhere Node >= 18 runs.

---

## Setup

```bash
npm install
cp .env .env.local
npm start
```

Dev mode (auto-restart):
```bash
npm run dev
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3001` | HTTP port |
| `WORKER_POOL_SIZE` | `2` | Tesseract concurrent workers |
| `MIN_CONFIDENCE` | `45` | Reject OCR reads below this (0-100) |
| `MAX_FILE_SIZE_MB` | `10` | Upload size cap |
| `NODE_ENV` | `development` | Set `production` to disable debug image saving |

---

## API

### GET /api/plate/health

```bash
curl http://localhost:3001/api/plate/health
```

### POST /api/plate/detect

```bash
curl -X POST http://localhost:3001/api/plate/detect \
  -F "image=@/path/to/car.jpg"
```

Response (200):
```json
{
  "requestId": "550e8400...",
  "success": true,
  "elapsedMs": 1842,
  "best": {
    "plate": "SBA1234L",
    "formatted": "SBA 1234 L",
    "prefix": "SBA",
    "digits": "1234",
    "checksum": "L",
    "checksumValid": true,
    "format": "private_car",
    "confidence": "high",
    "ocrConfidence": 87.4,
    "pipeline": "standard",
    "region": "lower",
    "psm": 7,
    "score": 107.4
  },
  "candidates": [],
  "meta": { "ocrAttempts": 24, "preprocessVariants": 15 }
}
```

### POST /api/plate/detect/batch

```bash
curl -X POST http://localhost:3001/api/plate/detect/batch \
  -F "images=@car1.jpg" \
  -F "images=@car2.jpg"
```

### POST /api/plate/validate

```bash
curl -X POST http://localhost:3001/api/plate/validate \
  -H "Content-Type: application/json" \
  -d '{ "plate": "SBA1234L" }'
```

---

## Detection pipeline

Each image produces **15 variants** (5 pipelines x 3 crop regions), run through **3 PSM modes** = up to **45 OCR attempts**.

| Pipeline | Tuned for |
|---|---|
| standard | Day, clean plate |
| night | Low light, noise, LED glow |
| washed | Overexposed, direct sun |
| dirty | Faded/dirty, low contrast |
| angled | Gantry cam perspective |

Candidate scoring: `ocrConfidence + 15 (valid checksum) - 20 (invalid checksum) + 5 (common format)`

---

## Supported SG plate formats

| Format | Example |
|---|---|
| Private car | SBA 1234 L |
| Taxi | SHA 5678 H |
| Private hire (PHV) | SX 9999 K |
| Motorcycle | FBA 123 |
| Government | QX 1234 |
| Diplomatic | D 1234 |

---

## Rate limits

| Endpoint | Limit |
|---|---|
| /detect | 30 req/min per IP |
| /detect/batch | 10 req/min per IP |

---

## Integrating with Drive-Thru system

```
Car approaches gantry
  POST /api/plate/detect  (frame from camera)
  plate.best.plate  -->  match against order.customer_plate in DB
  confidence=high   -->  auto-grant entry
  confidence=medium -->  grant + log for review  
  confidence=low    -->  flag for staff override
```
