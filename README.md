# Gantry License Plate Reader — Singapore

A license plate detection + OCR microservice tuned for **Singapore LTA plates**,
designed to run at a drive-thru gantry as part of a microservices architecture.

Detects plates via camera stream, reads them with a transformer OCR model, and
publishes `vehicle.plate.detected` events to RabbitMQ for the Arrival Orchestrator.

## Architecture

```
IP Webcam (phone at gantry)
       │  MJPEG / RTSP
       ▼
 StreamProcessor
  ├─ drain buffer (prevents stale frames)
  ├─ brightness gate (skip black frames)
  ├─ sharpness gate (Laplacian variance — skip motion blur)
  ├─ PlateReader.read(frame)
  │    └─ YOLO11n detect → Crop → Normalise colour → CLAHE → TrOCR → Clean text
  ├─ dedup (30s cooldown per plate)
  └─ EventPublisher.publish()
            │
            ▼
     RabbitMQ  [gantry.events / vehicle.plate.detected]
            │
            ▼
     Arrival Orchestrator (Node.js)
      └─ order lookup → staff alert
```

Also exposes a single-image HTTP endpoint (`POST /read-plate`) for debugging
and manual testing.

## RabbitMQ Event Payload

When a plate is detected, the following message is published to the
`gantry.events` topic exchange with routing key `vehicle.plate.detected`:

```json
{
  "event": "vehicle.plate.detected",
  "timestamp": "2026-04-03T08:15:30.000Z",
  "gantryId": "gantry-01",
  "plate": {
    "text": "SBA1234A",
    "confidence": 0.94,
    "bbox": [120, 340, 380, 420]
  }
}
```

Your Arrival Orchestrator binds to `vehicle.plate.#` to receive all
plate-related events.

---

## Singapore Plate Format

Singapore plates follow the LTA-mandated `SBA 1234 A` format:

| Part        | Detail                                                         |
|-------------|----------------------------------------------------------------|
| Prefix      | 1–3 letters (I and O never used; no vowel in 2nd char of 3-letter prefix) |
| Numbers     | 1–4 digits                                                     |
| Checksum    | 1 letter (F, I, N, O, Q, V, W never used as check digits)     |
| Max length  | 8 characters — e.g. `SBA1234A`                                |

**Two LTA-approved colour schemes** (both handled automatically):
- **White-on-black** — white text on black background (most common from dealerships)
- **Black-on-white** (front) / **black-on-yellow** (rear) — Euro reflective scheme

The system auto-detects the colour scheme on each plate crop and normalises it
before OCR, so both schemes work without any configuration.

---

## Why This Stack

| Choice    | Why                                                                           |
|-----------|-------------------------------------------------------------------------------|
| YOLO11n   | Detects plate *location* only — works globally regardless of plate style      |
| TrOCR     | Transformer OCR pretrained on printed text; handles SG's Charles Wright font accurately — EasyOCR confused Q→D, X→Y on this typeface |
| CLAHE     | TrOCR expects intact grayscale — hard binarization breaks strokes; CLAHE enhances contrast without destroying them |
| FastAPI   | Async, auto-docs at `/docs`, production-ready                                 |
| aio-pika  | Async RabbitMQ client with robust auto-reconnect                              |

No API keys. No cloud fees. Everything runs locally.

---

## Project Structure

```
sg-plate-reader/
├── plate_model.pt          ← YOLO weights (after running download_model.py)
├── download_model.py       ← one-shot model download from HuggingFace
├── plate_reader.py         ← core detection + OCR engine
├── stream_processor.py     ← camera stream → frame selection → detect → dedup
├── event_publisher.py      ← async RabbitMQ publisher
├── server.py               ← FastAPI endpoints + stream lifecycle
├── test_reader.py          ← local debug tool (image or webcam)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml      ← RabbitMQ + LPR service for local dev
├── .env.example
└── .gitignore
```

---

## Setup

### Prerequisites
- Python 3.10+
- RabbitMQ (for event publishing — optional for local testing)
- A camera source: IP Webcam app (Android), RTSP IP camera, or USB webcam

### Step 1: Create the project
```bash
cd D:\PROJECTS\licenseplatereader
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download the pretrained model
```bash
python download_model.py
```

This saves `plate_model.pt` (~6 MB) in your project folder.

Verify:
```bash
python -c "from ultralytics import YOLO; m = YOLO('plate_model.pt'); print('OK:', m.names)"
```

---

## Running

### Option A: Docker Compose (recommended)

Starts RabbitMQ and the LPR service together:

```bash
docker compose up -d
```

- Plate API + Swagger UI: `http://localhost:8000/docs`
- RabbitMQ management: `http://localhost:15672` (guest / guest)

To auto-start the camera stream on boot, uncomment the `STREAM_URL` line
in `docker-compose.yml`.

### Option B: Local development

Start RabbitMQ however you prefer (Docker, Homebrew, installer), then:

```bash
# Copy and edit env config
cp .env.example .env

# Start the server
python server.py
```

Server starts at `http://localhost:8000`. Swagger docs at `/docs`.

If RabbitMQ is not running, the service starts in **log-only mode** —
plates are detected and logged to stdout but events are not published.

---

## API Endpoints

### `GET /health`

Returns model, RabbitMQ, and stream status. Suitable for container health probes.

```bash
curl http://localhost:8000/health
```

### `POST /read-plate`

Upload a single image for plate detection. Useful for debugging.

```bash
curl -X POST http://localhost:8000/read-plate -F "file=@test_car.jpg"
```

```json
{
  "success": true,
  "plates": [
    {"text": "SBA1234A", "confidence": 0.94, "bbox": [120, 340, 380, 420]}
  ],
  "processingTimeMs": 142
}
```

### `POST /stream/start`

Connect to a camera and begin continuous plate detection.

```bash
curl -X POST http://localhost:8000/stream/start \
  -H "Content-Type: application/json" \
  -d '{"stream_url": "http://192.168.1.5:8080/video"}'
```

You can also set `STREAM_URL` in `.env` to auto-start on boot instead.

Optional overrides in the request body:

| Field                | Default      | Description                        |
|----------------------|--------------|------------------------------------|
| `stream_url`         | from env     | RTSP or MJPEG camera URL           |
| `gantry_id`          | `gantry-01`  | Included in every RabbitMQ event   |
| `process_interval_s` | `1.0`        | Seconds between processing frames  |
| `cooldown_s`         | `30.0`       | Dedup cooldown per plate           |

### `POST /stream/stop`

Stop the active camera stream.

```bash
curl -X POST http://localhost:8000/stream/stop
```

---

## Finding Your Camera Stream URL

**IP Webcam (Android — what you're using):**
1. Open IP Webcam on your phone
2. Tap **Start server**
3. The URL is shown at the bottom of the screen: `http://192.168.X.X:8080`
4. Your stream URL is: `http://192.168.X.X:8080/video`
5. Verify in your browser — you should see a live video feed

Your phone and PC must be on the same Wi-Fi network.

**RTSP IP camera:**
Check the camera's manual. Typically: `rtsp://admin:password@192.168.1.100:554/stream1`

---

## Local Debug Testing (no server, no RabbitMQ)

```bash
# From an image file
python test_reader.py plate_model.pt test_car.jpg

# From webcam / IP Webcam frame capture
python test_reader.py plate_model.pt
```

This saves debug images at each pipeline stage (`debug_crop_N.jpg`,
`debug_normalised_N.jpg`, `debug_processed_N.jpg`) so you can see
exactly what TrOCR receives.

---

## Integration with the Drive-Thru Pre-Order System

### Consuming events in the Arrival Orchestrator (Node.js)

```javascript
const amqp = require("amqplib");

async function listenForPlates() {
  const conn = await amqp.connect("amqp://localhost:5672");
  const ch = await conn.createChannel();

  await ch.assertExchange("gantry.events", "topic", { durable: true });
  const q = await ch.assertQueue("", { exclusive: true });
  await ch.bindQueue(q.queue, "gantry.events", "vehicle.plate.detected");

  ch.consume(q.queue, (msg) => {
    const event = JSON.parse(msg.content.toString());
    console.log("Vehicle arrived:", event.plate.text);
    // → look up pre-order by plate
    // → alert kitchen staff
    ch.ack(msg);
  });
}

listenForPlates();
```

### Single-image HTTP call (from other services)

```python
import requests

def detect_plate(image_path: str) -> str | None:
    with open(image_path, "rb") as f:
        resp = requests.post(
            "http://plate-reader:8000/read-plate",
            files={"file": f},
            timeout=10,
        )
    data = resp.json()
    if data["success"] and data["plates"]:
        return data["plates"][0]["text"]
    return None
```

---

## Singapore Gantry Tips

| Topic             | Recommendation                                                                        |
|-------------------|---------------------------------------------------------------------------------------|
| Camera angle      | 1–2 m height, slight downward angle toward the rear plate                            |
| Colour scheme     | System auto-detects; no config needed                                                 |
| Frame selection   | Handled by the stream processor — sharpness scoring picks the best frames            |
| Dedup             | Default 30s cooldown; adjust `COOLDOWN_S` based on your gantry throughput             |
| Confidence        | Start with `DETECT_CONF=0.5`; lower to `0.35` if plates are missed                   |
| Night / rain      | IR illuminator recommended — SG plates are retro-reflective (LTA requirement)         |
| Off-peak cars     | Red plate, same alphanumeric format — detected correctly                              |
| EV/PHEV plates    | LTA announced green plates (2026) — same format, no code changes needed               |

---

## Troubleshooting

| Problem                        | Fix                                                                         |
|--------------------------------|-----------------------------------------------------------------------------|
| No plates detected             | Lower `DETECT_CONF` to 0.35; check image is at least 640 px wide           |
| OCR reads wrong characters     | Check debug images — ensure plate is well-lit, not at extreme angle         |
| White-on-black plate garbled   | Check `debug_normalised_N.jpg` — should be inverted to dark-on-light       |
| Stream won't connect           | Verify URL in browser first; phone and PC must be on same Wi-Fi            |
| Stream lag / stale frames      | Normal — the buffer drain handles this; lower `PROCESS_INTERVAL` if needed  |
| RabbitMQ not connecting        | Service runs in log-only mode; check `AMQP_URL` and that RabbitMQ is up    |
| `ModuleNotFoundError`          | Ensure venv is activated                                                    |
| Slow on CPU                    | TrOCR is ~300ms per plate; YOLO11n is already the lightest variant          |
| Model not found (503)          | Run `python download_model.py` or set `MODEL_PATH` env var                  |