# sg-plate-reader

Detects and reads Singapore licence plates from a live camera stream or a still image, and publishes results downstream over RabbitMQ.

---

## How it works

```
Camera / image captures a frame in a set interval (default=6.0s)
      │
      ▼
  YOLO11n model -- locates the plate and puts bounding box in it
      │
      ▼
  Crop + normalise -- trim any parts of the image outside of the set bounding box
      │
      ▼
  fast-plate-ocr -- processes the image and returns string of the detected text inside of the cropped image
      │
      ▼
  SG checksum validation + dedup -- checks text against SG car plate format
      │
      ▼
  RabbitMQ  →  downstream services -- 
```

---

## Project structure

```
sg-plate-reader/
├── plate_model.pt          ← YOLO weights (downloaded by download_model.py)
├── download_model.py       ← one-shot model fetch from HuggingFace
├── plate_reader.py         ← detection + OCR engine
├── stream_processor.py     ← live stream handling, frame quality gates, dedup
├── event_publisher.py      ← RabbitMQ publisher
├── server.py               ← FastAPI app and endpoints
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Prerequisites

- Python 3.10+
- A camera source, any of the below:
  - *IP Webcam App* (Android) over MJPEG: `http://<phone-ip>:8080/video` (for demo purposes)
  - RTSP IP camera which exposes URL
- RabbitMQ — only required if you want events published downstream

---

## Setup

### 1. Create a virtual environment

```bash
cd D:\PROJECTS\licenseplatereader

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the YOLO model

```bash
python download_model.py
```

Saves `plate_model.pt` (~6 MB) to the project root. Run once.

Verify:

```bash
python -c "from ultralytics import YOLO; m = YOLO('plate_model.pt'); print('OK:', m.names)"
```

Expected Value:
```bash
OK: {0: 'License_Plate'}
```
This is because the correct model is trained on License Plates only

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values, at minimum set `STREAM_URL` to enable live streaming:

```env
STREAM_URL=http://192.168.1.5:8080/video
AMQP_URL=amqp://guest:guest@localhost:5672/
```

## How to Run
### Docker Compose
Starts RabbitMQ and the plate reader together:

```bash
docker compose up -d
```

| Service | URL |
|---|---|
| API + Swagger UI | http://localhost:8000/docs |
| RabbitMQ management | http://localhost:15672 (guest / guest) |

To auto-start the camera stream on boot, uncomment `STREAM_URL` in `docker-compose.yml`.

### Local Python

```bash
python server.py
```

---

## API

### `POST /read-plate`

Send an image, get back the detected plate.

```bash
curl -X POST http://localhost:8000/read-plate \
  -F "file=@car.jpg"
```

```json
{
  "plate": "SBA1234A", // the string text
  "confidence": 0.94,
  "checksum_valid": true,
  "bbox": [120, 340, 380, 420] // the bounding box which determines the approximate location of the car plate
}
```

### `POST /stream/start` · `POST /stream/stop`

Start or stop the live camera stream. If `STREAM_URL` is set in `.env`, the stream starts automatically on boot.

### `GET /health`

```json
{
  "status": "ok",
  "stream": "running",
  "rabbitmq": "connected"
}
```

---

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `PORT` | `8000` | HTTP port |
| `MODEL_PATH` | `plate_model.pt` | Path to YOLO weights .pt file |
| `DETECT_CONF` | `0.5` | YOLO Model minimum detection confidence threshold |
| `STREAM_URL` | — | MJPEG or RTSP camera URL |
| `AMQP_URL` | `amqp://guest:guest@localhost:5672/` | RabbitMQ connection string |
| `GANTRY_ID` | `gantry-01` | Included in every published event |
| `PROCESS_INTERVAL` | `6.0` | Seconds between frame grabs from the stream (default=6.0s) |
| `COOLDOWN_S` | `30.0` | Suppress duplicate events for the same plate within this window (default=30.0s) |

---

## RabbitMQ event

Published to the `gantry.events` topic exchange with routing key `vehicle.plate.detected`:

```json
{
  "plate": "SBA1234A",
  "confidence": 0.74,
  "checksum_valid": true,
  "gantry_id": "gantry-01", 
  "timestamp": "2026-04-06T10:00:00Z"
}
```

Any service binding to `vehicle.plate.#` routing key on `gantry.events` RabbitMQ exchange will receive it.

## Plate formats supported

| Type | Example | Checksum |
|---|---|---|
| Private car | SBA 1234 A | ✓ |
| Taxi | SHA 5678 H | ✓ |
| Private hire (PHV) | SX 9999 K | ✓ |
| Motorcycle | FBA 123 | X |
| Government | QX 1234 | X |

## Troubleshooting

**YOLO finds no plates**  
Check that `plate_model.pt` exists in the project root. Re-run `download_model.py` if missing.

**Black frames from webcam**  
The stream processor discards underexposed frames automatically. If it persists, check the IP Webcam app's exposure settings.

**RabbitMQ connection refused**  
The service starts in log-only mode when RabbitMQ is unreachable — plates are still detected and logged, just not published. Run `docker compose up -d rabbitmq` and it will reconnect on its own.