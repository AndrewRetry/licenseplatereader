# License Plate Reader (SG-Based)

Detects and reads Singapore licence plates from a live camera stream and publishes results downstream over RabbitMQ.

---

## Project structure

```
licenseplatereader/
├── docker-compose.yml
├── .env.example
├── plate_model.pt                  ← YOLO weights (see Setup)
│
├── camera_service/                 ← Buffers live camera frames (port 8002)
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── ocr_service/                    ← YOLO detection + OCR inference (port 8001)
│   ├── main.py
│   ├── plate_reader.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── detect_plate_service/           ← Orchestrator: quality gates, dedup, publish (port 8081)
│   ├── main.py
│   ├── stream_processor.py
│   ├── event_publisher.py
│   ├── requirements.txt
│   └── Dockerfile
│
└── gantry-interface/               ← Dashboard + AMQP consumer + Arrival bridge (port 3500)
    ├── src/
    │   ├── index.js
    │   └── amqpConsumer.js
    ├── public/
    │   └── index.html
    ├── package.json
    ├── .env.example
    └── Dockerfile
```

---

## Prerequisites

- Docker + Docker Compose
- A camera source — one of:
  - *IP Webcam* app (Android) over MJPEG: `http://<phone-ip>:8080/video`
  - Any RTSP IP camera
- An Arrival API — a service that accepts `POST /arrival` with `{ "license_plate": "SBA1234A" }`

---

## Setup

### 1. Create the shared Docker network

```bash
docker network create drivethrough-net
```

### 2. Download the YOLO model

```bash
python download_model.py
```

Saves `plate_model.pt` to the project root. Run once.

### 3. Configure environment

Root `.env`:

```bash
cp .env.example .env
```

Gantry interface `.env`:

```bash
cp gantry-interface/.env.example gantry-interface/.env
```

Set at minimum in root `.env`:

```env
STREAM_URL=http://<phone-ip>:8080/video
AMQP_URL=amqp://guest:guest@rabbitmq:5672/
```

Set at minimum in `gantry-interface/.env`:

```env
AMQP_URL=amqp://guest:guest@rabbitmq:5672/
ARRIVAL_API_URL=http://your-arrival-service/arrival
```

### 4. Start all services

```bash
docker compose up -d
```

| Service | URL |
|---|---|
| Detect plate orchestrator | http://localhost:8081/docs |
| OCR service | http://localhost:8001/docs |
| Camera service | http://localhost:8002 |
| Gantry dashboard | http://localhost:3500 |
| RabbitMQ management UI | http://localhost:15672 (guest / guest) |