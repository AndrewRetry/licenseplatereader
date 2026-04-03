"""
server.py — FastAPI microservice for the Gantry License Plate Reader.

Two modes of operation:

  1. HTTP endpoint  (POST /read-plate)
     Upload a single image, get plate text back.  Useful for debugging,
     manual testing, and one-shot reads from other services.

  2. Stream mode  (POST /stream/start)
     Connect to an RTSP/MJPEG camera, continuously detect plates, and
     publish ``vehicle.plate.detected`` events to RabbitMQ for the
     Arrival Orchestrator to consume.

Endpoints:
  GET   /health          Liveness + readiness (model, RabbitMQ, stream status)
  GET   /dashboard       Live monitoring UI with camera feed + detection log
  GET   /detections      Recent plate detections (JSON, feeds the dashboard)
  POST  /read-plate      Single-image plate read
  POST  /stream/start    Start camera stream processing
  POST  /stream/stop     Stop camera stream processing

Run:
  python server.py
  # or: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

from plate_reader import PlateReader
from event_publisher import EventPublisher
from stream_processor import StreamProcessor

# ------------------------------------------------------------------
# Configuration (env vars with safe defaults)
# ------------------------------------------------------------------
MODEL_PATH       = os.getenv("MODEL_PATH", "plate_model.pt")
DETECT_CONF      = float(os.getenv("DETECT_CONF", "0.5"))
USE_GPU          = os.getenv("USE_GPU", "false").lower() == "true"
HOST             = os.getenv("HOST", "0.0.0.0")
PORT             = int(os.getenv("PORT", "8000"))

AMQP_URL         = os.getenv("AMQP_URL", "amqp://guest:guest@localhost:5672/")
GANTRY_ID        = os.getenv("GANTRY_ID", "gantry-01")
STREAM_URL       = os.getenv("STREAM_URL", "")
PROCESS_INTERVAL = float(os.getenv("PROCESS_INTERVAL", "1.0"))
COOLDOWN_S       = float(os.getenv("COOLDOWN_S", "30.0"))

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("plate-api")

# ------------------------------------------------------------------
# Shared state — populated during lifespan
# ------------------------------------------------------------------
_state: dict = {
    "reader": None,       # PlateReader
    "publisher": None,    # EventPublisher | None
    "stream": None,       # StreamProcessor | None
}


# ------------------------------------------------------------------
# Request / response schemas
# ------------------------------------------------------------------

class StreamStartRequest(BaseModel):
    """Optional overrides when starting the stream via the API."""
    stream_url: str | None = None
    gantry_id: str | None = None
    process_interval_s: float | None = None
    cooldown_s: float | None = None


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model → connect RabbitMQ → (optional) auto-start stream."""

    # 1. Load the plate reader model
    logger.info("Loading plate reader model from %s …", MODEL_PATH)
    try:
        _state["reader"] = PlateReader(
            model_path=MODEL_PATH,
            detect_conf=DETECT_CONF,
            gpu=USE_GPU,
        )
        logger.info("Plate reader ready.")
    except FileNotFoundError as e:
        logger.error("MODEL NOT FOUND: %s", e)

    # 2. Connect RabbitMQ (non-fatal — falls back to log-only)
    publisher = EventPublisher(amqp_url=AMQP_URL)
    try:
        await publisher.connect()
        _state["publisher"] = publisher
        logger.info("RabbitMQ connected.")
    except Exception as e:
        logger.warning(
            "RabbitMQ unavailable (%s) — running in log-only mode. "
            "Events will be logged but not published.",
            e,
        )
        _state["publisher"] = None

    # 3. Auto-start stream if STREAM_URL is set and the model loaded
    if STREAM_URL and _state["reader"]:
        try:
            stream = StreamProcessor(
                reader=_state["reader"],
                publisher=_state["publisher"],
                stream_url=STREAM_URL,
                gantry_id=GANTRY_ID,
                process_interval_s=PROCESS_INTERVAL,
                cooldown_s=COOLDOWN_S,
            )
            await stream.start()
            _state["stream"] = stream
            logger.info("Auto-started stream from STREAM_URL env var.")
        except Exception as e:
            logger.error("Failed to auto-start stream: %s", e)

    yield  # ---- app is running ----

    # Shutdown
    if _state["stream"]:
        await _state["stream"].stop()
    if _state["publisher"]:
        await _state["publisher"].close()
    _state["reader"] = None
    logger.info("Shutdown complete.")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="Gantry Plate Reader API",
    description=(
        "License plate detection microservice for the Drive-Thru Pre-Order "
        "platform.  Supports single-image HTTP reads and continuous camera "
        "stream processing with RabbitMQ event publishing."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness + readiness check."""
    reader: PlateReader | None = _state["reader"]
    publisher: EventPublisher | None = _state["publisher"]
    stream: StreamProcessor | None = _state["stream"]

    return {
        "status": "ok" if reader else "model_not_loaded",
        "model": MODEL_PATH,
        "gpu": USE_GPU,
        "gantryId": GANTRY_ID,
        "rabbitmq": "connected" if (publisher and publisher.is_connected) else "disconnected",
        "stream": {
            "active": stream.is_running if stream else False,
            "streamUrl": stream.stream_url if stream else None,
            "recentPlates": stream.recent_plates if stream else {},
        },
    }


# ------------------------------------------------------------------
# GET /detections
# ------------------------------------------------------------------

@app.get("/detections")
async def detections():
    """Recent plate detections from the stream processor.

    Returns the last 50 detections (newest first).  The dashboard
    polls this endpoint every 2 seconds for live updates.
    """
    stream: StreamProcessor | None = _state["stream"]
    return {
        "streamActive": stream.is_running if stream else False,
        "streamUrl": stream.stream_url if stream else None,
        "detections": stream.detection_log if stream else [],
    }


# ------------------------------------------------------------------
# POST /read-plate
# ------------------------------------------------------------------

@app.post("/read-plate")
async def read_plate(file: UploadFile = File(...)):
    """Upload an image (JPEG/PNG) and receive detected plate text."""
    reader = _state["reader"]
    if reader is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got: {content_type}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    start = time.time()
    try:
        plates = reader.read_from_bytes(image_bytes)
    except Exception as e:
        logger.exception("Error processing image: %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e))
    elapsed_ms = round((time.time() - start) * 1000)

    logger.info(
        "Processed '%s' — %d plate(s) in %d ms",
        file.filename, len(plates), elapsed_ms,
    )

    return JSONResponse({
        "success": True,
        "plates": plates,
        "processingTimeMs": elapsed_ms,
    })


# ------------------------------------------------------------------
# POST /stream/start
# ------------------------------------------------------------------

@app.post("/stream/start")
async def stream_start(body: StreamStartRequest | None = None):
    """Start processing a camera stream."""
    reader = _state["reader"]
    if reader is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if _state["stream"] and _state["stream"].is_running:
        raise HTTPException(
            status_code=409,
            detail="Stream already running. POST /stream/stop first.",
        )

    url = (body and body.stream_url) or STREAM_URL
    if not url:
        raise HTTPException(
            status_code=400,
            detail=(
                "No stream URL provided. Set the STREAM_URL environment "
                "variable or pass stream_url in the request body."
            ),
        )

    gantry   = (body and body.gantry_id) or GANTRY_ID
    interval = (body and body.process_interval_s) or PROCESS_INTERVAL
    cooldown = (body and body.cooldown_s) or COOLDOWN_S

    stream = StreamProcessor(
        reader=reader,
        publisher=_state["publisher"],
        stream_url=url,
        gantry_id=gantry,
        process_interval_s=interval,
        cooldown_s=cooldown,
    )

    try:
        await stream.start()
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail=str(e))

    _state["stream"] = stream

    return {
        "status": "started",
        "streamUrl": url,
        "gantryId": gantry,
        "processIntervalS": interval,
        "cooldownS": cooldown,
        "rabbitmq": "connected" if _state["publisher"] else "log-only",
    }


# ------------------------------------------------------------------
# POST /stream/stop
# ------------------------------------------------------------------

@app.post("/stream/stop")
async def stream_stop():
    """Stop the active camera stream."""
    stream: StreamProcessor | None = _state["stream"]
    if not stream or not stream.is_running:
        raise HTTPException(status_code=409, detail="No active stream to stop.")

    await stream.stop()
    _state["stream"] = None
    return {"status": "stopped"}


# ------------------------------------------------------------------
# GET /dashboard — Live monitoring UI
# ------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gantry LPR — Live Dashboard</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --text: #e1e4ed;
    --text-dim: #7a7f8e;
    --accent: #4ade80;
    --accent-dim: #166534;
    --warning: #f59e0b;
    --danger: #ef4444;
    --plate-bg: #1e293b;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* --- Header --- */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  .header h1 {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.02em;
  }
  .header h1 span { color: var(--accent); }
  .status-bar { display: flex; gap: 16px; align-items: center; }
  .status-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    background: var(--bg);
    border: 1px solid var(--border);
  }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--danger);
  }
  .status-dot.ok { background: var(--accent); }
  .status-dot.warn { background: var(--warning); }

  /* --- Main layout --- */
  .main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    padding: 20px 24px;
    max-height: calc(100vh - 65px);
  }

  /* --- Camera panel --- */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  .panel-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
  }
  .camera-feed {
    width: 100%;
    aspect-ratio: 16/9;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }
  .camera-feed img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
  .camera-placeholder {
    color: var(--text-dim);
    font-size: 14px;
    text-align: center;
    padding: 20px;
  }
  .camera-placeholder p { margin-top: 8px; font-size: 12px; }

  /* --- Detection log --- */
  .detection-list {
    padding: 8px;
    overflow-y: auto;
    max-height: calc(100vh - 140px);
  }
  .detection-card {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 14px;
    border-radius: 8px;
    margin-bottom: 6px;
    background: var(--bg);
    border: 1px solid var(--border);
    transition: border-color 0.3s;
  }
  .detection-card.new {
    border-color: var(--accent);
    box-shadow: 0 0 12px rgba(74, 222, 128, 0.1);
  }
  .plate-text {
    font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text);
    background: var(--plate-bg);
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid var(--border);
    white-space: nowrap;
  }
  .detection-meta {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .detection-conf {
    font-size: 13px;
    font-weight: 500;
  }
  .conf-high { color: var(--accent); }
  .conf-mid { color: var(--warning); }
  .conf-low { color: var(--danger); }
  .detection-time {
    font-size: 12px;
    color: var(--text-dim);
  }
  .detection-gantry {
    font-size: 11px;
    color: var(--text-dim);
  }

  /* --- Empty state --- */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    color: var(--text-dim);
    text-align: center;
  }
  .empty-state .icon { font-size: 40px; margin-bottom: 12px; }
  .empty-state p { font-size: 13px; margin-top: 6px; }

  /* --- Counter badge --- */
  .count-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 22px;
    height: 22px;
    padding: 0 7px;
    border-radius: 11px;
    background: var(--accent-dim);
    color: var(--accent);
    font-size: 12px;
    font-weight: 600;
    margin-left: 8px;
  }

  @media (max-width: 900px) {
    .main { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="header">
  <h1><span>&#9679;</span> Gantry LPR Dashboard</h1>
  <div class="status-bar">
    <div class="status-pill">
      <div class="status-dot" id="dot-model"></div>
      <span id="label-model">Model</span>
    </div>
    <div class="status-pill">
      <div class="status-dot" id="dot-rabbitmq"></div>
      <span id="label-rabbitmq">RabbitMQ</span>
    </div>
    <div class="status-pill">
      <div class="status-dot" id="dot-stream"></div>
      <span id="label-stream">Stream</span>
    </div>
  </div>
</div>

<div class="main">
  <!-- Camera feed -->
  <div class="panel">
    <div class="panel-header">Camera Feed</div>
    <div class="camera-feed" id="camera-container">
      <div class="camera-placeholder" id="camera-placeholder">
        <div style="font-size: 32px;">&#128247;</div>
        <p>No stream active.<br>Start a stream via POST /stream/start</p>
      </div>
    </div>
  </div>

  <!-- Detection log -->
  <div class="panel">
    <div class="panel-header">
      Detections <span class="count-badge" id="detection-count">0</span>
    </div>
    <div class="detection-list" id="detection-list">
      <div class="empty-state" id="empty-state">
        <div class="icon">&#128690;</div>
        <strong>Waiting for vehicles…</strong>
        <p>Plates will appear here as they are detected.</p>
      </div>
    </div>
  </div>
</div>

<script>
(function() {
  const API = window.location.origin;
  const POLL_HEALTH_MS = 3000;
  const POLL_DETECT_MS = 2000;

  let lastDetectionCount = 0;
  let knownTimestamps = new Set();

  // --- Status polling ---
  async function pollHealth() {
    try {
      const r = await fetch(API + '/health');
      const d = await r.json();

      setDot('dot-model', d.status === 'ok' ? 'ok' : '');
      document.getElementById('label-model').textContent =
        d.status === 'ok' ? 'Model OK' : 'Model Error';

      setDot('dot-rabbitmq', d.rabbitmq === 'connected' ? 'ok' : 'warn');
      document.getElementById('label-rabbitmq').textContent =
        d.rabbitmq === 'connected' ? 'RabbitMQ' : 'RabbitMQ (offline)';

      const streamActive = d.stream && d.stream.active;
      setDot('dot-stream', streamActive ? 'ok' : '');
      document.getElementById('label-stream').textContent =
        streamActive ? 'Streaming' : 'Stream Off';

      // Update camera feed
      const streamUrl = d.stream && d.stream.streamUrl;
      updateCameraFeed(streamUrl, streamActive);

    } catch(e) {
      setDot('dot-model', '');
      setDot('dot-rabbitmq', '');
      setDot('dot-stream', '');
    }
  }

  function setDot(id, state) {
    const el = document.getElementById(id);
    el.className = 'status-dot' + (state ? ' ' + state : '');
  }

  // --- Camera feed ---
  let currentFeedUrl = null;
  function updateCameraFeed(streamUrl, active) {
    const container = document.getElementById('camera-container');
    const placeholder = document.getElementById('camera-placeholder');

    if (active && streamUrl && streamUrl !== currentFeedUrl) {
      currentFeedUrl = streamUrl;
      // IP Webcam serves MJPEG at /video — browsers render it natively in <img>
      const img = document.createElement('img');
      img.src = streamUrl;
      img.alt = 'Live camera feed';
      img.onerror = function() {
        // If the browser can't reach the phone IP, show a fallback
        this.style.display = 'none';
        placeholder.innerHTML =
          '<div style="font-size:32px">&#128247;</div>' +
          '<p>Camera feed unreachable from browser.<br>' +
          'Stream is still processing on the server.<br>' +
          '<small style="color:#4ade80">' + streamUrl + '</small></p>';
        placeholder.style.display = 'block';
      };
      // Remove old img if any
      const old = container.querySelector('img');
      if (old) old.remove();
      placeholder.style.display = 'none';
      container.appendChild(img);
    } else if (!active) {
      currentFeedUrl = null;
      const old = container.querySelector('img');
      if (old) old.remove();
      placeholder.style.display = 'block';
      placeholder.innerHTML =
        '<div style="font-size:32px">&#128247;</div>' +
        '<p>No stream active.<br>Start a stream via POST /stream/start</p>';
    }
  }

  // --- Detections polling ---
  async function pollDetections() {
    try {
      const r = await fetch(API + '/detections');
      const d = await r.json();

      const list = document.getElementById('detection-list');
      const empty = document.getElementById('empty-state');
      const countEl = document.getElementById('detection-count');

      const detections = d.detections || [];
      countEl.textContent = detections.length;

      if (detections.length === 0) {
        empty.style.display = 'flex';
        return;
      }
      empty.style.display = 'none';

      // Build HTML
      let html = '';
      detections.forEach(function(det, i) {
        const key = det.timestamp + det.text;
        const isNew = !knownTimestamps.has(key);

        const conf = det.confidence;
        let confClass = 'conf-high';
        if (conf < 0.5) confClass = 'conf-low';
        else if (conf < 0.75) confClass = 'conf-mid';

        const time = formatTime(det.timestamp);

        html += '<div class="detection-card' + (isNew ? ' new' : '') + '">' +
          '<div class="plate-text">' + escHtml(det.text) + '</div>' +
          '<div class="detection-meta">' +
            '<div class="detection-conf ' + confClass + '">' +
              (conf * 100).toFixed(1) + '% confidence</div>' +
            '<div class="detection-time">' + time + '</div>' +
            '<div class="detection-gantry">Gantry: ' + escHtml(det.gantryId || '—') + '</div>' +
          '</div>' +
        '</div>';
      });
      list.innerHTML = html;

      // Play beep on new detections
      if (detections.length > lastDetectionCount && lastDetectionCount > 0) {
        beep();
      }
      lastDetectionCount = detections.length;

      // Track known detections
      knownTimestamps.clear();
      detections.forEach(function(det) {
        knownTimestamps.add(det.timestamp + det.text);
      });

      // Remove "new" highlight after 3 seconds
      setTimeout(function() {
        document.querySelectorAll('.detection-card.new').forEach(function(el) {
          el.classList.remove('new');
        });
      }, 3000);

    } catch(e) { /* server unreachable — status dots handle this */ }
  }

  function formatTime(iso) {
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch(e) { return iso; }
  }

  function escHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

  // Tiny beep using Web Audio API
  function beep() {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = 880;
      osc.type = 'sine';
      gain.gain.value = 0.1;
      osc.start();
      osc.stop(ctx.currentTime + 0.12);
    } catch(e) {}
  }

  // --- Start polling ---
  pollHealth();
  pollDetections();
  setInterval(pollHealth, POLL_HEALTH_MS);
  setInterval(pollDetections, POLL_DETECT_MS);
})();
</script>

</body>
</html>"""


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Live monitoring dashboard.

    Open in a browser to see the camera feed, status indicators,
    and a real-time detection log.
    """
    return DASHBOARD_HTML


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False, log_level="info")