import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import httpx

from event_publisher import EventPublisher
from stream_processor import OrchestratorStreamProcessor

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

AMQP_URL = os.getenv("AMQP_URL", "amqp://guest:guest@localhost:5672/")
GANTRY_ID = os.getenv("GANTRY_ID", "gantry-01")
CAMERA_SERVICE_URL = os.getenv("CAMERA_SERVICE_URL", "http://camera_service:8002")
OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://ocr_service:8001")
STREAM_URL = os.getenv("STREAM_URL", "")          # ← read at startup
PROCESS_INTERVAL = float(os.getenv("PROCESS_INTERVAL", "6.0"))  # ← default 6s
COOLDOWN_S = float(os.getenv("COOLDOWN_S", "30.0"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detect-plate-service")

_state: dict = {
    "publisher": None,
    "stream": None,
}


class StreamStartRequest(BaseModel):
    stream_url: str | None = None
    gantry_id: str | None = None
    process_interval_s: float | None = None
    cooldown_s: float | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Connect RabbitMQ
    publisher = EventPublisher(amqp_url=AMQP_URL)
    try:
        await publisher.connect()
        _state["publisher"] = publisher
        logger.info("RabbitMQ connected.")
    except Exception as e:
        logger.error(            # ← was warning, easy to miss
            "RabbitMQ unavailable — running in LOG-ONLY mode. "
            "No plate events will be published. Error: %s", e
        )
        _state["publisher"] = None

    # 2. Auto-start stream if STREAM_URL is configured
    if STREAM_URL:
        logger.info("STREAM_URL configured — auto-starting stream processor.")
        stream = OrchestratorStreamProcessor(
            publisher=_state["publisher"],
            camera_url=CAMERA_SERVICE_URL,
            ocr_url=OCR_SERVICE_URL,
            gantry_id=GANTRY_ID,
            interval=PROCESS_INTERVAL,
            cooldown=COOLDOWN_S,
        )
        try:
            await stream.start()
            _state["stream"] = stream
            logger.info("Stream processor auto-started (interval=%.1fs).", PROCESS_INTERVAL)
        except Exception as e:
            logger.error("Failed to auto-start stream processor: %s", e)
    else:
        logger.info("No STREAM_URL set — waiting for POST /stream/start.")

    yield

    # Teardown
    if _state["stream"]:
        await _state["stream"].stop()
    if _state["publisher"]:
        await _state["publisher"].close()


app = FastAPI(title="Detect Plate Orchestrator", lifespan=lifespan)


@app.get("/health")
async def health():
    stream = _state["stream"]
    return {
        "status": "ok",
        "rabbitmq": "connected" if _state["publisher"] else "disconnected",
        "stream": {"active": stream.is_running if stream else False},
    }


@app.post("/stream/start")
async def stream_start(body: StreamStartRequest | None = None):
    if _state["stream"] and _state["stream"].is_running:
        raise HTTPException(status_code=409, detail="Stream already running")

    url = (body and body.stream_url) or STREAM_URL
    if not url:
        raise HTTPException(status_code=400, detail="No stream_url provided and STREAM_URL not configured")

    # Push new URL to camera service if explicitly provided
    if body and body.stream_url:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
            try:
                await client.post(f"{CAMERA_SERVICE_URL}/config", params={"stream_url": url})
            except Exception as e:
                logger.error("Failed to update camera service: %s", e)
                raise HTTPException(status_code=502, detail="Camera service unreachable")

    gantry = (body and body.gantry_id) or GANTRY_ID
    interval = (body and body.process_interval_s) or PROCESS_INTERVAL
    cooldown = (body and body.cooldown_s) or COOLDOWN_S

    stream = OrchestratorStreamProcessor(
        publisher=_state["publisher"],
        camera_url=CAMERA_SERVICE_URL,
        ocr_url=OCR_SERVICE_URL,
        gantry_id=gantry,
        interval=interval,
        cooldown=cooldown,
    )
    await stream.start()
    _state["stream"] = stream
    return {"status": "started", "gantry": gantry, "interval_s": interval}


@app.post("/stream/stop")
async def stream_stop():
    stream = _state["stream"]
    if not stream or not stream.is_running:
        raise HTTPException(status_code=409, detail="No stream running")
    await stream.stop()
    _state["stream"] = None
    return {"status": "stopped"}


@app.get("/detections")
async def detections():
    stream = _state["stream"]
    return {
        "streamActive": stream.is_running if stream else False,
        "detections": stream.detection_log if stream else [],
    }


@app.get("/video")
async def proxy_video():
    client = httpx.AsyncClient()
    req = client.build_request("GET", f"{CAMERA_SERVICE_URL}/stream")
    r = await client.send(req, stream=True)
    return StreamingResponse(
        r.aiter_raw(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        background=r.aclose,
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="info")