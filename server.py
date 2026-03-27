"""
server.py — FastAPI REST endpoint for license plate reading.

Endpoints:
  POST /read-plate       Upload an image, get plate text back.
  GET  /health           Health check (is the model loaded?).
  GET  /docs             Auto-generated Swagger UI.

Run:
  python server.py
  # or: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import time
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from plate_reader import PlateReader

# ------------------------------------------------------------------
# Configuration via environment variables (safe defaults)
# ------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "plate_model.pt")
DETECT_CONF = float(os.getenv("DETECT_CONF", "0.5"))
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("plate-api")

# ------------------------------------------------------------------
# App + model init
# ------------------------------------------------------------------
app = FastAPI(
    title="Gantry Plate Reader API",
    description="Upload a car image → get license plate text back.",
    version="1.0.0",
)

reader: PlateReader | None = None


@app.on_event("startup")
async def startup():
    """Load the model once when the server starts."""
    global reader
    logger.info("Loading plate reader model from %s ...", MODEL_PATH)
    try:
        reader = PlateReader(
            model_path=MODEL_PATH,
            detect_conf=DETECT_CONF,
            gpu=USE_GPU,
        )
        logger.info("Plate reader ready.")
    except FileNotFoundError as e:
        logger.error("MODEL NOT FOUND: %s", e)
        logger.error(
            "Set MODEL_PATH env var or place plate_model.pt in this directory."
        )
        # Server still starts so /health can report the problem


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check — use this for readiness probes."""
    return {
        "status": "ok" if reader else "model_not_loaded",
        "model": MODEL_PATH,
        "gpu": USE_GPU,
    }


@app.post("/read-plate")
async def read_plate(file: UploadFile = File(...)):
    """
    Upload an image (JPEG/PNG) and receive detected plate text.

    Returns:
    ```json
    {
      "success": true,
      "plates": [
        {"text": "SGX1234A", "confidence": 0.94, "bbox": [120,340,380,420]}
      ],
      "processing_time_ms": 142
    }
    ```
    """
    if reader is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs and MODEL_PATH.",
        )

    # Validate file type
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got: {content_type}",
        )

    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Process
    start = time.time()
    try:
        plates = reader.read_from_bytes(image_bytes)
    except Exception as e:
        logger.exception("Error processing image")
        raise HTTPException(status_code=500, detail=str(e))
    elapsed_ms = round((time.time() - start) * 1000)

    logger.info(
        "Processed %s — %d plate(s) in %d ms",
        file.filename,
        len(plates),
        elapsed_ms,
    )

    return JSONResponse({
        "success": True,
        "plates": plates,
        "processing_time_ms": elapsed_ms,
    })


# ------------------------------------------------------------------
# Run directly: python server.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
