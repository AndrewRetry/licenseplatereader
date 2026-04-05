import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from plate_reader import PlateReader

MODEL_PATH = os.getenv("MODEL_PATH", "plate_model.pt")
DETECT_CONF = float(os.getenv("DETECT_CONF", "0.5"))
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr-service")

_state = {"reader": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the plate reader model
    logger.info("Loading plate reader model from %s …", MODEL_PATH)
    try:
        _state["reader"] = PlateReader(
            model_path=MODEL_PATH,
            detect_conf=DETECT_CONF,
            gpu=USE_GPU,
        )
        logger.info("Plate reader ready.")
    except Exception as e:
        logger.error("MODEL NOT FOUND OR LOAD ERROR: %s", e)

    yield
    _state["reader"] = None

app = FastAPI(title="OCR Inference Service", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok" if _state["reader"] else "model_error"}

@app.post("/read-plate")
async def read_plate(file: UploadFile = File(...)):
    reader = _state["reader"]
    if reader is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    start = time.time()
    try:
        plates = reader.read_from_bytes(image_bytes)
    except Exception as e:
        logger.exception("Error processing image")
        raise HTTPException(status_code=500, detail=str(e))
    
    elapsed_ms = round((time.time() - start) * 1000)
    logger.info("Processed %d plates in %d ms", len(plates), elapsed_ms)

    return JSONResponse({
        "success": True,
        "plates": plates,
        "processingTimeMs": elapsed_ms,
    })
