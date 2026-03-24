"""
License plate detection microservice.

Identical REST contract to the Node.js predecessor — drop-in replacement.
Endpoints:
  GET  /api/plate/health
  POST /api/plate/detect        (multipart: image file)
  POST /api/plate/detect/batch  (multipart: multiple image files)
  POST /api/plate/validate      (JSON: { "plate": "SBA1234L" })
"""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.detection import detect
from src.debug import draw_regions
from src.hf_detector import _get_yolos, detect_plate_regions_hf
from src.hf_ocr import _get_trocr
from src.validation import validate_plate
from src.logger import get_logger

load_dotenv()

logger = get_logger("licenseplatereader")

MAX_FILE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


# ── Startup: pre-load HuggingFace models so the first request is fast ────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up HuggingFace models (YOLOS + TrOCR)…")
    try:
        _get_yolos()
        _get_trocr()
        logger.info("HuggingFace models ready")
    except Exception as exc:
        logger.warning("Model warmup failed — will retry on first request", error=str(exc))
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="License Plate Reader",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _read_image(file: UploadFile) -> bytes:
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported image type: {file.content_type}")
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, "Image exceeds size limit")
    return data


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/plate/health")
def health():
    return {"status": "ok", "version": "4.0.0", "engine": "yolos+trocr", "models": {
        "detection": "nickmuchi/yolos-small-rego-plates-detection",
        "ocr": "microsoft/trocr-base-printed",
    }}


@app.post("/api/plate/detect")
async def detect_single(image: Annotated[UploadFile, File()]):
    request_id = str(uuid.uuid4())
    image_bytes = await _read_image(image)

    try:
        result = detect(image_bytes)
    except Exception as exc:
        logger.exception("Detection error", request_id=request_id)
        raise HTTPException(500, str(exc)) from exc

    return JSONResponse({"requestId": request_id, **result})


@app.post("/api/plate/detect/batch")
async def detect_batch(images: Annotated[list[UploadFile], File()]):
    if len(images) > 10:
        raise HTTPException(400, "Maximum 10 images per batch")

    results = []
    for file in images:
        request_id = str(uuid.uuid4())
        try:
            image_bytes = await _read_image(file)
            result = detect(image_bytes)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Batch item error", request_id=request_id)
            result = {"success": False, "error": str(exc)}
        results.append({"requestId": request_id, "filename": file.filename, **result})

    return JSONResponse({"results": results, "count": len(results)})


class ValidateRequest(BaseModel):
    plate: str


@app.post("/api/plate/validate")
def validate(body: ValidateRequest):
    result = validate_plate(body.plate)
    if result is None:
        return JSONResponse({"valid": False, "plate": body.plate})
    return JSONResponse({
        "valid": True,
        "plate": result.plate,
        "formatted": result.formatted,
        "format": result.format,
        "checksumValid": result.checksum_valid,
        "confidence": result.confidence,
    })


@app.post("/api/plate/debug")
async def debug_detect(image: Annotated[UploadFile, File()]):
    """
    Same as /detect but also returns the input image with detected regions
    drawn on it as a base64-encoded JPEG in the response.

    Intended for development/tuning only — not for production traffic.
    """
    request_id = str(uuid.uuid4())
    image_bytes = await _read_image(image)

    try:
        regions = detect_plate_regions_hf(image_bytes)
        result   = detect(image_bytes)
        best_plate = result.get("best", {}).get("plate") if result.get("best") else None
        annotated = draw_regions(image_bytes, regions, best_plate)
    except Exception as exc:
        logger.exception("Debug detection error", request_id=request_id)
        raise HTTPException(500, str(exc)) from exc

    import base64
    return JSONResponse({
        "requestId": request_id,
        **result,
        "debug": {
            "regionsDetected": len(regions),
            "annotatedImage": "data:image/jpeg;base64," + base64.b64encode(annotated).decode(),
        },
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "3001"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
