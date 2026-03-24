"""
HuggingFace YOLOS plate detector.

Model: nickmuchi/yolos-small-rego-plates-detection
  - YOLOS (You Only Look at One Sequence) fine-tuned on licence plates
  - Apache 2.0 licence, free, runs CPU-only
  - Auto-downloads from HuggingFace Hub on first use (~110 MB)

Strategy:
  1. YOLOS transformer detection  ← primary (neural, no hand-tuned thresholds)
  2. OpenCV contour fallback       ← if YOLOS finds nothing
  3. Horizontal strip fallback     ← guaranteed last resort
"""

import io
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .logger import get_logger
from .plate_detector import PlateRegion, _contour_candidates, _strip_fallback

logger = get_logger(__name__)

_YOLOS_MODEL_ID = "nickmuchi/yolos-small-rego-plates-detection"
_CONFIDENCE_THRESHOLD = 0.45   # minimum YOLOS score to accept a detection


@lru_cache(maxsize=1)
def _get_yolos():
    """
    Lazy singleton — loads model once, cached for the process lifetime.
    Import is deferred so the module loads without torch at import time.
    """
    from transformers import YolosForObjectDetection, YolosImageProcessor  # noqa: PLC0415

    logger.info("Loading YOLOS plate detector from HuggingFace Hub", model=_YOLOS_MODEL_ID)
    processor = YolosImageProcessor.from_pretrained(_YOLOS_MODEL_ID)
    model = YolosForObjectDetection.from_pretrained(_YOLOS_MODEL_ID)
    model.eval()
    logger.info("YOLOS detector ready")
    return processor, model


def _yolos_candidates(img_bgr: np.ndarray) -> list[PlateRegion]:
    """Run YOLOS transformer detection on a BGR numpy image."""
    import torch  # noqa: PLC0415

    try:
        processor, model = _get_yolos()
    except Exception as exc:
        logger.warning("YOLOS model load failed, skipping", error=str(exc))
        return []

    h, w = img_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    inputs = processor(images=pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[h, w]])
    results = processor.post_process_object_detection(
        outputs,
        threshold=_CONFIDENCE_THRESHOLD,
        target_sizes=target_sizes,
    )[0]

    candidates: list[PlateRegion] = []

    scores = results["scores"].tolist()
    boxes  = results["boxes"].tolist()

    for score, box in zip(scores, boxes):
        x0, y0, x1, y1 = [int(round(v)) for v in box]

        # Clamp to image bounds
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)

        bw, bh = x1 - x0, y1 - y0
        if bw < 10 or bh < 5:
            continue

        # Small pad
        pad_x = max(0, int(bw * 0.04))
        pad_y = max(0, int(bh * 0.08))
        x0p = max(0, x0 - pad_x)
        y0p = max(0, y0 - pad_y)
        x1p = min(w, x1 + pad_x)
        y1p = min(h, y1 + pad_y)

        crop = img_bgr[y0p:y1p, x0p:x1p]
        if crop.size == 0:
            continue

        candidates.append(PlateRegion(
            crop=crop,
            x=x0p, y=y0p,
            w=x1p - x0p, h=y1p - y0p,
            method="yolos",
            score=float(score),
        ))

    candidates.sort(key=lambda r: r.score, reverse=True)
    logger.debug("YOLOS detections", count=len(candidates))
    return candidates


def detect_plate_regions_hf(image_bytes: bytes) -> list[PlateRegion]:
    """
    Entry point for the HuggingFace detection pipeline.

    Priority order:
      1. YOLOS neural detection  (best)
      2. OpenCV contour detection (good on high-contrast frames)
      3. Horizontal strip fallback (always returns ≥4 regions)
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    # 1. YOLOS
    yolos_hits = _yolos_candidates(img)
    if yolos_hits:
        logger.debug("Using YOLOS regions", count=len(yolos_hits))
        return yolos_hits

    # 2. OpenCV contour fallback
    contour_hits = _contour_candidates(img)
    if contour_hits:
        logger.debug("YOLOS found nothing, using OpenCV contours", count=len(contour_hits))
        return contour_hits

    # 3. Strip fallback
    logger.debug("No detector found regions, using strip fallback")
    return _strip_fallback(img)
