"""
Detection pipeline orchestrator.

Flow:
  image bytes
    → hf_detector   (YOLOS: find where the plate is)
    → hf_ocr        (TrOCR: read the text in each region)
    → validation    (SG plate format + checksum)
    → score + rank  (best candidate wins)

Fallback chain inside each layer means the service degrades gracefully
even if the HuggingFace models haven't been downloaded yet.
"""

import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

from .hf_detector import detect_plate_regions_hf
from .hf_ocr import ocr_crop_hf
from .logger import get_logger
from .validation import PlateResult, validate_plate

logger = get_logger(__name__)

MIN_OCR_CONFIDENCE = 25.0   # discard TrOCR reads below this score


@dataclass
class Candidate:
    plate: str
    formatted: str
    prefix: str
    digits: str
    checksum: Optional[str]
    checksum_valid: Optional[bool]
    format: str
    confidence: str          # high | medium | low
    ocr_confidence: float
    method: str              # yolos | contour | haar | strip_*
    raw_ocr_text: str
    score: float


def _compute_score(ocr_conf: float, validation: PlateResult) -> float:
    score = ocr_conf
    if validation.checksum_valid is True:
        score += 15
    elif validation.checksum_valid is False:
        score -= 20
    return score


def detect(image_bytes: bytes) -> dict[str, Any]:
    """
    Full detection pipeline on raw image bytes.

    Returns a dict compatible with the service REST contract.
    """
    start = time.time()

    # Step 1: Localise plate regions (YOLOS → contour → strip)
    regions = detect_plate_regions_hf(image_bytes)
    logger.debug("Regions ready", count=len(regions))

    # Step 2: OCR each region (TrOCR)
    seen: dict[str, Candidate] = {}

    for region in regions:
        result = ocr_crop_hf(region.crop)
        if result is None:
            continue

        text, ocr_conf = result
        if ocr_conf < MIN_OCR_CONFIDENCE:
            continue

        validation = validate_plate(text)
        if validation is None:
            continue

        score = _compute_score(ocr_conf, validation)

        if validation.plate in seen:
            if score > seen[validation.plate].score:
                existing = seen[validation.plate]
                existing.score = score
                existing.ocr_confidence = ocr_conf
                existing.method = region.method
                existing.raw_ocr_text = text
            continue

        seen[validation.plate] = Candidate(
            plate=validation.plate,
            formatted=validation.formatted,
            prefix=validation.prefix,
            digits=validation.digits,
            checksum=validation.checksum,
            checksum_valid=validation.checksum_valid,
            format=validation.format,
            confidence=validation.confidence,
            ocr_confidence=ocr_conf,
            method=region.method,
            raw_ocr_text=text,
            score=score,
        )

    # Step 3: Rank
    ranked = sorted(seen.values(), key=lambda c: c.score, reverse=True)
    elapsed_ms = round((time.time() - start) * 1000)

    def _serialise(c: Candidate) -> dict:
        d = asdict(c)
        d["ocrConfidence"] = d.pop("ocr_confidence")
        d["checksumValid"] = d.pop("checksum_valid")
        return d

    best = _serialise(ranked[0]) if ranked else None

    logger.info(
        "Detection complete",
        found=len(ranked),
        best=best["plate"] if best else None,
        elapsed_ms=elapsed_ms,
    )

    return {
        "success": len(ranked) > 0,
        "elapsedMs": elapsed_ms,
        "best": best,
        "candidates": [_serialise(c) for c in ranked[1:]],
        "meta": {
            "regionsInspected": len(regions),
            "ocrAttempts": len(regions),
            "detectionEngine": "yolos",
            "ocrEngine": "trocr",
        },
    }
