"""
Plate region detector.

Strategy (no manual reticle required):
  1. Contour-based detection — finds rectangular regions with plate-like
     aspect ratios using edge detection and morphological ops.
  2. Haar cascade fallback — uses OpenCV's bundled cascade classifier.
  3. Horizontal-strip fallback — scans likely vehicle positions if both
     above methods find nothing.

Returns a list of (crop_buffer, label, confidence_score) tuples ordered
by detection confidence.
"""

import io
import os

from .logger import get_logger
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = get_logger(__name__)

# SG plates are ~52cm × 11.4cm → aspect ratio ~4.56
# Allow generous range to handle angles and partial plates
MIN_ASPECT = 2.0
MAX_ASPECT = 7.5

# Minimum plate area relative to image area
MIN_AREA_RATIO = 0.002   # plate must be at least 0.2% of the frame
MAX_AREA_RATIO = 0.25    # but no more than 25%

# Haar cascade ships with every opencv-python install
_CASCADE_PATH = os.path.join(
    os.path.dirname(cv2.__file__),
    "data",
    "haarcascade_russian_plate_number.xml",
)


@dataclass
class PlateRegion:
    crop: np.ndarray       # BGR crop of the detected region
    x: int
    y: int
    w: int
    h: int
    method: str            # 'contour' | 'haar' | 'strip'
    score: float           # higher = more confident


def _load_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _contour_candidates(img: np.ndarray) -> list[PlateRegion]:
    """
    Classic ANPR contour approach:
      grayscale → bilateral filter → Canny → dilate → findContours
      → filter by aspect ratio + area
    """
    h, w = img.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bilateral filter: smooths noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)

    edges = cv2.Canny(filtered, threshold1=30, threshold2=200)

    # Dilate edges to close small gaps on plate borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[PlateRegion] = []

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box_w, box_h = sorted(rect[1])        # ensure box_h <= box_w
        if box_h < 5:
            continue

        aspect = box_w / box_h
        area = box_w * box_h
        area_ratio = area / img_area

        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO):
            continue

        # Use upright bounding rect for the crop (simpler, works well enough)
        bx, by, bw, bh = cv2.boundingRect(cnt)

        # Slight padding
        pad_x = max(0, int(bw * 0.05))
        pad_y = max(0, int(bh * 0.10))
        bx = max(0, bx - pad_x)
        by = max(0, by - pad_y)
        bw = min(w - bx, bw + 2 * pad_x)
        bh = min(h - by, bh + 2 * pad_y)

        crop = img[by: by + bh, bx: bx + bw]
        if crop.size == 0:
            continue

        # Score: prefer larger plates in lower half of frame (where cars park)
        position_bonus = 1.3 if (by + bh / 2) > (h * 0.5) else 1.0
        score = area_ratio * position_bonus

        candidates.append(PlateRegion(
            crop=crop, x=bx, y=by, w=bw, h=bh,
            method="contour", score=score,
        ))

    candidates.sort(key=lambda r: r.score, reverse=True)
    return candidates[:6]   # top-6 contour candidates


def _haar_candidates(img: np.ndarray) -> list[PlateRegion]:
    """Haar cascade detector — fallback when contours find nothing useful."""
    if not os.path.exists(_CASCADE_PATH):
        logger.debug("Haar cascade not found, skipping")
        return []

    cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 15),
    )

    candidates: list[PlateRegion] = []

    if len(detections) == 0:
        return candidates

    h, w = img.shape[:2]
    for (bx, by, bw, bh) in detections:
        crop = img[by: by + bh, bx: bx + bw]
        area_ratio = (bw * bh) / (w * h)
        candidates.append(PlateRegion(
            crop=crop, x=bx, y=by, w=bw, h=bh,
            method="haar", score=area_ratio,
        ))

    candidates.sort(key=lambda r: r.score, reverse=True)
    return candidates


def _strip_fallback(img: np.ndarray) -> list[PlateRegion]:
    """
    Last resort: return horizontal strips from likely plate positions.
    Mirrors the old Node.js region logic but as a genuine fallback only.
    """
    h, w = img.shape[:2]
    strips = [
        ("lower",  0.55, 0.38),
        ("mid",    0.20, 0.60),
        ("centre", 0.30, 0.50),
        ("full",   0.00, 1.00),
    ]
    results = []
    for label, top_frac, height_frac in strips:
        y0 = int(h * top_frac)
        sh = max(10, int(h * height_frac))
        crop = img[y0: y0 + sh, 0:w]
        results.append(PlateRegion(
            crop=crop, x=0, y=y0, w=w, h=sh,
            method=f"strip_{label}", score=0.1,
        ))
    return results


def detect_plate_regions(image_bytes: bytes) -> list[PlateRegion]:
    """
    Entry point. Returns candidate plate regions ordered by confidence.
    Always returns at least one region (strip fallback guarantees this).
    """
    img = _load_image(image_bytes)

    contour_hits = _contour_candidates(img)
    haar_hits    = _haar_candidates(img)

    # Merge: contour candidates first (generally better), then haar
    combined = contour_hits + [h for h in haar_hits if h not in contour_hits]

    if combined:
        logger.debug("Detected plate region(s) via CV", count=len(combined))
        return combined

    logger.debug("No CV regions found, using strip fallback")
    return _strip_fallback(img)
