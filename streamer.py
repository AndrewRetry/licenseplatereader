"""
Gantry camera streamer.

Captures from a USB webcam, triggers plate detection on motion,
and calls a webhook when a plate is confirmed.

Flow:
  webcam frame
    → MOG2 background subtraction  (is there a car-sized object?)
    → POST /api/plate/detect        (only when motion detected)
    → confidence filter             (high → fire, medium → fire + review flag)
    → cooldown check                (don't re-fire same plate within N seconds)
    → POST webhook

Run:
  python streamer.py

Config (env / .env):
  CAMERA_INDEX          int     default 0
  CAMERA_BACKEND        str     default msmf   (msmf | dshow | auto)
  DETECT_URL            str     default http://localhost:3001/api/plate/detect
  WEBHOOK_URL           str     default http://localhost:9000/gantry/webhook
  COOLDOWN_SECONDS      int     default 10
  MOTION_AREA_MIN       int     default 8000   (px² — tune to your frame size)
  JPEG_QUALITY          int     default 85
  POLL_INTERVAL_MS      int     default 500    (ms between detection calls while motion active)
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from src.logger import get_logger

load_dotenv()

logger = get_logger("streamer")

# ── Config ─────────────────────────────────────────────────────────────────────

CAMERA_INDEX     = int(os.getenv("CAMERA_INDEX",     "0"))
CAMERA_BACKEND   = os.getenv("CAMERA_BACKEND",  "msmf").lower()
DETECT_URL       = os.getenv("DETECT_URL",      "http://localhost:3001/api/plate/detect")
WEBHOOK_URL      = os.getenv("WEBHOOK_URL",     "http://localhost:9000/gantry/webhook")
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "10"))
MOTION_AREA_MIN  = int(os.getenv("MOTION_AREA_MIN",  "8000"))
JPEG_QUALITY     = int(os.getenv("JPEG_QUALITY",     "85"))
POLL_INTERVAL_MS = int(os.getenv("POLL_INTERVAL_MS", "500"))

_BACKENDS = {
    "msmf":  cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "auto":  cv2.CAP_ANY,
}


def _open_camera(index: int, backend_name: str) -> cv2.VideoCapture:
    """
    Open camera with the specified backend.
    Falls back through MSMF → DSHOW → AUTO if the chosen backend fails.
    """
    order = [backend_name] + [b for b in ("msmf", "dshow", "auto") if b != backend_name]

    for name in order:
        backend = _BACKENDS.get(name, cv2.CAP_ANY)
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                logger.info("Camera opened", index=index, backend=name)
                return cap
            cap.release()

    raise RuntimeError(
        f"Cannot open camera index {index} with any backend. "
        f"Run  python find_camera.py  to discover a working index."
    )


# ── Cooldown tracker ───────────────────────────────────────────────────────────

class CooldownTracker:
    """
    Remembers recently fired plates.
    Thread-safe — detection and webhook calls run in a pool.
    """

    def __init__(self, seconds: int):
        self._seconds = seconds
        self._last_fired: dict[str, float] = {}
        self._lock = threading.Lock()

    def should_fire(self, plate: str) -> bool:
        now = time.monotonic()
        with self._lock:
            last = self._last_fired.get(plate, 0.0)
            if now - last >= self._seconds:
                self._last_fired[plate] = now
                return True
            return False

    def remaining(self, plate: str) -> float:
        now = time.monotonic()
        with self._lock:
            last = self._last_fired.get(plate, 0.0)
            return max(0.0, self._seconds - (now - last))


# ── Webhook dispatch ───────────────────────────────────────────────────────────

def _call_webhook(plate: str, result: dict, requires_review: bool) -> None:
    payload = {
        "event":          "plate_detected",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "plate":          plate,
        "formatted":      result.get("formatted"),
        "confidence":     result.get("confidence"),
        "checksumValid":  result.get("checksumValid"),
        "format":         result.get("format"),
        "ocrConfidence":  result.get("ocrConfidence"),
        "requiresReview": requires_review,
    }

    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        r.raise_for_status()
        logger.info(
            "Webhook delivered",
            plate=plate,
            requires_review=requires_review,
            status=r.status_code,
        )
    except requests.RequestException as exc:
        logger.error("Webhook failed", plate=plate, error=str(exc))


# ── Detection call ─────────────────────────────────────────────────────────────

def _detect_frame(frame: np.ndarray, cooldown: CooldownTracker) -> None:
    """
    Encode frame, POST to detect service, dispatch webhook if warranted.
    Runs in a thread — must not block the camera loop.
    """
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    try:
        r = requests.post(
            DETECT_URL,
            files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        logger.warning("Detect request failed", error=str(exc))
        return

    if not data.get("success"):
        return

    best       = data["best"]
    plate      = best["plate"]
    confidence = best["confidence"]

    if confidence == "low":
        logger.debug("Low confidence, skipping", plate=plate)
        return

    requires_review = confidence == "medium"

    if not cooldown.should_fire(plate):
        remaining = cooldown.remaining(plate)
        logger.debug("Cooldown active, skipping", plate=plate, remaining_s=round(remaining, 1))
        return

    logger.info(
        "Plate confirmed",
        plate=plate,
        confidence=confidence,
        requires_review=requires_review,
    )

    _call_webhook(plate, best, requires_review)


# ── Motion detection ───────────────────────────────────────────────────────────

def _has_motion(fgmask: np.ndarray) -> bool:
    """
    True when the foreground mask contains a contour large enough to be a car.
    Morphological cleanup removes sensor noise before contour search.
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,  kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) >= MOTION_AREA_MIN for c in contours)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run() -> None:
    logger.info(
        "Streamer starting",
        camera=CAMERA_INDEX,
        backend=CAMERA_BACKEND,
        detect_url=DETECT_URL,
        webhook_url=WEBHOOK_URL,
        cooldown_s=COOLDOWN_SECONDS,
        motion_area_min=MOTION_AREA_MIN,
    )

    cap      = _open_camera(CAMERA_INDEX, CAMERA_BACKEND)
    bg_sub   = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
    cooldown = CooldownTracker(COOLDOWN_SECONDS)
    executor = ThreadPoolExecutor(max_workers=2)

    poll_interval_s = POLL_INTERVAL_MS / 1000.0
    last_detection  = 0.0
    motion_active   = False

    logger.info("Camera open — watching for motion")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame, retrying…")
                time.sleep(0.1)
                continue

            fgmask = bg_sub.apply(frame)
            motion = _has_motion(fgmask)

            if motion and not motion_active:
                logger.info("Motion detected — starting detection")
            elif not motion and motion_active:
                logger.info("Motion cleared")

            motion_active = motion

            now = time.monotonic()
            if motion and (now - last_detection) >= poll_interval_s:
                last_detection = now
                executor.submit(_detect_frame, frame.copy(), cooldown)

            time.sleep(0.03)

    except KeyboardInterrupt:
        logger.info("Streamer stopped")
    finally:
        cap.release()
        executor.shutdown(wait=False)


if __name__ == "__main__":
    run()