"""
stream_processor.py — Continuous camera stream → plate detection → event publishing.

Connects to an RTSP/MJPEG camera, grabs frames at a configurable interval,
runs the plate reader pipeline, deduplicates, and publishes events to RabbitMQ.
"""

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np

from plate_reader import PlateReader
from event_publisher import EventPublisher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame quality thresholds
# ---------------------------------------------------------------------------
_MIN_BRIGHTNESS = 10.0     # mean pixel value — below this the frame is unusable
_MIN_SHARPNESS = 5.0       # Laplacian variance — below this the frame is blurry

# ---------------------------------------------------------------------------
# Reconnection back-off
# ---------------------------------------------------------------------------
_RECONNECT_BASE_S = 2.0    # first retry delay
_RECONNECT_MAX_S = 30.0    # ceiling
_RECONNECT_FACTOR = 2.0    # multiplier per attempt

# ---------------------------------------------------------------------------
# Detection log (ring buffer for the dashboard)
# ---------------------------------------------------------------------------
_MAX_LOG_ENTRIES = 50


def _frame_brightness(frame: np.ndarray) -> float:
    """Mean brightness of a BGR frame (0–255)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _frame_sharpness(frame: np.ndarray) -> float:
    """Laplacian variance — higher = sharper.  Motion blur scores low."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class StreamProcessor:
    """Reads a camera stream, detects plates, publishes events."""

    def __init__(
        self,
        reader: PlateReader,
        publisher: EventPublisher | None,
        stream_url: str,
        gantry_id: str = "gantry-01",
        process_interval_s: float = 1.0,
        cooldown_s: float = 30.0,
    ):
        self._reader = reader
        self._publisher = publisher
        self._stream_url = stream_url
        self._gantry_id = gantry_id
        self._process_interval_s = process_interval_s
        self._cooldown_s = cooldown_s

        self._recent_plates: dict[str, float] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._detection_log: deque[dict] = deque(maxlen=_MAX_LOG_ENTRIES)

    def _open_capture(self) -> cv2.VideoCapture:
        if self._stream_url.isdigit():
            index = int(self._stream_url)
            import sys
            if sys.platform == "win32":
                return cv2.VideoCapture(index, cv2.CAP_DSHOW)
            return cv2.VideoCapture(index)
        return cv2.VideoCapture(self._stream_url)

    async def start(self) -> None:
        if self._running:
            logger.warning("Stream processor already running.")
            return

        cap = self._open_capture()
        if not cap.isOpened():
            raise ConnectionError(f"Cannot open camera: {self._stream_url}")
        cap.release()

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Stream started on %s", self._stream_url)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Stream stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def detection_log(self) -> list[dict]:
        return list(reversed(self._detection_log))

    async def _run_loop(self) -> None:
        backoff = _RECONNECT_BASE_S
        while self._running:
            cap = await asyncio.to_thread(self._open_capture)
            if not cap.isOpened():
                logger.warning("Cannot open stream %s — retrying in %.0fs", self._stream_url, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * _RECONNECT_FACTOR, _RECONNECT_MAX_S)
                continue

            backoff = _RECONNECT_BASE_S
            logger.info("Camera stream connected.")

            try:
                await self._process_stream(cap)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Stream processing error — will reconnect")
            finally:
                cap.release()

    async def _process_stream(self, cap: cv2.VideoCapture) -> None:
        last_process_time = 0.0

        while self._running:
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret or frame is None:
                logger.warning("Lost frame — stream may have dropped.")
                return

            now = time.monotonic()
            if now - last_process_time < self._process_interval_s:
                await asyncio.sleep(0.005)
                continue

            last_process_time = now

            if _frame_brightness(frame) < _MIN_BRIGHTNESS or _frame_sharpness(frame) < _MIN_SHARPNESS:
                continue

            plates = await asyncio.to_thread(self._reader.read, frame)
            if not plates:
                continue

            self._prune_expired()
            timestamp = datetime.now(timezone.utc).isoformat()

            for plate in plates:
                text = plate["text"]
                is_valid = plate["checksum_valid"] # NEW

                if self._is_duplicate(text):
                    continue

                self._recent_plates[text] = time.monotonic()

                # Log with checksum status for the dashboard
                self._detection_log.append({
                    "text": text,
                    "confidence": plate["confidence"],
                    "bbox": plate["bbox"],
                    "gantryId": self._gantry_id,
                    "timestamp": timestamp,
                    "checksum_valid": is_valid # NEW
                })

                logger.info(
                    "Detected %s (conf=%.3f, valid=%s)",
                    text, plate["confidence"], is_valid
                )

                if self._publisher:
                    # Publisher now checks checksum_valid before firing event
                    await self._publisher.publish_plate_detected(
                        plate_text=text,
                        confidence=plate["confidence"],
                        bbox=plate["bbox"],
                        gantry_id=self._gantry_id,
                        checksum_valid=is_valid, # NEW
                        frame_timestamp=timestamp,
                    )

    def _is_duplicate(self, plate_text: str) -> bool:
        last_seen = self._recent_plates.get(plate_text)
        return False if last_seen is None else (time.monotonic() - last_seen) < self._cooldown_s

    def _prune_expired(self) -> None:
        now = time.monotonic()
        expired = [t for t, ts in self._recent_plates.items() if (now - ts) >= self._cooldown_s]
        for t in expired:
            del self._recent_plates[t]