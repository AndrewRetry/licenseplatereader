"""
stream_processor.py — Continuous camera stream → plate detection → event publishing.

Connects to an RTSP/MJPEG camera, grabs frames at a configurable interval,
runs the plate reader pipeline, deduplicates, and publishes events to RabbitMQ.

Frame selection:
  OpenCV buffers several frames internally.  If we only call cap.read() once
  every N seconds we get a *stale* frame, not the latest one.  The loop
  therefore drains the buffer continuously (fast reads, no processing) and
  only runs the heavy YOLO + TrOCR pipeline at the configured interval.

  Before processing, the frame is scored for quality:
    - Brightness  — rejects black/underexposed frames (camera glitch, cap lens)
    - Sharpness   — Laplacian variance; rejects motion-blurred frames
  This keeps OCR accuracy high without needing the caller to care about it.

Deduplication:
  After a plate is detected, it enters a cooldown window.  The same plate
  text will not fire another RabbitMQ event until the cooldown expires.
  This prevents the Arrival Orchestrator from receiving 30 duplicate events
  while a car sits at the gantry.

Camera reconnection:
  If the stream drops (Wi-Fi blip, camera reboot), the processor retries
  with exponential back-off up to a configurable ceiling, then keeps trying
  at that ceiling indefinitely.  No operator intervention required.
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
        """
        Args:
            reader:              Initialised PlateReader instance.
            publisher:           RabbitMQ publisher (None → log-only mode).
            stream_url:          RTSP or MJPEG URL.
            gantry_id:           Identifier for this gantry (included in events).
            process_interval_s:  Seconds between processing attempts.
            cooldown_s:          Seconds before the same plate can fire again.
        """
        self._reader = reader
        self._publisher = publisher
        self._stream_url = stream_url
        self._gantry_id = gantry_id
        self._process_interval_s = process_interval_s
        self._cooldown_s = cooldown_s

        # plate_text → monotonic timestamp of last publish
        self._recent_plates: dict[str, float] = {}
        self._running = False
        self._task: asyncio.Task | None = None

        # Ring buffer of recent detections — exposed via /detections for the dashboard
        self._detection_log: deque[dict] = deque(maxlen=_MAX_LOG_ENTRIES)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the stream and begin processing in a background task."""
        if self._running:
            logger.warning("Stream processor already running.")
            return

        # Verify the stream can be opened before backgrounding
        cap = cv2.VideoCapture(self._stream_url)
        if not cap.isOpened():
            raise ConnectionError(f"Cannot open camera stream: {self._stream_url}")
        cap.release()

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Stream started  url=%s  gantry=%s  interval=%.1fs  cooldown=%.0fs",
            self._stream_url,
            self._gantry_id,
            self._process_interval_s,
            self._cooldown_s,
        )

    async def stop(self) -> None:
        """Signal the loop to stop and wait for it to finish."""
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
    def stream_url(self) -> str:
        return self._stream_url

    @property
    def recent_plates(self) -> dict[str, float]:
        """Snapshot of the dedup cache — useful for the /health endpoint."""
        return dict(self._recent_plates)

    @property
    def detection_log(self) -> list[dict]:
        """Most recent detections (newest first) — feeds the dashboard."""
        return list(reversed(self._detection_log))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Core loop: drain buffer → score frame → detect → dedup → publish."""
        backoff = _RECONNECT_BASE_S

        while self._running:
            cap = await asyncio.to_thread(cv2.VideoCapture, self._stream_url)

            if not cap.isOpened():
                logger.warning(
                    "Cannot open stream %s — retrying in %.0fs",
                    self._stream_url, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * _RECONNECT_FACTOR, _RECONNECT_MAX_S)
                continue

            # Connected — reset back-off
            backoff = _RECONNECT_BASE_S
            logger.info("Camera stream connected: %s", self._stream_url)

            try:
                await self._process_stream(cap)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Stream processing error — will reconnect")
            finally:
                cap.release()

    async def _process_stream(self, cap: cv2.VideoCapture) -> None:
        """Read frames from an open capture, process at the configured interval."""
        last_process_time = 0.0

        while self._running:
            # Drain the buffer: read every frame so the next read is fresh
            ret, frame = await asyncio.to_thread(cap.read)

            if not ret or frame is None:
                logger.warning("Lost frame — stream may have dropped.")
                return  # exits to _run_loop which will reconnect

            now = time.monotonic()
            if now - last_process_time < self._process_interval_s:
                # Yield back to the event loop between drain reads so HTTP
                # requests and publishes aren't starved.
                await asyncio.sleep(0.005)
                continue

            last_process_time = now

            # --- Frame quality gate ---
            brightness = _frame_brightness(frame)
            if brightness < _MIN_BRIGHTNESS:
                logger.debug("Skipped dark frame  brightness=%.1f", brightness)
                continue

            sharpness = _frame_sharpness(frame)
            if sharpness < _MIN_SHARPNESS:
                logger.debug("Skipped blurry frame  sharpness=%.1f", sharpness)
                continue

            # --- Plate detection (CPU-bound, run in thread) ---
            plates = await asyncio.to_thread(self._reader.read, frame)
            if not plates:
                continue

            # --- Dedup + publish ---
            self._prune_expired()
            timestamp = datetime.now(timezone.utc).isoformat()

            for plate in plates:
                text = plate["text"]

                if self._is_duplicate(text):
                    logger.debug("Duplicate skipped: %s", text)
                    continue

                self._recent_plates[text] = time.monotonic()

                # Append to the dashboard detection log
                self._detection_log.append({
                    "text": text,
                    "confidence": plate["confidence"],
                    "bbox": plate["bbox"],
                    "gantryId": self._gantry_id,
                    "timestamp": timestamp,
                })

                logger.info(
                    "Plate detected  text=%s  conf=%.3f  gantry=%s",
                    text, plate["confidence"], self._gantry_id,
                )

                if self._publisher:
                    await self._publisher.publish_plate_detected(
                        plate_text=text,
                        confidence=plate["confidence"],
                        bbox=plate["bbox"],
                        gantry_id=self._gantry_id,
                        frame_timestamp=timestamp,
                    )

    # ------------------------------------------------------------------
    # Dedup helpers
    # ------------------------------------------------------------------

    def _is_duplicate(self, plate_text: str) -> bool:
        last_seen = self._recent_plates.get(plate_text)
        if last_seen is None:
            return False
        return (time.monotonic() - last_seen) < self._cooldown_s

    def _prune_expired(self) -> None:
        now = time.monotonic()
        expired = [
            text
            for text, ts in self._recent_plates.items()
            if (now - ts) >= self._cooldown_s
        ]
        for text in expired:
            del self._recent_plates[text]