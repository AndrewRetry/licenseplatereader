import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone

import cv2
import httpx
import numpy as np

from event_publisher import EventPublisher

logger = logging.getLogger(__name__)

_MAX_LOG_ENTRIES   = 50
_SHARPNESS_MIN     = 80.0   # Laplacian variance below this = too blurry to bother
_BRIGHTNESS_MIN    = 30     # mean pixel value below this = frame is too dark


def _frame_is_usable(jpeg_bytes: bytes) -> tuple[bool, str]:
    """
    Decode JPEG and run lightweight quality checks before paying OCR cost.
    Returns (ok, reason_if_rejected).
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return False, "decode_failed"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    if brightness < _BRIGHTNESS_MIN:
        return False, f"too_dark ({brightness:.0f})"

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < _SHARPNESS_MIN:
        return False, f"too_blurry ({sharpness:.0f})"

    return True, "ok"


class OrchestratorStreamProcessor:
    """Polls camera_service → quality-gates frame → OCR service → dedup → publish."""

    def __init__(
        self,
        publisher: EventPublisher | None,
        camera_url: str,
        ocr_url: str,
        gantry_id: str,
        interval: float,
        cooldown: float,
    ):
        self._publisher    = publisher
        self._camera_url   = camera_url
        self._ocr_url      = ocr_url
        self._gantry_id    = gantry_id
        self._interval     = interval
        self._cooldown     = cooldown

        self._recent_plates: dict[str, float] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._detection_log: deque[dict] = deque(maxlen=_MAX_LOG_ENTRIES)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Stream processor started  gantry=%s  interval=%.1fs  cooldown=%.0fs",
            self._gantry_id, self._interval, self._cooldown,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Stream processor stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def detection_log(self) -> list[dict]:
        return list(reversed(self._detection_log))

    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            while self._running:
                start = time.monotonic()
                await self._process_one_frame(client)
                elapsed = time.monotonic() - start
                await asyncio.sleep(max(0.0, self._interval - elapsed))

    async def _process_one_frame(self, client: httpx.AsyncClient) -> None:
        # 1. Fetch frame
        try:
            resp = await client.get(f"{self._camera_url}/frame")
        except httpx.RequestError as e:
            logger.warning("Camera service unreachable: %s", e)
            return

        if resp.status_code != 200:
            logger.warning("Camera returned HTTP %d", resp.status_code)
            return

        image_bytes = resp.content

        # 2. Quality gate — skip before paying OCR cost
        usable, reason = _frame_is_usable(image_bytes)
        if not usable:
            logger.debug("Frame rejected: %s", reason)
            return

        # 3. Send to OCR service
        try:
            ocr_resp = await client.post(
                f"{self._ocr_url}/read-plate",
                files={"file": ("frame.jpg", image_bytes, "image/jpeg")},
            )
        except httpx.RequestError as e:
            logger.warning("OCR service unreachable: %s", e)
            return

        if ocr_resp.status_code != 200:
            logger.warning("OCR returned HTTP %d: %s", ocr_resp.status_code, ocr_resp.text)
            return

        plates = ocr_resp.json().get("plates", [])
        if not plates:
            return

        # 4. Dedup + publish
        self._prune_expired()
        timestamp = datetime.now(timezone.utc).isoformat()

        for plate in plates:
            text     = plate["text"]
            is_valid = plate.get("checksum_valid", False)

            if self._is_duplicate(text):
                logger.info("Dedup skip (cooldown active): %s", text)   # ← was DEBUG
                continue

            self._recent_plates[text] = time.monotonic()
            self._detection_log.append({
                "text":           text,
                "confidence":     plate["confidence"],
                "bbox":           plate["bbox"],
                "gantryId":       self._gantry_id,
                "timestamp":      timestamp,
                "checksum_valid": is_valid,
            })

            logger.info(
                "Detected %s  conf=%.2f  checksum=%s",
                text, plate["confidence"], "OK" if is_valid else "FAIL",
            )

            if self._publisher:
                await self._publisher.publish_plate_detected(
                    plate_text=text,
                    confidence=plate["confidence"],
                    bbox=plate["bbox"],
                    gantry_id=self._gantry_id,
                    checksum_valid=is_valid,
                    frame_timestamp=timestamp,
                )
            else:
                logger.warning(          # ← was silent
                    "No publisher — plate detected but NOT published: %s (conf=%.2f)",
                    text, plate["confidence"],
                )

    def _is_duplicate(self, plate_text: str) -> bool:
        last_seen = self._recent_plates.get(plate_text)
        return last_seen is not None and (time.monotonic() - last_seen) < self._cooldown

    def _prune_expired(self) -> None:
        now = time.monotonic()
        expired = [t for t, ts in self._recent_plates.items() if (now - ts) >= self._cooldown]
        for t in expired:
            del self._recent_plates[t]