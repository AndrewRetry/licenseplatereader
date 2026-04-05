import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone
import httpx

from event_publisher import EventPublisher

logger = logging.getLogger(__name__)

_MAX_LOG_ENTRIES = 50

class OrchestratorStreamProcessor:
    """Orchestrates pulling from camera_service and posting to ocr_service."""
    def __init__(
        self,
        publisher: EventPublisher | None,
        camera_url: str,
        ocr_url: str,
        gantry_id: str,
        interval: float,
        cooldown: float,
    ):
        self._publisher = publisher
        self._camera_url = camera_url
        self._ocr_url = ocr_url
        self._gantry_id = gantry_id
        self._interval = interval
        self._cooldown = cooldown

        self._recent_plates: dict[str, float] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._detection_log: deque[dict] = deque(maxlen=_MAX_LOG_ENTRIES)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Orchestrator stream processing started.")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Orchestrator stream processing stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def detection_log(self) -> list[dict]:
        return list(reversed(self._detection_log))

    async def _run_loop(self) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            while self._running:
                start_time = time.monotonic()
                
                try:
                    # 1. Fetch frame from camera service
                    resp = await client.get(f"{self._camera_url}/frame")
                    if resp.status_code == 200:
                        image_bytes = resp.content
                        
                        # 2. Post frame to OCR service
                        files = {'file': ('frame.jpg', image_bytes, 'image/jpeg')}
                        ocr_resp = await client.post(f"{self._ocr_url}/read-plate", files=files)
                        
                        if ocr_resp.status_code == 200:
                            result = ocr_resp.json()
                            plates = result.get("plates", [])
                            
                            self._prune_expired()
                            timestamp = datetime.now(timezone.utc).isoformat()
                            
                            for plate in plates:
                                text = plate["text"]
                                if self._is_duplicate(text):
                                    continue
                                    
                                self._recent_plates[text] = time.monotonic()
                                is_valid = plate.get("checksum_valid", False)
                                
                                self._detection_log.append({
                                    "text": text,
                                    "confidence": plate["confidence"],
                                    "bbox": plate["bbox"],
                                    "gantryId": self._gantry_id,
                                    "timestamp": timestamp,
                                    "checksum_valid": is_valid
                                })
                                
                                logger.info(f"Detected {text} at {self._gantry_id}")
                                
                                if self._publisher:
                                    await self._publisher.publish_plate_detected(
                                        plate_text=text,
                                        confidence=plate["confidence"],
                                        bbox=plate["bbox"],
                                        gantry_id=self._gantry_id,
                                        checksum_valid=is_valid,
                                        frame_timestamp=timestamp,
                                    )
                                    
                except httpx.RequestError as e:
                    logger.warning(f"Error communicating with microservices: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in orchestrator loop: {e}")

                # Sleep to maintain interval
                elapsed = time.monotonic() - start_time
                sleep_time = max(0, self._interval - elapsed)
                await asyncio.sleep(sleep_time)

    def _is_duplicate(self, plate_text: str) -> bool:
        last_seen = self._recent_plates.get(plate_text)
        return False if last_seen is None else (time.monotonic() - last_seen) < self._cooldown

    def _prune_expired(self) -> None:
        now = time.monotonic()
        expired = [t for t, ts in self._recent_plates.items() if (now - ts) >= self._cooldown]
        for t in expired:
            del self._recent_plates[t]
