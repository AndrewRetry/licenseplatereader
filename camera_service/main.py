import os
import time
import logging
import threading
import cv2
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camera-service")

STREAM_URL = os.getenv("STREAM_URL", "http://192.168.50.224:8080/video")


class CameraManager:
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.cap: cv2.VideoCapture | None = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self):
        if not self.stream_url:
            logger.info("No STREAM_URL configured — camera idle.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        if self.cap:
            self.cap.release()
            self.cap = None

    def _open_capture(self) -> cv2.VideoCapture:
        if self.stream_url.isdigit():
            return cv2.VideoCapture(int(self.stream_url))
        return cv2.VideoCapture(self.stream_url)

    def _capture_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open_capture()
                if not self.cap.isOpened():
                    logger.warning("Failed to open stream — retrying in 2s…")
                    time.sleep(2)
                    continue
                logger.info("Connected to stream: %s", self.stream_url)

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Stream dropped — reconnecting…")
                self.cap.release()
                self.cap = None
                time.sleep(1)
                continue

            with self.lock:
                self.latest_frame = frame

            time.sleep(0.01)


_camera: CameraManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _camera
    _camera = CameraManager(STREAM_URL)
    _camera.start()
    yield
    if _camera:
        _camera.stop()


app = FastAPI(title="Camera Service", lifespan=lifespan)


@app.get("/health")
def health():
    if not _camera or not _camera.stream_url:
        return {"status": "no_stream_url"}
    with _camera.lock:
        status = "ok" if _camera.latest_frame is not None else "connecting"
    return {"status": status}


@app.get("/frame")
def get_frame():
    if not _camera:
        raise HTTPException(status_code=503, detail="Camera not initialised")
    with _camera.lock:
        if _camera.latest_frame is None:
            raise HTTPException(status_code=503, detail="No frame available yet")
        ret, buffer = cv2.imencode(
            ".jpg", _camera.latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        )
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


def _mjpeg_generator():
    while True:
        if not _camera:
            time.sleep(0.1)
            continue
        with _camera.lock:
            frame = _camera.latest_frame
        if frame is None:
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        )
        if not ret:
            time.sleep(0.1)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        time.sleep(0.05)


@app.get("/stream")
def get_stream():
    if not _camera:
        raise HTTPException(status_code=503, detail="Camera not initialised")
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/config")
def update_config(stream_url: str):
    global _camera
    if _camera:
        _camera.stop()
    _camera = CameraManager(stream_url)
    _camera.start()
    return {"status": "updated", "stream_url": stream_url}