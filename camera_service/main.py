import os
import time
import logging
import threading
import cv2

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camera-service")

STREAM_URL = os.getenv("STREAM_URL", "")

class CameraManager:
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        if not self.stream_url:
            logger.info("No STREAM_URL configured.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

    def _open_capture(self):
         if self.stream_url.isdigit():
             return cv2.VideoCapture(int(self.stream_url))
         return cv2.VideoCapture(self.stream_url)

    def _capture_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open_capture()
                if not self.cap.isOpened():
                    logger.warning("Failed to open stream, retrying in 2s...")
                    time.sleep(2)
                    continue
                logger.info("Connected to stream.")

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Stream dropped, reconnecting...")
                self.cap.release()
                self.cap = None
                time.sleep(1)
                continue

            with self.lock:
                self.latest_frame = frame

            time.sleep(0.01)

camera_manager = None
app = FastAPI(title="Camera Service")

@app.on_event("startup")
def startup_event():
    global camera_manager
    camera_manager = CameraManager(STREAM_URL)
    camera_manager.start()

@app.on_event("shutdown")
def shutdown_event():
    if camera_manager:
        camera_manager.stop()

@app.get("/health")
def health():
    if not camera_manager or not camera_manager.stream_url:
        return {"status": "no_stream_url"}
    with camera_manager.lock:
        status = "ok" if camera_manager.latest_frame is not None else "connecting"
    return {"status": status}

@app.get("/frame")
def get_frame():
    """Returns the latest frame as a clear JPEG image for the Orchestrator/OCR."""
    if not camera_manager:
         raise HTTPException(status_code=503, detail="Camera manager not initialized")
    with camera_manager.lock:
        if camera_manager.latest_frame is None:
            raise HTTPException(status_code=503, detail="No frame available yet")
        # Quality 100 for best OCR results
        ret, buffer = cv2.imencode('.jpg', camera_manager.latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to encode frame")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")

def generate_mjpeg_stream():
    """Generator function yielding MJPEG frames for public broadcast/dashboard."""
    while True:
        try:
            if not camera_manager:
                time.sleep(0.1)
                continue
            with camera_manager.lock:
                frame = camera_manager.latest_frame
                
            if frame is None:
                time.sleep(0.1)
                continue
                
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                time.sleep(0.1)
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05) # Cap stream framerate
        except Exception as e:
            logger.error(f"MJPEG stream error: {e}")
            break

@app.get("/stream")
def get_stream():
    """Exposes the camera feed as an MJPEG stream"""
    if not camera_manager:
         raise HTTPException(status_code=503, detail="Camera manager not initialized")
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/config")
def update_config(stream_url: str):
    """Allows updating the stream url at runtime"""
    global camera_manager
    if camera_manager:
        camera_manager.stop()
    os.environ["STREAM_URL"] = stream_url
    camera_manager = CameraManager(stream_url)
    camera_manager.start()
    return {"status": "updated", "stream_url": stream_url}
