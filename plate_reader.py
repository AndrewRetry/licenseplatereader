"""
plate_reader.py — License plate detection and OCR engine, tuned for Singapore plates.

Pipeline:
  Image → YOLO11n (detect plate bbox) → Crop → Normalise colour →
  Preprocess → TrOCR → Clean text → "QX1728A"

Why TrOCR over EasyOCR:
  EasyOCR's general English model was not trained on license plate fonts (Charles
  Wright typeface). It commonly confuses Q→D, X→Y, 0→O on white-on-black plates.
  TrOCR (microsoft/trocr-base-printed) is a transformer trained on printed text
  and handles these characters far more reliably on CPU.

Usage:
  reader = PlateReader("plate_model.pt")
  results = reader.read(image_bgr)
  # [{"text": "QX1728A", "confidence": 0.94, "bbox": [x1,y1,x2,y2]}]
"""

import re
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# TrOCR model — downloaded automatically from HuggingFace on first run (~400 MB)
_TROCR_MODEL_ID = "microsoft/trocr-base-printed"

# Cap crop width before TrOCR to keep CPU inference fast (~300 ms vs 1+ s)
_MAX_CROP_WIDTH = 512


class PlateReader:
    """License plate detector + TrOCR reader."""

    def __init__(
        self,
        model_path: str = "plate_model.pt",
        detect_conf: float = 0.25,
        gpu: bool = False,
    ):
        """
        Args:
            model_path:  Path to YOLO11 .pt weights for plate detection.
            detect_conf: Minimum YOLO detection confidence (0.0–1.0).
            gpu:         Use GPU for inference. False = CPU-only.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Run download_model.py first."
            )
        self.detector = YOLO(model_path)
        self.detect_conf = detect_conf
        logger.info("YOLO model loaded from %s", model_path)

        logger.info("Loading TrOCR processor and model (%s)...", _TROCR_MODEL_ID)
        self.processor = TrOCRProcessor.from_pretrained(_TROCR_MODEL_ID)
        self.trocr = VisionEncoderDecoderModel.from_pretrained(_TROCR_MODEL_ID)
        self.trocr.eval()
        logger.info("TrOCR ready (GPU=%s)", gpu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, image: np.ndarray) -> list[dict]:
        """
        Detect and read all license plates in an image.

        Args:
            image: BGR image (OpenCV format), e.g. from cv2.imread().

        Returns:
            List of dicts with keys:
              - text:       plate string, e.g. "QX1728A"
              - confidence: YOLO detection confidence (0–1)
              - bbox:       [x1, y1, x2, y2] pixel coordinates
        """
        plates = []
        for bbox, conf in self._detect_plates(image):
            x1, y1, x2, y2 = bbox
            crop       = self._crop_plate(image, x1, y1, x2, y2)
            normalised = self._normalise_colour_scheme(crop)
            raw_text   = self._ocr_read(normalised)
            clean_text = self._clean_plate_text(raw_text)

            if clean_text:
                plates.append({
                    "text":       clean_text,
                    "confidence": round(float(conf), 3),
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                })
        return plates

    def read_from_path(self, image_path: str) -> list[dict]:
        """Convenience: read plates from an image file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.read(image)

    def read_from_bytes(self, image_bytes: bytes) -> list[dict]:
        """Convenience: read plates from raw image bytes (e.g., uploaded file)."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image bytes")
        return self.read(image)

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _detect_plates(self, image: np.ndarray) -> list[tuple]:
        """Run YOLO and return [(bbox, confidence)] sorted by confidence desc."""
        results = self.detector(image, conf=self.detect_conf, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                detections.append(([x1, y1, x2, y2], conf))
        detections.sort(key=lambda d: d[1], reverse=True)
        return detections

    def _crop_plate(
        self, image: np.ndarray, x1: float, y1: float, x2: float, y2: float
    ) -> np.ndarray:
        """Crop the plate region with a small padding."""
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.10)
        x1 = max(0, int(x1) - pad_x)
        y1 = max(0, int(y1) - pad_y)
        x2 = min(w, int(x2) + pad_x)
        y2 = min(h, int(y2) + pad_y)
        return image[y1:y2, x1:x2]

    def _normalise_colour_scheme(self, crop: np.ndarray) -> np.ndarray:
        """
        Normalise Singapore plate colour scheme so text is always dark-on-light.

        LTA permits two schemes:
          - White-on-black (white text, black bg) → invert before OCR
          - Black-on-white / black-on-yellow       → use as-is
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < 100:
            logger.debug("White-on-black plate detected — inverting")
            return cv2.bitwise_not(crop)
        return crop

    def _ocr_read(self, crop: np.ndarray) -> str:
        """
        Run TrOCR on the plate crop.

        TrOCR expects a PIL RGB image. We cap the width to _MAX_CROP_WIDTH
        before inference to keep CPU latency acceptable.
        """
        # BGR → RGB PIL image
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Cap width to limit TrOCR inference time on CPU
        if pil_img.width > _MAX_CROP_WIDTH:
            ratio = _MAX_CROP_WIDTH / pil_img.width
            pil_img = pil_img.resize(
                (_MAX_CROP_WIDTH, max(1, int(pil_img.height * ratio))),
                Image.LANCZOS,
            )

        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values
        generated_ids = self.trocr.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _clean_plate_text(self, raw: str) -> str:
        """Uppercase, strip spaces, keep only alphanumeric characters."""
        return re.sub(r"[^A-Z0-9]", "", raw.upper().strip())


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    model = sys.argv[1] if len(sys.argv) > 1 else "plate_model.pt"
    image = sys.argv[2] if len(sys.argv) > 2 else "test_car.jpg"

    reader = PlateReader(model)
    results = reader.read_from_path(image)

    if results:
        for r in results:
            print(f"  Plate: {r['text']}  (conf: {r['confidence']})")
    else:
        print("  No plates detected.")