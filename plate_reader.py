"""
plate_reader.py — Core license plate detection and OCR engine.

Pipeline:
  Image → YOLOv8n (detect plate bbox) → Crop → Preprocess → EasyOCR → Clean text

Usage:
  reader = PlateReader("plate_model.pt")
  results = reader.read(image_bgr)
  # [{"text": "SGX1234A", "confidence": 0.94, "bbox": [x1,y1,x2,y2]}]
"""

import re
import logging
from pathlib import Path

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PlateReader:
    """Lightweight license plate detector + OCR reader."""

    def __init__(
        self,
        model_path: str = "plate_model.pt",
        detect_conf: float = 0.5,
        ocr_languages: list[str] | None = None,
        gpu: bool = False,
    ):
        """
        Args:
            model_path:     Path to YOLOv8 .pt weights trained on license plates.
            detect_conf:    Minimum confidence for plate detection (0.0–1.0).
            ocr_languages:  Language codes for EasyOCR. Default ["en"].
            gpu:            Use GPU for inference. False = CPU-only (fine for gantry).
        """
        # --- Load YOLO detector ---
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "See README.md Phase 2 to train or download one."
            )
        self.detector = YOLO(model_path)
        self.detect_conf = detect_conf
        logger.info("YOLO model loaded from %s", model_path)

        # --- Load EasyOCR reader ---
        langs = ocr_languages or ["en"]
        self.ocr = easyocr.Reader(langs, gpu=gpu)
        logger.info("EasyOCR loaded for languages: %s (GPU=%s)", langs, gpu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, image: np.ndarray) -> list[dict]:
        """
        Detect and read all license plates in an image.

        Args:
            image: BGR image (OpenCV format), e.g. from cv2.imread().

        Returns:
            List of dicts, each with keys:
              - text:       cleaned plate string, e.g. "SGX1234A"
              - confidence: YOLO detection confidence (0–1)
              - bbox:       [x1, y1, x2, y2] pixel coordinates
        """
        plates = []

        # Step 1: Detect plate bounding boxes
        detections = self._detect_plates(image)

        for bbox, conf in detections:
            x1, y1, x2, y2 = bbox

            # Step 2: Crop with padding
            crop = self._crop_plate(image, x1, y1, x2, y2)

            # Step 3: Preprocess for OCR
            processed = self._preprocess(crop)

            # Step 4: OCR
            raw_text = self._ocr_read(processed)

            # Step 5: Clean
            clean_text = self._clean_plate_text(raw_text)

            if clean_text:
                plates.append({
                    "text": clean_text,
                    "confidence": round(float(conf), 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
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
    # Internal steps
    # ------------------------------------------------------------------

    def _detect_plates(self, image: np.ndarray) -> list[tuple]:
        """Run YOLO and return list of (bbox, confidence)."""
        results = self.detector(image, conf=self.detect_conf, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                detections.append(([x1, y1, x2, y2], conf))

        # Sort by confidence descending
        detections.sort(key=lambda d: d[1], reverse=True)
        return detections

    def _crop_plate(
        self, image: np.ndarray, x1: float, y1: float, x2: float, y2: float
    ) -> np.ndarray:
        """Crop the plate region with a small padding."""
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.10)

        # Clamp to image bounds
        x1 = max(0, int(x1) - pad_x)
        y1 = max(0, int(y1) - pad_y)
        x2 = min(w, int(x2) + pad_x)
        y2 = min(h, int(y2) + pad_y)

        return image[y1:y2, x1:x2]

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess the plate crop to improve OCR accuracy.

        Steps:
          1. Convert to grayscale
          2. Resize to a standard height (better for OCR)
          3. Apply CLAHE for contrast enhancement
          4. Light bilateral filter to reduce noise but keep edges
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Resize so plate height is ~80px (consistent for OCR)
        target_h = 80
        scale = target_h / gray.shape[0] if gray.shape[0] > 0 else 1
        if scale != 1:
            new_w = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

        # CLAHE: adaptive contrast (handles uneven lighting at gantry)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Gentle noise reduction
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        return gray

    def _ocr_read(self, processed: np.ndarray) -> str:
        """Run EasyOCR on the preprocessed plate image."""
        results = self.ocr.readtext(
            processed,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            paragraph=True,        # merge nearby text into one line
            detail=0,              # return plain strings only
        )
        return " ".join(results) if results else ""

    def _clean_plate_text(self, raw: str) -> str:
        """
        Post-process OCR output to fix common misreads.

        Customize the char_map for your country's plate format.
        """
        text = raw.upper().strip()

        # Remove anything that's not alphanumeric
        text = re.sub(r"[^A-Z0-9]", "", text)

        # Common OCR substitution errors
        char_map = {
            "O": "0",  # In numeric positions, O→0
            "I": "1",  # I→1
            "S": "5",  # S→5
            "B": "8",  # B→8
            "G": "6",  # G→6
            "Z": "2",  # Z→2
        }
        # NOTE: These replacements are aggressive and depend on plate format.
        # For Singapore plates (SXX #### X), you'd only apply these in
        # the numeric middle section. For now, we return raw alphanumeric.
        # Uncomment below if you want aggressive correction:
        #
        # corrected = []
        # for i, ch in enumerate(text):
        #     if i >= 3 and i <= 6 and ch in char_map:  # numeric section
        #         corrected.append(char_map[ch])
        #     else:
        #         corrected.append(ch)
        # text = "".join(corrected)

        return text


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
