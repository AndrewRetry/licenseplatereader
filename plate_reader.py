"""
plate_reader.py — License plate detection and OCR engine, tuned for Singapore plates.

Singapore plate format:  SBA 1234 A
  - Prefix:    1–3 letters  (I and O never used; no vowels in 2nd char of 3-letter prefix)
  - Numbers:   1–4 digits
  - Checksum:  1 letter     (F, I, N, O, Q, V, W never used as check digits)

Two LTA-approved colour schemes:
  - White-on-black  (white text, black background) — most common
  - Black-on-white  (front) / black-on-yellow (rear) — Euro scheme

Pipeline:
  Image → YOLOv8n (detect plate bbox) → Crop → Normalise colour →
  Preprocess → EasyOCR → SG-aware clean + validate → "SBA1234A"

Usage:
  reader = PlateReader("plate_model.pt")
  results = reader.read(image_bgr)
  # [{"text": "SBA1234A", "confidence": 0.94, "bbox": [x1,y1,x2,y2]}]
"""

import re
import logging
from pathlib import Path

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singapore plate constants
# ---------------------------------------------------------------------------

# Matches: 1–3 prefix letters, 1–4 digits, 1 checksum letter
# I and O are excluded from all positions; checksum never uses F,I,N,O,Q,V,W
_SG_PLATE_RE = re.compile(r"^[A-HJ-NPR-Z]{1,3}[0-9]{1,4}[A-EG-HJ-MPRSTU-Z]$")

# Minimum / maximum character count for a plausible SG plate
_SG_MIN_LEN = 4   # e.g. S12A (edge case)
_SG_MAX_LEN = 8   # e.g. SBA1234A (standard)

# OCR confuses these in numeric regions: letter → digit
_ALPHA_TO_DIGIT: dict[str, str] = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
    "G": "6",
    "Z": "2",
}

# OCR confuses these in letter regions: digit → letter
_DIGIT_TO_ALPHA: dict[str, str] = {v: k for k, v in _ALPHA_TO_DIGIT.items()}


class PlateReader:
    """Lightweight license plate detector + OCR reader, tuned for Singapore."""

    def __init__(
        self,
        model_path: str = "plate_model.pt",
        detect_conf: float = 0.5,
        ocr_languages: list[str] | None = None,
        gpu: bool = False,
    ):
        """
        Args:
            model_path:    Path to YOLOv8 .pt weights trained on license plates.
            detect_conf:   Minimum YOLO detection confidence (0.0–1.0).
            ocr_languages: EasyOCR language codes. Defaults to ["en"] — correct for SG plates.
            gpu:           Use GPU for inference. False = CPU-only (adequate for gantry use).
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "See README.md Phase 2 to train or download one."
            )
        self.detector = YOLO(model_path)
        self.detect_conf = detect_conf
        logger.info("YOLO model loaded from %s", model_path)

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
            List of dicts with keys:
              - text:       cleaned Singapore plate string, e.g. "SBA1234A"
              - confidence: YOLO detection confidence (0–1)
              - bbox:       [x1, y1, x2, y2] pixel coordinates
        """
        plates = []
        detections = self._detect_plates(image)

        for bbox, conf in detections:
            x1, y1, x2, y2 = bbox

            crop = self._crop_plate(image, x1, y1, x2, y2)
            normalised = self._normalise_colour_scheme(crop)
            processed = self._preprocess(normalised)
            raw_text = self._ocr_read(processed)
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
        Detect and normalise Singapore plate colour scheme.

        LTA permits two schemes:
          - White-on-black (white text, black bg) — must invert for OCR
          - Black-on-white / black-on-yellow (black text, light bg) — use as-is

        We detect scheme by measuring the mean brightness of the crop.
        A dark mean indicates white-on-black; we invert so text is always dark.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        if mean_brightness < 100:
            # White-on-black plate: invert so text becomes dark on light background
            logger.debug("White-on-black plate detected (mean=%.1f) — inverting", mean_brightness)
            return cv2.bitwise_not(crop)

        # Black-on-white or black-on-yellow: use as-is
        return crop

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess the (already colour-normalised) plate crop for OCR.

        Steps:
          1. Grayscale
          2. Resize to standard 80 px height (consistent character size for EasyOCR)
          3. CLAHE for adaptive contrast (handles uneven gantry lighting)
          4. Bilateral filter to reduce noise while preserving character edges
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        target_h = 80
        if gray.shape[0] > 0 and gray.shape[0] != target_h:
            scale = target_h / gray.shape[0]
            new_w = max(1, int(gray.shape[1] * scale))
            gray = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        return gray

    def _ocr_read(self, processed: np.ndarray) -> str:
        """Run EasyOCR on the preprocessed plate image."""
        results = self.ocr.readtext(
            processed,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            paragraph=True,   # merge nearby text into one line
            detail=0,         # plain strings only
        )
        return " ".join(results) if results else ""

    def _clean_plate_text(self, raw: str) -> str:
        """
        Post-process OCR output for Singapore plates.

        Steps:
          1. Uppercase, strip spaces, keep only alphanumeric
          2. Apply position-aware OCR error corrections:
             - Prefix (letter) region:  digits that look like letters are fixed
             - Number region:           letters that look like digits are fixed
             - Checksum (last char):    ensure it is a letter
          3. Validate against Singapore plate format
        """
        text = raw.upper().strip()
        text = re.sub(r"[^A-Z0-9]", "", text)

        if not (_SG_MIN_LEN <= len(text) <= _SG_MAX_LEN):
            return ""

        text = self._apply_sg_corrections(text)

        if not self._is_valid_sg_plate(text):
            return ""

        return text

    def _apply_sg_corrections(self, text: str) -> str:
        """
        Apply position-aware corrections based on Singapore plate structure:
          [PREFIX letters] [NUMBER digits] [CHECKSUM letter]

        We locate where the digit block starts and apply corrections
        to each region accordingly.
        """
        chars = list(text)
        n = len(chars)

        # --- Ensure last character is a letter (checksum) ---
        if chars[-1].isdigit():
            chars[-1] = _DIGIT_TO_ALPHA.get(chars[-1], chars[-1])

        # --- Find where the numeric block starts ---
        # Scan from left; the first character that looks like a digit
        # (or is a letter commonly confused with a digit) marks the boundary.
        digit_start = n - 1  # fallback: no digit block found
        for i in range(n - 1):  # exclude the last checksum char
            ch = chars[i]
            if ch.isdigit() or ch in _ALPHA_TO_DIGIT:
                digit_start = i
                break

        # --- Correct prefix region (should be letters) ---
        for i in range(digit_start):
            if chars[i].isdigit():
                chars[i] = _DIGIT_TO_ALPHA.get(chars[i], chars[i])

        # --- Correct numeric region (should be digits) ---
        for i in range(digit_start, n - 1):
            if chars[i].isalpha():
                chars[i] = _ALPHA_TO_DIGIT.get(chars[i], chars[i])

        return "".join(chars)

    def _is_valid_sg_plate(self, text: str) -> bool:
        """
        Return True if text matches the Singapore plate format.

        Format: [1–3 prefix letters][1–4 digits][1 checksum letter]
        Exclusions:
          - I and O never appear (too similar to 1 and 0)
          - Checksum never uses F, I, N, O, Q, V, W
        """
        return bool(_SG_PLATE_RE.match(text))


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