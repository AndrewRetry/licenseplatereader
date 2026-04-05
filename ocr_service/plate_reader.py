import re
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

logger = logging.getLogger(__name__)

_OCR_MODEL = "cct-xs-v1-global-model"
_MIN_CROP_WIDTH = 128
_MAX_CROP_WIDTH = 512

_DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '6': 'G'}
_LETTER_TO_DIGIT = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2', 'G': '6'}
_SUFFIX_FIXES    = {'0': 'U', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '6': 'G'}


class PlateReader:
    def __init__(
        self,
        model_path: str = "plate_model.pt",
        detect_conf: float = 0.25,
        gpu: bool = False,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at '{model_path}'.")

        self.detector = YOLO(model_path)
        self.detect_conf = detect_conf

        # fast-plate-ocr uses ONNX runtime — no torch needed
        logger.info("Loading fast-plate-ocr (%s) …", _OCR_MODEL)
        self.ocr = LicensePlateRecognizer(_OCR_MODEL)
        logger.info("fast-plate-ocr ready.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read(self, image: np.ndarray) -> list[dict]:
        plates = []
        for bbox, conf in self._detect_plates(image):
            x1, y1, x2, y2 = bbox
            crop       = self._crop_plate(image, x1, y1, x2, y2)
            normalised = self._normalise_colour_scheme(crop)
            raw_text   = self._ocr_read(normalised)
            clean_text = self._clean_plate_text(raw_text)
            fixed_text = self._recover_sg_plate(clean_text)

            if not fixed_text:
                continue

            is_valid = self._validate_sg_checksum(fixed_text)
            plates.append({
                "text":           fixed_text,
                "confidence":     round(float(conf), 3),
                "bbox":           [int(x1), int(y1), int(x2), int(y2)],
                "checksum_valid": is_valid,
            })
            logger.info(
                "Read: %s (raw: %s)  det_conf=%.2f  checksum=%s",
                fixed_text, clean_text, conf, "OK" if is_valid else "FAIL",
            )
        return plates

    def read_from_bytes(self, image_bytes: bytes) -> list[dict]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image bytes")
        return self.read(image)

    def read_from_path(self, image_path: str) -> list[dict]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        return self.read(image)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_plates(self, image: np.ndarray) -> list[tuple]:
        results = self.detector(image, conf=self.detect_conf, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                detections.append(([x1, y1, x2, y2], conf))
        detections.sort(key=lambda d: d[1], reverse=True)
        return detections

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def _crop_plate(
        self,
        image: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
    ) -> np.ndarray:
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.10)
        x1 = max(0, int(x1) - pad_x)
        y1 = max(0, int(y1) - pad_y)
        x2 = min(w, int(x2) + pad_x)
        y2 = min(h, int(y2) + pad_y)
        return image[y1:y2, x1:x2]

    def _normalise_colour_scheme(self, crop: np.ndarray) -> np.ndarray:
        """Invert white-on-black plates to dark-on-light."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < 110:
            return cv2.bitwise_not(crop)
        return crop

    def _ocr_read(self, crop: np.ndarray) -> str:
        h, w = crop.shape[:2]
        if w < _MIN_CROP_WIDTH:
            scale = _MIN_CROP_WIDTH / w
            crop = cv2.resize(crop, (_MIN_CROP_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
        elif w > _MAX_CROP_WIDTH:
            scale = _MAX_CROP_WIDTH / w
            crop = cv2.resize(crop, (_MAX_CROP_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

        results = self.ocr.run(crop)
        if not results:
            return ""
        return results[0].plate

    # ------------------------------------------------------------------
    # Text cleaning and recovery — unchanged
    # ------------------------------------------------------------------

    def _clean_plate_text(self, raw: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", raw.upper().strip())

    def _recover_sg_plate(self, text: str) -> str:
        if len(text) < 3:
            return text

        i = 0
        while i < len(text) and (text[i].isalpha() or text[i] in _DIGIT_TO_LETTER):
            i += 1
            if i >= 4:
                break

        prefix_raw = text[:i]

        j = i
        while j < len(text) and (text[j].isdigit() or text[j] in _LETTER_TO_DIGIT):
            j += 1
            if j - i >= 4:
                break

        digits_raw = text[i:j]
        suffix_raw = text[j:]

        if not prefix_raw or not digits_raw or not suffix_raw:
            return text

        prefix = "".join(_DIGIT_TO_LETTER.get(c, c) for c in prefix_raw)
        digits = "".join(_LETTER_TO_DIGIT.get(c, c) for c in digits_raw)
        suffix = _SUFFIX_FIXES.get(suffix_raw[0], suffix_raw[0])

        return prefix + digits + suffix

    # ------------------------------------------------------------------
    # SG checksum — unchanged
    # ------------------------------------------------------------------

    def _validate_sg_checksum(self, plate: str) -> bool:
        match = re.match(r'^([A-Z]{1,3})([0-9]{1,4})([A-Z])$', plate)
        if not match:
            return False

        prefix, numbers, suffix = match.groups()

        if len(prefix) == 3:
            p1 = ord(prefix[1]) - 64
            p2 = ord(prefix[2]) - 64
        elif len(prefix) == 2:
            p1 = ord(prefix[0]) - 64
            p2 = ord(prefix[1]) - 64
        else:
            p1 = 0
            p2 = ord(prefix[0]) - 64

        n1, n2, n3, n4 = (int(d) for d in numbers.zfill(4))
        total = (p1 * 9) + (p2 * 4) + (n1 * 5) + (n2 * 4) + (n3 * 3) + (n4 * 2)
        remainder = total % 19

        mapping = {
             0: 'A',  1: 'Z',  2: 'Y',  3: 'X',  4: 'U',
             5: 'T',  6: 'S',  7: 'R',  8: 'P',  9: 'M',
            10: 'L', 11: 'K', 12: 'J', 13: 'H', 14: 'G',
            15: 'E', 16: 'D', 17: 'C', 18: 'B',
        }

        expected = mapping[remainder]
        if expected != suffix:
            logger.debug("Checksum fail: plate=%s expected=%s got=%s", plate, expected, suffix)
            return False
        return True