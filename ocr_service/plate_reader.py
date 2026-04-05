import re
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_TROCR_MODEL_ID = "microsoft/trocr-base-printed"
_MIN_CROP_WIDTH  = 128
_MAX_CROP_WIDTH  = 512

# Characters OCR commonly confuses, by position type
# In letter positions: digits that look like letters
_DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '6': 'G'}
# In digit positions: letters that look like digits
_LETTER_TO_DIGIT = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2', 'G': '6'}
# In the suffix (last char) specifically: 0 must be U because O is not a valid SG check digit
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

        self.device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
        logger.info("YOLO model loaded from %s", model_path)
        logger.info("Loading TrOCR (%s) on device=%s …", _TROCR_MODEL_ID, self.device)

        self.processor = TrOCRProcessor.from_pretrained(_TROCR_MODEL_ID)
        self.trocr = VisionEncoderDecoderModel.from_pretrained(_TROCR_MODEL_ID)
        self.trocr = self.trocr.to(self.device)
        self.trocr.eval()

        self._warmup()
        logger.info("TrOCR ready (device=%s)", self.device)

    def _warmup(self) -> None:
        """
        Run one dummy inference so the first real request doesn't pay
        JIT compilation cost. Uses a blank image — output is discarded.
        """
        dummy = Image.fromarray(np.ones((64, 256, 3), dtype=np.uint8) * 200)
        pixel_values = self.processor(images=dummy, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        with torch.no_grad():
            self.trocr.generate(pixel_values, num_beams=1)
        logger.info("TrOCR warmup complete.")

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
        """Invert white-on-black plates to dark-on-light for TrOCR."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < 110:
            return cv2.bitwise_not(crop)
        return crop

    def _preprocess_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """
        CLAHE contrast enhancement only.
        TrOCR is a ViT — it handles its own normalisation internally.
        Hard binarisation or morphological ops degrade stroke integrity.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    def _ocr_read(self, crop: np.ndarray) -> str:
        processed = self._preprocess_for_ocr(crop)
        pil_img = Image.fromarray(processed)

        if pil_img.width < _MIN_CROP_WIDTH:
            scale = _MIN_CROP_WIDTH / pil_img.width
            pil_img = pil_img.resize(
                (_MIN_CROP_WIDTH, max(1, int(pil_img.height * scale))),
                Image.LANCZOS,
            )

        if pil_img.width > _MAX_CROP_WIDTH:
            scale = _MAX_CROP_WIDTH / pil_img.width
            pil_img = pil_img.resize(
                (_MAX_CROP_WIDTH, max(1, int(pil_img.height * scale))),
                Image.LANCZOS,
            )

        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.trocr.generate(pixel_values, num_beams=4)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # ------------------------------------------------------------------
    # Text cleaning and recovery
    # ------------------------------------------------------------------

    def _clean_plate_text(self, raw: str) -> str:
        """Strip everything that isn't an uppercase letter or digit."""
        return re.sub(r"[^A-Z0-9]", "", raw.upper().strip())

    def _recover_sg_plate(self, text: str) -> str:
        """
        Apply position-aware OCR confusion correction for SG plate format.

        SG format: [1-3 letters][1-4 digits][1 letter]

        TrOCR commonly confuses visually similar chars across types, e.g.
        U→0 in the suffix position. We fix these using the known structure —
        letters must appear in prefix/suffix, digits in the middle — without
        guessing at specific character identities beyond known confusion pairs.

        Returns the corrected string, or the original if format can't be parsed.
        """
        # Must be long enough to be a plate (min: 1 letter + 1 digit + 1 letter = 3)
        if len(text) < 3:
            return text

        # --- Identify prefix (leading letters) ---
        i = 0
        while i < len(text) and (text[i].isalpha() or text[i] in _DIGIT_TO_LETTER):
            i += 1
            if i >= 4:  # prefix is max 3 letters
                break

        prefix_raw = text[:i]

        # --- Identify digit block ---
        j = i
        while j < len(text) and (text[j].isdigit() or text[j] in _LETTER_TO_DIGIT):
            j += 1
            if j - i >= 4:  # max 4 digits
                break

        digits_raw = text[i:j]

        # Suffix is the remaining character(s); we expect exactly 1
        suffix_raw = text[j:]

        # If we can't parse a prefix + digits + suffix, return as-is
        if not prefix_raw or not digits_raw or not suffix_raw:
            return text

        # --- Fix each region ---
        # Prefix: every char should be a letter
        prefix = "".join(
            _DIGIT_TO_LETTER.get(c, c) for c in prefix_raw
        )

        # Digits: every char should be a digit
        digits = "".join(
            _LETTER_TO_DIGIT.get(c, c) for c in digits_raw
        )

        # Suffix: should be exactly one letter
        # Use _SUFFIX_FIXES so that 0→U (not 0→O, since O is not a valid SG check digit)
        suffix_char = suffix_raw[0]
        suffix = _SUFFIX_FIXES.get(suffix_char, suffix_char)

        return prefix + digits + suffix

    # ------------------------------------------------------------------
    # SG checksum
    # ------------------------------------------------------------------

    def _validate_sg_checksum(self, plate: str) -> bool:
        """
        LTA checksum algorithm.
        Weights:  9  4  5  4  3  2  (last-2-prefix + 4 digits)
        Excluded check digits: F I N O Q V W
        Remainder → check letter (mod 19):
          0=A 1=Z 2=Y 3=X 4=U 5=T 6=S 7=R 8=P 9=M
          10=L 11=K 12=J 13=H 14=G 15=E 16=D 17=C 18=B
        """
        match = re.match(r'^([A-Z]{1,3})([0-9]{1,4})([A-Z])$', plate)
        if not match:
            logger.debug("Checksum skip — format mismatch: %s", plate)
            return False

        prefix, numbers, suffix = match.groups()

        if len(prefix) == 3:
            p1 = ord(prefix[1]) - 64
            p2 = ord(prefix[2]) - 64
        elif len(prefix) == 2:
            p1 = ord(prefix[0]) - 64
            p2 = ord(prefix[1]) - 64
        else:
            # Single-letter prefix: first slot is zero-weighted
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
        is_valid = expected == suffix

        if not is_valid:
            logger.debug(
                "Checksum fail: plate=%s  remainder=%d  expected=%s  got=%s",
                plate, remainder, expected, suffix,
            )
        return is_valid