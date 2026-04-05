import re
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_TROCR_MODEL_ID = "microsoft/trocr-base-printed"
_MIN_CROP_WIDTH  = 128   # upscale tiny crops
_MAX_CROP_WIDTH  = 512   # downscale huge crops

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
        logger.info("YOLO model loaded from %s", model_path)

        logger.info("Loading TrOCR (%s)…", _TROCR_MODEL_ID)
        self.processor = TrOCRProcessor.from_pretrained(_TROCR_MODEL_ID)
        self.trocr = VisionEncoderDecoderModel.from_pretrained(_TROCR_MODEL_ID)
        self.trocr.eval()
        logger.info("TrOCR ready (GPU=%s)", gpu)

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

            if not clean_text:
                continue

            is_valid = self._validate_sg_checksum(clean_text)
            plates.append({
                "text":           clean_text,
                "confidence":     round(float(conf), 3),
                "bbox":           [int(x1), int(y1), int(x2), int(y2)],
                "checksum_valid": is_valid,
            })
            logger.info(
                "Read: %s  det_conf=%.2f  checksum=%s",
                clean_text, conf, "OK" if is_valid else "FAIL",
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
        Hard binarisation or unsharp masking degrades stroke integrity
        and hurts the encoder. CLAHE is the correct and sufficient step.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    def _ocr_read(self, crop: np.ndarray) -> str:
        processed = self._preprocess_for_ocr(crop)
        pil_img = Image.fromarray(processed)

        # Upscale tiny crops — TrOCR degrades on very small inputs
        if pil_img.width < _MIN_CROP_WIDTH:
            scale = _MIN_CROP_WIDTH / pil_img.width
            pil_img = pil_img.resize(
                (_MIN_CROP_WIDTH, max(1, int(pil_img.height * scale))),
                Image.LANCZOS,
            )

        # Downscale excessively large crops
        if pil_img.width > _MAX_CROP_WIDTH:
            scale = _MAX_CROP_WIDTH / pil_img.width
            pil_img = pil_img.resize(
                (_MAX_CROP_WIDTH, max(1, int(pil_img.height * scale))),
                Image.LANCZOS,
            )

        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values

        # num_beams=4: beam search instead of greedy — meaningful accuracy
        # gain for short sequences like plate numbers, negligible latency cost
        generated_ids = self.trocr.generate(pixel_values, num_beams=4)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    def _clean_plate_text(self, raw: str) -> str:
        """Strip everything that isn't an uppercase letter or digit."""
        return re.sub(r"[^A-Z0-9]", "", raw.upper().strip())

    # ------------------------------------------------------------------
    # SG checksum
    # ------------------------------------------------------------------

    def _validate_sg_checksum(self, plate: str) -> bool:
        """
        LTA checksum algorithm.
        Weights:  9  4  5  4  3  2  (last-2-prefix + 4 digits)
        Excluded check digits: F I N O Q V W
        Remainder → check letter (mod 19, 19 possible values):
          0=A 1=Z 2=Y 3=X 4=U 5=T 6=S 7=R 8=P 9=M
          10=L 11=K 12=J 13=H 14=G 15=E 16=D 17=C 18=B
        """
        match = re.match(r'^([A-Z]{1,3})([0-9]{1,4})([A-Z])$', plate)
        if not match:
            logger.debug("Checksum skip — format mismatch: %s", plate)
            return False

        prefix, numbers, suffix = match.groups()

        # Normalise prefix to exactly 2 characters for weight calculation
        if len(prefix) == 3:
            p1 = ord(prefix[1]) - 64   # middle letter
            p2 = ord(prefix[2]) - 64   # last letter
        elif len(prefix) == 2:
            p1 = ord(prefix[0]) - 64
            p2 = ord(prefix[1]) - 64
        else:
            # Single-letter prefix (e.g. old "S" plates): first slot is zero-weighted
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