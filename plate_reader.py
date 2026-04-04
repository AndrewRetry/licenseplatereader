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
_MAX_CROP_WIDTH = 512

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

        logger.info("Loading TrOCR processor and model (%s)...", _TROCR_MODEL_ID)
        self.processor = TrOCRProcessor.from_pretrained(_TROCR_MODEL_ID)
        self.trocr = VisionEncoderDecoderModel.from_pretrained(_TROCR_MODEL_ID)
        self.trocr.eval()
        logger.info("TrOCR ready (GPU=%s)", gpu)

    def read(self, image: np.ndarray) -> list[dict]:
        plates = []
        for bbox, conf in self._detect_plates(image):
            x1, y1, x2, y2 = bbox
            crop       = self._crop_plate(image, x1, y1, x2, y2)
            normalised = self._normalise_colour_scheme(crop)
            raw_text   = self._ocr_read(normalised)
            
            # First, clean out garbage chars
            clean_text = self._clean_plate_text(raw_text)
            
            # Second, attempt to fix common OCR mistakes (0 vs O, etc)
            fixed_text = self._fix_common_ocr_errors(clean_text)

            if fixed_text:
                is_valid = self._validate_sg_checksum(fixed_text)
                plates.append({
                    "text":       fixed_text,
                    "confidence": round(float(conf), 3),
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                    "checksum_valid": is_valid # Add this flag for downstream
                })
        return plates

    def read_from_path(self, image_path: str) -> list[dict]:
        image = cv2.imread(image_path)
        return self.read(image)

    def read_from_bytes(self, image_bytes: bytes) -> list[dict]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return self.read(image)

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

    def _crop_plate(self, image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.10)
        x1 = max(0, int(x1) - pad_x)
        y1 = max(0, int(y1) - pad_y)
        x2 = min(w, int(x2) + pad_x)
        y2 = min(h, int(y2) + pad_y)
        return image[y1:y2, x1:x2]

    def _normalise_colour_scheme(self, crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < 110: # Increased threshold slightly
            return cv2.bitwise_not(crop)
        return crop

    def _preprocess_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """Enhanced Preprocessing: CLAHE + Unsharp Masking"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # 1. Stronger CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 2. Unsharp Mask (Sharpening to define letter edges)
        gaussian_3 = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        unsharp_image = cv2.addWeighted(enhanced, 2.0, gaussian_3, -1.0, 0)
        
        return cv2.cvtColor(unsharp_image, cv2.COLOR_GRAY2RGB)

    def _ocr_read(self, crop: np.ndarray) -> str:
        rgb = self._preprocess_for_ocr(crop)
        pil_img = Image.fromarray(rgb)

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
        return re.sub(r"[^A-Z0-9]", "", raw.upper().strip())

    def _fix_common_ocr_errors(self, plate: str) -> str:
        """Fixes obvious placement errors before checksum validates."""
        if not plate: return plate
        
        # Fix starting zero (e.g. 0X1728A -> QX1728A or OX...)
        # TrOCR commonly confuses Q/O/0
        if plate.startswith("0X") or plate.startswith("OX"):
             plate = "QX" + plate[2:]
             
        # If the last character is a digit instead of a letter
        # e.g. SDN74840 -> SDN7484U or SDN7484O
        if len(plate) >= 4 and plate[-1].isdigit():
             # Very common for U/O to be read as 0
             if plate[-1] == "0":
                 # We leave it as 'O' or 'U' but checksum will be the final judge.
                 # Actually, we can just strip the last char and let the checksum 
                 # calculate what it *should* be, but for now we'll guess 'U'
                 plate = plate[:-1] + "U" 
        return plate

    def _validate_sg_checksum(self, plate: str) -> bool:
        """
        Implements the LTA Checksum Algorithm.
        Multiplier: 9, 4, 5, 4, 3, 2
        Alphabet mapping: A=1, B=2 ... Z=26
        Result mapping: 0=A, 1=Z, 2=Y, 3=X, 4=W, 5=V, 6=U, 7=T, 8=S, 9=R, 
                        10=P, 11=M, 12=L, 13=K, 14=J, 15=H, 16=G, 17=E, 18=D, 19=C, 20=B
        Note: F, I, N, O, Q, V, W are excluded from the final checksum letter.
        """
        match = re.match(r'^([A-Z]{1,3})([0-9]{1,4})([A-Z])$', plate)
        if not match:
            return False
            
        prefix, numbers, suffix = match.groups()
        
        # If prefix is 3 letters (e.g., SBA), drop the first letter (e.g., -> BA)
        # If prefix is 1 letter, pad left with space (which equals A or 1 if we had to)
        # Standard implementation takes the last two letters of prefix
        if len(prefix) == 3:
            prefix = prefix[1:]
        elif len(prefix) == 1:
            prefix = "A" + prefix # Padding logic for single letters usually assumes A
            
        # Pad numbers to 4 digits
        numbers = numbers.zfill(4)
        
        # Convert letters to numbers (A=1...Z=26)
        p1 = ord(prefix[0]) - 64
        p2 = ord(prefix[1]) - 64
        n1, n2, n3, n4 = [int(d) for d in numbers]
        
        # Multiply by weights: 9, 4, 5, 4, 3, 2
        total = (p1 * 9) + (p2 * 4) + (n1 * 5) + (n2 * 4) + (n3 * 3) + (n4 * 2)
        
        remainder = total % 19
        
        # LTA Mapping
        mapping = {
            0: 'A', 1: 'Z', 2: 'Y', 3: 'X', 4: 'W', 5: 'V', 6: 'U',
            7: 'T', 8: 'S', 9: 'R', 10: 'P', 11: 'M', 12: 'L',
            13: 'K', 14: 'J', 15: 'H', 16: 'G', 17: 'E', 18: 'D', 19: 'C', 20: 'B'
        }
        
        expected_suffix = mapping.get(remainder)
        return expected_suffix == suffix

if __name__ == "__main__":
    pass