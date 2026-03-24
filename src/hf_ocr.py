"""
HuggingFace TrOCR OCR engine.

Model: microsoft/trocr-base-printed
  - Transformer encoder-decoder, fine-tuned on printed text
  - MIT licence, free, CPU-capable
  - Auto-downloads from HuggingFace Hub on first use (~400 MB)

Returns (text, confidence_0_to_100) per crop.
Confidence is derived from the mean per-token probability of the
greedy-decoded sequence — not a calibrated score, but a useful signal
for ranking candidates.
"""

import math
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .logger import get_logger

logger = get_logger(__name__)

_TROCR_MODEL_ID = "microsoft/trocr-base-printed"
TARGET_WIDTH = 384    # TrOCR's ViT encoder was trained at 384×384


@lru_cache(maxsize=1)
def _get_trocr():
    """
    Lazy singleton. Deferred import keeps the module loadable without torch.
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: PLC0415

    logger.info("Loading TrOCR from HuggingFace Hub", model=_TROCR_MODEL_ID)
    processor = TrOCRProcessor.from_pretrained(_TROCR_MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(_TROCR_MODEL_ID)
    model.eval()
    logger.info("TrOCR ready")
    return processor, model


def _preprocess_crop(crop_bgr: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """
    Return 3 preprocessing variants of a plate crop.
    TrOCR works best with clear, high-contrast images.
    """
    h, w = crop_bgr.shape[:2]
    scale = max(1, TARGET_WIDTH // max(w, 1))
    resized = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # CLAHE + Otsu threshold
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray)
    _, thresh = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inverted (dark-on-light → light-on-dark)
    inv = cv2.bitwise_not(thresh)

    # Raw gray (TrOCR handles contrast internally)
    return [
        (thresh, "thresh"),
        (inv,    "inv"),
        (gray,   "gray"),
    ]


def _bgr_to_pil_rgb(img: np.ndarray) -> Image.Image:
    """Convert a grayscale or BGR numpy array to a 3-channel PIL RGB image."""
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _confidence_from_scores(model, sequences, scores) -> float:
    """
    Derive a 0–100 confidence score from TrOCR generation log-probs.

    Uses model.compute_transition_scores() which returns per-token log-probs.
    Mean over non-padding tokens → sigmoid-scaled to 0–100.
    """
    import torch  # noqa: PLC0415

    try:
        transition_scores = model.compute_transition_scores(
            sequences, scores, normalize_logits=True
        )
        log_probs = transition_scores[0]
        # Filter out padding/special tokens (they have score 0.0)
        valid = log_probs[log_probs < 0]
        if len(valid) == 0:
            return 50.0
        mean_log_prob = valid.mean().item()           # negative value, e.g. -0.3
        # Map: -0.0 → ~100, -1.0 → ~50, -3.0 → ~5
        confidence = 100.0 / (1.0 + math.exp(-5.0 * (mean_log_prob + 0.5)))
        return round(min(100.0, max(0.0, confidence)), 1)
    except Exception:
        return 60.0   # safe fallback if score computation fails


def ocr_crop_hf(crop_bgr: np.ndarray) -> Optional[tuple[str, float]]:
    """
    Run TrOCR on a plate crop.

    Args:
        crop_bgr: BGR numpy array of the plate region

    Returns:
        (text, confidence_0_to_100) or None if OCR produced nothing.
    """
    import torch  # noqa: PLC0415

    try:
        processor, model = _get_trocr()
    except Exception as exc:
        logger.warning("TrOCR load failed", error=str(exc))
        return None

    variants = _preprocess_crop(crop_bgr)

    best_text: Optional[str] = None
    best_conf: float = 0.0

    for img_variant, label in variants:
        pil_img = _bgr_to_pil_rgb(img_variant)
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values

        try:
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    output_scores=True,
                    return_dict_in_generate=True,
                    num_beams=1,        # greedy — fast, good enough for plates
                    max_new_tokens=20,  # SG plates are at most ~9 chars
                )
        except Exception as exc:
            logger.warning("TrOCR generate failed", variant=label, error=str(exc))
            continue

        text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
        if not text:
            continue

        conf = _confidence_from_scores(model, outputs.sequences, outputs.scores)
        logger.debug("TrOCR result", variant=label, text=text, conf=conf)

        if conf > best_conf:
            best_conf = conf
            best_text = text

    if not best_text:
        return None

    return best_text, best_conf
