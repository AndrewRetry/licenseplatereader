"""
Unit tests for src/hf_ocr.py

TrOCR model is fully mocked — these run without torch/transformers installed.
They test preprocessing, confidence derivation, variant selection logic.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.hf_ocr import (
    TARGET_WIDTH,
    _bgr_to_pil_rgb,
    _confidence_from_scores,
    _preprocess_crop,
    ocr_crop_hf,
)


def _make_plate_crop(text: str = "SBA1234L", w: int = 300, h: int = 60) -> np.ndarray:
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    cv2.putText(img, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    return img


# ── _preprocess_crop ──────────────────────────────────────────────────────────

class TestPreprocessCrop:
    def test_returns_three_variants(self):
        crop = _make_plate_crop()
        result = _preprocess_crop(crop)
        assert len(result) == 3

    def test_variant_labels(self):
        crop = _make_plate_crop()
        labels = [label for _, label in _preprocess_crop(crop)]
        assert "thresh" in labels
        assert "inv"    in labels
        assert "gray"   in labels

    def test_each_variant_is_numpy(self):
        crop = _make_plate_crop()
        for img, _ in _preprocess_crop(crop):
            assert isinstance(img, np.ndarray)

    def test_upscales_small_crops(self):
        tiny_crop = np.full((20, 50, 3), 200, dtype=np.uint8)
        for img, _ in _preprocess_crop(tiny_crop):
            assert img.shape[1] >= 50   # must not shrink

    def test_gray_variant_is_2d(self):
        crop = _make_plate_crop()
        variants = {label: img for img, label in _preprocess_crop(crop)}
        assert variants["gray"].ndim == 2

    def test_thresh_variant_is_binary(self):
        crop = _make_plate_crop()
        variants = {label: img for img, label in _preprocess_crop(crop)}
        unique = np.unique(variants["thresh"])
        assert set(unique).issubset({0, 255})

    def test_inv_is_inverse_of_thresh(self):
        crop = _make_plate_crop()
        variants = {label: img for img, label in _preprocess_crop(crop)}
        reconstructed = cv2.bitwise_not(variants["thresh"])
        assert np.array_equal(reconstructed, variants["inv"])


# ── _bgr_to_pil_rgb ───────────────────────────────────────────────────────────

class TestBgrToPilRgb:
    def test_bgr_array_converts(self):
        from PIL import Image
        bgr = np.zeros((60, 200, 3), dtype=np.uint8)
        result = _bgr_to_pil_rgb(bgr)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_grayscale_array_converts(self):
        from PIL import Image
        gray = np.zeros((60, 200), dtype=np.uint8)
        result = _bgr_to_pil_rgb(gray)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_size_preserved(self):
        bgr = np.zeros((60, 200, 3), dtype=np.uint8)
        result = _bgr_to_pil_rgb(bgr)
        assert result.size == (200, 60)   # PIL: (width, height)


# ── _confidence_from_scores ───────────────────────────────────────────────────

class TestConfidenceFromScores:
    def test_returns_float_in_range(self):
        """With a mocked model that raises, should return safe fallback."""
        mock_model = MagicMock()
        mock_model.compute_transition_scores.side_effect = RuntimeError("fail")
        result = _confidence_from_scores(mock_model, MagicMock(), MagicMock())
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0

    def test_high_log_prob_gives_high_confidence(self):
        import torch
        mock_model = MagicMock()
        # log_prob close to 0 → very confident
        mock_model.compute_transition_scores.return_value = torch.tensor([[-0.05, -0.03]])
        result = _confidence_from_scores(mock_model, MagicMock(), MagicMock())
        assert result > 60.0

    def test_low_log_prob_gives_low_confidence(self):
        import torch
        mock_model = MagicMock()
        # very negative log_prob → low confidence
        mock_model.compute_transition_scores.return_value = torch.tensor([[-4.0, -5.0]])
        result = _confidence_from_scores(mock_model, MagicMock(), MagicMock())
        assert result < 40.0


# ── ocr_crop_hf ───────────────────────────────────────────────────────────────

class TestOcrCropHf:
    def _make_mock_generate_output(self, text: str, log_prob: float = -0.2):
        import torch

        # TrOCRProcessor mock
        mock_processor = MagicMock()
        mock_processor.return_value = MagicMock(pixel_values=torch.zeros(1, 3, 384, 384))
        mock_processor.batch_decode.return_value = [text]

        # Model generate output
        mock_output = MagicMock()
        mock_output.sequences = torch.zeros((1, 10), dtype=torch.long)
        mock_output.scores = [torch.zeros(1, 50257)] * 8

        mock_model = MagicMock()
        mock_model.generate.return_value = mock_output
        mock_model.compute_transition_scores.return_value = \
            torch.full((1, 8), log_prob)

        return mock_processor, mock_model

    @patch("src.hf_ocr._get_trocr")
    def test_returns_text_and_confidence(self, mock_get):
        proc, mdl = self._make_mock_generate_output("SBA1234L")
        mock_get.return_value = (proc, mdl)
        crop = _make_plate_crop("SBA1234L")
        result = ocr_crop_hf(crop)
        assert result is not None
        text, conf = result
        assert isinstance(text, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 100.0

    @patch("src.hf_ocr._get_trocr")
    def test_empty_text_returns_none(self, mock_get):
        proc, mdl = self._make_mock_generate_output("")  # all variants return ""
        mock_get.return_value = (proc, mdl)
        crop = _make_plate_crop()
        result = ocr_crop_hf(crop)
        assert result is None

    @patch("src.hf_ocr._get_trocr")
    def test_model_load_failure_returns_none(self, mock_get):
        mock_get.side_effect = RuntimeError("no torch")
        crop = _make_plate_crop()
        result = ocr_crop_hf(crop)
        assert result is None

    @patch("src.hf_ocr._get_trocr")
    def test_generate_exception_returns_none(self, mock_get):
        proc = MagicMock()
        proc.return_value = MagicMock(pixel_values=MagicMock())
        proc.batch_decode.return_value = [""]
        mdl = MagicMock()
        mdl.generate.side_effect = RuntimeError("CUDA OOM")
        mock_get.return_value = (proc, mdl)
        crop = _make_plate_crop()
        result = ocr_crop_hf(crop)
        assert result is None

    @patch("src.hf_ocr._get_trocr")
    def test_best_variant_wins(self, mock_get):
        """
        If one preprocessing variant returns higher confidence, that text wins.
        We mock 3 calls: thresh→"WRONG" low conf, inv→"SBA1234L" high conf, gray→"".
        """
        import torch
        mock_proc = MagicMock()
        mock_proc.return_value = MagicMock(pixel_values=torch.zeros(1, 3, 384, 384))
        # batch_decode returns different text each call
        mock_proc.batch_decode.side_effect = [["WRONG"], ["SBA1234L"], [""]]

        mock_output = MagicMock()
        mock_output.sequences = torch.zeros((1, 10), dtype=torch.long)
        mock_output.scores = [torch.zeros(1, 100)] * 5
        mock_mdl = MagicMock()
        mock_mdl.generate.return_value = mock_output
        # First call low conf, second high conf, third irrelevant
        mock_mdl.compute_transition_scores.side_effect = [
            torch.full((1, 5), -3.0),    # WRONG → low conf
            torch.full((1, 5), -0.1),    # SBA1234L → high conf
            RuntimeError("empty"),       # gray → handled by fallback
        ]
        mock_get.return_value = (mock_proc, mock_mdl)

        crop = _make_plate_crop()
        result = ocr_crop_hf(crop)
        # Should pick the high-confidence result
        assert result is not None
        text, conf = result
        assert text == "SBA1234L"
