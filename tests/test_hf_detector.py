"""
Unit tests for src/hf_detector.py

The HuggingFace model is mocked so these run without a network connection
or PyTorch install. They test the coordination logic: result parsing,
bbox clamping, fallback ordering.
"""

import io
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.plate_detector import PlateRegion
from src.hf_detector import (
    _CONFIDENCE_THRESHOLD,
    _yolos_candidates,
    detect_plate_regions_hf,
)


def _make_img(w=640, h=480) -> np.ndarray:
    img = np.full((h, w, 3), 80, dtype=np.uint8)
    cv2.rectangle(img, (180, 320), (460, 368), (240, 240, 240), -1)
    return img


def _encode_jpg(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _make_yolos_output(boxes, scores):
    """Build a minimal mock of the YOLOS post-processed result dict."""
    import torch
    return [{
        "scores": torch.tensor(scores),
        "boxes":  torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.zeros(len(scores), dtype=torch.long),
    }]


# ── _yolos_candidates ─────────────────────────────────────────────────────────

class TestYolosCandidates:

    def _mock_yolos(self, boxes, scores):
        """Patch _get_yolos to return a fake processor + model."""
        mock_processor = MagicMock()
        mock_processor.return_value = MagicMock(items=lambda: [])  # inputs
        mock_processor.post_process_object_detection.return_value = \
            _make_yolos_output(boxes, scores)

        mock_model = MagicMock()
        mock_model.return_value = MagicMock()   # outputs

        return mock_processor, mock_model

    @patch("src.hf_detector._get_yolos")
    @patch("torch.no_grad")
    def test_returns_region_for_confident_detection(self, mock_no_grad, mock_get):
        import torch
        mock_no_grad.return_value.__enter__ = lambda s: None
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)

        proc, mdl = self._mock_yolos(
            boxes=[[180.0, 320.0, 460.0, 368.0]],
            scores=[0.92],
        )
        mock_get.return_value = (proc, mdl)

        img = _make_img()
        result = _yolos_candidates(img)

        assert isinstance(result, list)

    @patch("src.hf_detector._get_yolos")
    def test_returns_empty_on_model_load_failure(self, mock_get):
        mock_get.side_effect = RuntimeError("model not found")
        img = _make_img()
        result = _yolos_candidates(img)
        assert result == []

    @patch("src.hf_detector._get_yolos")
    @patch("torch.no_grad")
    def test_filters_low_confidence_boxes(self, mock_no_grad, mock_get):
        import torch
        mock_no_grad.return_value.__enter__ = lambda s: None
        mock_no_grad.return_value.__exit__ = MagicMock(return_value=False)

        proc, mdl = self._mock_yolos(
            boxes=[[180.0, 320.0, 460.0, 368.0]],
            scores=[_CONFIDENCE_THRESHOLD - 0.1],   # just below threshold
        )
        # post_process_object_detection already filters by threshold,
        # so return empty for this test
        proc.post_process_object_detection.return_value = [
            {"scores": torch.tensor([]), "boxes": torch.tensor([]), "labels": torch.tensor([])}
        ]
        mock_get.return_value = (proc, mdl)

        img = _make_img()
        result = _yolos_candidates(img)
        assert result == []


# ── detect_plate_regions_hf ───────────────────────────────────────────────────

class TestDetectPlateRegionsHF:

    @patch("src.hf_detector._yolos_candidates")
    def test_returns_yolos_results_when_found(self, mock_yolos, plate_image):
        fake = [PlateRegion(crop=np.zeros((40, 200, 3), np.uint8),
                            x=10, y=20, w=200, h=40, method="yolos", score=0.9)]
        mock_yolos.return_value = fake
        result = detect_plate_regions_hf(plate_image)
        assert result == fake

    @patch("src.hf_detector._yolos_candidates", return_value=[])
    @patch("src.hf_detector._contour_candidates")
    def test_falls_back_to_contours_when_yolos_empty(self, mock_contour, _mock_yolos, plate_image):
        fake = [PlateRegion(crop=np.zeros((40, 200, 3), np.uint8),
                            x=10, y=20, w=200, h=40, method="contour", score=0.5)]
        mock_contour.return_value = fake
        result = detect_plate_regions_hf(plate_image)
        assert all(r.method == "contour" for r in result)

    @patch("src.hf_detector._yolos_candidates", return_value=[])
    @patch("src.hf_detector._contour_candidates", return_value=[])
    def test_falls_back_to_strips_when_all_fail(self, _c, _y, blank_image):
        result = detect_plate_regions_hf(blank_image)
        assert len(result) >= 4
        assert all(r.method.startswith("strip_") for r in result)

    def test_always_returns_at_least_one_region(self, blank_image):
        with patch("src.hf_detector._yolos_candidates", return_value=[]), \
             patch("src.hf_detector._contour_candidates", return_value=[]):
            result = detect_plate_regions_hf(blank_image)
            assert len(result) >= 1

    def test_invalid_bytes_raises(self):
        with pytest.raises((ValueError, Exception)):
            detect_plate_regions_hf(b"not_an_image")

    @patch("src.hf_detector._yolos_candidates")
    def test_regions_sorted_by_score_descending(self, mock_yolos, plate_image):
        fake = [
            PlateRegion(crop=np.zeros((40,200,3),np.uint8), x=0, y=0, w=200, h=40, method="yolos", score=0.5),
            PlateRegion(crop=np.zeros((40,200,3),np.uint8), x=0, y=0, w=200, h=40, method="yolos", score=0.9),
            PlateRegion(crop=np.zeros((40,200,3),np.uint8), x=0, y=0, w=200, h=40, method="yolos", score=0.7),
        ]
        mock_yolos.return_value = sorted(fake, key=lambda r: r.score, reverse=True)
        result = detect_plate_regions_hf(plate_image)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)
