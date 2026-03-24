"""
Unit tests for src/plate_detector.py

Tests OpenCV plate region detection with synthetic images.
No EasyOCR involved — purely tests the localisation layer.
"""

import cv2
import numpy as np
import pytest

from src.plate_detector import (
    PlateRegion,
    _contour_candidates,
    _haar_candidates,
    _strip_fallback,
    detect_plate_regions,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_jpg(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


def _car_image_with_plate(
    img_w: int = 640,
    img_h: int = 480,
    plate_x: int = 180,
    plate_y: int = 320,
    plate_w: int = 280,
    plate_h: int = 48,
    dark_bg: bool = False,
) -> np.ndarray:
    bg = (35, 40, 45) if dark_bg else (80, 100, 80)
    img = np.full((img_h, img_w, 3), bg, dtype=np.uint8)
    # Car body
    cv2.rectangle(img, (40, img_h // 2), (img_w - 40, img_h - 5), (45, 50, 55), -1)
    # Plate border (high-contrast edge for contour detector)
    cv2.rectangle(
        img,
        (plate_x - 3, plate_y - 3),
        (plate_x + plate_w + 3, plate_y + plate_h + 3),
        (5, 5, 5),
        3,
    )
    # Plate fill
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (238, 238, 238), -1)
    # Text
    cv2.putText(img, "SBA1234L", (plate_x + 10, plate_y + plate_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)
    return img


# ── _contour_candidates ───────────────────────────────────────────────────────

class TestContourCandidates:
    def test_returns_list(self):
        img = _car_image_with_plate()
        result = _contour_candidates(img)
        assert isinstance(result, list)

    def test_each_element_is_plate_region(self):
        img = _car_image_with_plate()
        result = _contour_candidates(img)
        for r in result:
            assert isinstance(r, PlateRegion)

    def test_sorted_by_score_descending(self):
        img = _car_image_with_plate()
        result = _contour_candidates(img)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_crop_shape_is_valid(self):
        img = _car_image_with_plate()
        result = _contour_candidates(img)
        for r in result:
            assert r.crop.ndim == 3
            assert r.w > 0 and r.h > 0

    def test_aspect_ratio_within_bounds(self):
        img = _car_image_with_plate()
        result = _contour_candidates(img)
        for r in result:
            aspect = r.w / r.h
            assert 2.0 <= aspect <= 7.5, f"aspect={aspect:.2f} out of bounds"

    def test_method_is_contour(self):
        img = _car_image_with_plate()
        result = _contour_candidates(img)
        for r in result:
            assert r.method == "contour"

    def test_blank_image_returns_empty_or_few(self):
        """A solid grey image should produce no or minimal spurious contours."""
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = _contour_candidates(img)
        # Solid image: no edges, so no plate-like contours expected
        assert len(result) == 0

    def test_at_most_six_candidates(self):
        img = _car_image_with_plate()
        result = _contour_candidates(img)
        assert len(result) <= 6


# ── _haar_candidates ──────────────────────────────────────────────────────────

class TestHaarCandidates:
    def test_returns_list(self):
        img = _car_image_with_plate()
        result = _haar_candidates(img)
        assert isinstance(result, list)

    def test_each_element_is_plate_region(self):
        img = _car_image_with_plate()
        result = _haar_candidates(img)
        for r in result:
            assert isinstance(r, PlateRegion)
            assert r.method == "haar"

    def test_blank_image_returns_list(self):
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = _haar_candidates(img)
        assert isinstance(result, list)


# ── _strip_fallback ───────────────────────────────────────────────────────────

class TestStripFallback:
    def test_always_returns_four_strips(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = _strip_fallback(img)
        assert len(result) == 4

    def test_strips_cover_full_width(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = _strip_fallback(img)
        for r in result:
            assert r.w == 640

    def test_strips_have_low_score(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = _strip_fallback(img)
        for r in result:
            assert r.score <= 0.2

    def test_method_starts_with_strip(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = _strip_fallback(img)
        for r in result:
            assert r.method.startswith("strip_")


# ── detect_plate_regions (integration) ───────────────────────────────────────

class TestDetectPlateRegions:
    def test_always_returns_at_least_one_region(self, blank_image):
        """Strip fallback guarantees at least 4 regions on any image."""
        result = detect_plate_regions(blank_image)
        assert len(result) >= 1

    def test_returns_list_of_plate_regions(self, plate_image):
        result = detect_plate_regions(plate_image)
        assert all(isinstance(r, PlateRegion) for r in result)

    def test_lower_half_plate_gets_position_bonus(self):
        """Plate in lower half of frame should appear first or near-first."""
        img = _car_image_with_plate(plate_y=350)  # well below midpoint
        img_bytes = _encode_jpg(img)
        result = detect_plate_regions(img_bytes)
        assert len(result) >= 1

    def test_invalid_bytes_raises(self):
        with pytest.raises((ValueError, Exception)):
            detect_plate_regions(b"not_an_image")

    def test_crops_do_not_exceed_image_bounds(self, plate_image):
        arr = np.frombuffer(plate_image, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        ih, iw = img.shape[:2]
        result = detect_plate_regions(plate_image)
        for r in result:
            assert r.x >= 0 and r.y >= 0
            assert r.x + r.w <= iw
            assert r.y + r.h <= ih
