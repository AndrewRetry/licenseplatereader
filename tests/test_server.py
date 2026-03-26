"""
Integration tests for the FastAPI server.

Uses TestClient (sync, no real HTTP) — fast and dependency-free.
The detect endpoint is mocked so these tests don't require EasyOCR/GPU.
"""

import json
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import app


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """
    Session-scoped TestClient wrapping the full lifespan context.
    Using `with TestClient(app)` ensures startup/shutdown hooks fire exactly
    once per test session rather than being skipped entirely.
    """
    with TestClient(app) as c:
        yield c


# ── Shared mock payloads ───────────────────────────────────────────────────────

_MOCK_RESULT = {
    "success": True,
    "elapsedMs": 42,
    "best": {
        "plate": "SBA1234L",
        "formatted": "SBA 1234 L",
        "prefix": "SBA",
        "digits": "1234",
        "checksum": "L",
        "checksumValid": True,
        "format": "private_car",
        "confidence": "high",
        "ocrConfidence": 91.0,
        "method": "yolos",
        "rawOcrText": "SBA1234L",
        "score": 106.0,
    },
    "candidates": [],
    "meta": {
        "regionsInspected": 3,
        "ocrAttempts": 3,
        "detectionEngine": "yolos",
        "ocrEngine": "trocr",
    },
}

_MOCK_FAILURE = {
    "success": False,
    "elapsedMs": 10,
    "best": None,
    "candidates": [],
    "meta": {"regionsInspected": 4, "ocrAttempts": 4},
}


def _jpg_bytes(text: str = "SBA1234L") -> bytes:
    img = np.full((120, 320, 3), 200, dtype=np.uint8)
    cv2.putText(img, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ── /api/plate/health ──────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/api/plate/health")
        assert r.status_code == 200

    def test_status_ok(self, client):
        r = client.get("/api/plate/health")
        assert r.json()["status"] == "ok"

    def test_engine_field(self, client):
        r = client.get("/api/plate/health")
        assert "yolos" in r.json()["engine"] or "trocr" in r.json()["engine"]

    def test_version_present(self, client):
        r = client.get("/api/plate/health")
        assert "version" in r.json()


# ── Convenience redirects ──────────────────────────────────────────────────────

class TestRedirects:
    def test_root_redirects_to_health(self, client):
        r = client.get("/", follow_redirects=True)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_slash_health_redirects(self, client):
        r = client.get("/health", follow_redirects=True)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_favicon_returns_no_content(self, client):
        r = client.get("/favicon.ico")
        assert r.status_code in (204, 404)   # 204 preferred, 404 acceptable


# ── /api/plate/detect ──────────────────────────────────────────────────────────

class TestDetectSingle:
    @patch("server.detect", return_value=_MOCK_RESULT)
    def test_200_with_valid_jpeg(self, _mock, client):
        r = client.post("/api/plate/detect", files={"image": ("car.jpg", _jpg_bytes(), "image/jpeg")})
        assert r.status_code == 200

    @patch("server.detect", return_value=_MOCK_RESULT)
    def test_response_has_request_id(self, _mock, client):
        r = client.post("/api/plate/detect", files={"image": ("car.jpg", _jpg_bytes(), "image/jpeg")})
        assert "requestId" in r.json()

    @patch("server.detect", return_value=_MOCK_RESULT)
    def test_response_has_best_plate(self, _mock, client):
        r = client.post("/api/plate/detect", files={"image": ("car.jpg", _jpg_bytes(), "image/jpeg")})
        assert r.json()["best"]["plate"] == "SBA1234L"

    @patch("server.detect", return_value=_MOCK_RESULT)
    def test_response_has_raw_ocr_text(self, _mock, client):
        """rawOcrText must be camelCase in the response — not raw_ocr_text."""
        r = client.post("/api/plate/detect", files={"image": ("car.jpg", _jpg_bytes(), "image/jpeg")})
        assert "rawOcrText" in r.json()["best"]
        assert "raw_ocr_text" not in r.json()["best"]

    @patch("server.detect", return_value=_MOCK_FAILURE)
    def test_200_even_when_no_plate_found(self, _mock, client):
        """No plate → still 200 with success=False (not a server error)."""
        r = client.post("/api/plate/detect", files={"image": ("blank.jpg", _jpg_bytes(), "image/jpeg")})
        assert r.status_code == 200
        assert r.json()["success"] is False

    def test_400_on_unsupported_mime(self, client):
        r = client.post("/api/plate/detect", files={"image": ("doc.pdf", b"%PDF", "application/pdf")})
        assert r.status_code == 400

    def test_422_when_no_file_field(self, client):
        r = client.post("/api/plate/detect")
        assert r.status_code == 422

    @patch("server.detect", side_effect=RuntimeError("boom"))
    def test_500_on_internal_error(self, _mock, client):
        r = client.post("/api/plate/detect", files={"image": ("car.jpg", _jpg_bytes(), "image/jpeg")})
        assert r.status_code == 500


# ── /api/plate/detect/batch ────────────────────────────────────────────────────

class TestDetectBatch:
    @patch("server.detect", return_value=_MOCK_RESULT)
    def test_200_with_two_images(self, _mock, client):
        files = [
            ("images", ("a.jpg", _jpg_bytes(), "image/jpeg")),
            ("images", ("b.jpg", _jpg_bytes(), "image/jpeg")),
        ]
        r = client.post("/api/plate/detect/batch", files=files)
        assert r.status_code == 200

    @patch("server.detect", return_value=_MOCK_RESULT)
    def test_response_has_results_array(self, _mock, client):
        files = [("images", ("a.jpg", _jpg_bytes(), "image/jpeg"))]
        r = client.post("/api/plate/detect/batch", files=files)
        assert "results" in r.json()

    @patch("server.detect", return_value=_MOCK_RESULT)
    def test_count_matches_uploaded(self, _mock, client):
        files = [("images", (f"{i}.jpg", _jpg_bytes(), "image/jpeg")) for i in range(3)]
        r = client.post("/api/plate/detect/batch", files=files)
        assert r.json()["count"] == 3

    def test_400_when_more_than_10_images(self, client):
        files = [("images", (f"{i}.jpg", _jpg_bytes(), "image/jpeg")) for i in range(11)]
        r = client.post("/api/plate/detect/batch", files=files)
        assert r.status_code == 400


# ── /api/plate/validate ────────────────────────────────────────────────────────

class TestValidate:
    def test_valid_plate_returns_true(self, client):
        r = client.post("/api/plate/validate", json={"plate": "SBA1234L"})
        assert r.status_code == 200
        assert r.json()["valid"] is True

    def test_valid_plate_returns_format(self, client):
        r = client.post("/api/plate/validate", json={"plate": "SBA1234L"})
        assert r.json()["format"] == "private_car"

    def test_invalid_plate_returns_false(self, client):
        r = client.post("/api/plate/validate", json={"plate": "XXXXXXXXX"})
        assert r.json()["valid"] is False

    def test_government_plate(self, client):
        r = client.post("/api/plate/validate", json={"plate": "QX1234"})
        assert r.json()["valid"] is True
        assert r.json()["format"] == "government"

    def test_missing_body_422(self, client):
        r = client.post("/api/plate/validate")
        assert r.status_code == 422

    def test_formatted_field_present(self, client):
        r = client.post("/api/plate/validate", json={"plate": "SBA1234L"})
        assert "formatted" in r.json()

    def test_lowercase_plate_normalised(self, client):
        """Lowercase input should still validate (normalise handles it)."""
        r = client.post("/api/plate/validate", json={"plate": "sba1234l"})
        assert r.json()["valid"] is True