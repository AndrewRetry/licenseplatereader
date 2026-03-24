"""
Shared pytest fixtures.

Generates synthetic test images in-memory — no external files needed.
"""

import io
import sys
import os

import cv2
import numpy as np
import pytest

# Make src importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_plate_image(
    plate_text: str = "SBA 1234 L",
    img_w: int = 640,
    img_h: int = 480,
    plate_x: int = 180,
    plate_y: int = 310,
    plate_w: int = 280,
    plate_h: int = 50,
    dark_bg: bool = False,
) -> bytes:
    """
    Render a synthetic car image with a white rectangular plate in a known position.
    Returns JPEG bytes.
    """
    bg_colour = (35, 40, 45) if dark_bg else (90, 110, 90)
    img = np.full((img_h, img_w, 3), bg_colour, dtype=np.uint8)

    # Rough car body
    cv2.rectangle(img, (50, img_h // 2), (img_w - 50, img_h - 10), (50, 55, 60), -1)

    # Plate border
    cv2.rectangle(
        img,
        (plate_x - 2, plate_y - 2),
        (plate_x + plate_w + 2, plate_y + plate_h + 2),
        (10, 10, 10),
        2,
    )
    # Plate fill
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (240, 240, 240), -1)

    # Plate text
    cv2.putText(
        img,
        plate_text,
        (plate_x + 8, plate_y + plate_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (10, 10, 10),
        2,
    )

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


@pytest.fixture
def plate_image() -> bytes:
    """Standard day-time car image with SBA1234L plate."""
    return _make_plate_image("SBA1234L")


@pytest.fixture
def night_plate_image() -> bytes:
    """Dark background car image."""
    return _make_plate_image("SHA5678H", dark_bg=True)


@pytest.fixture
def small_plate_image() -> bytes:
    """Plate is smaller and higher in the frame — tests strip fallback."""
    return _make_plate_image("SX9999K", plate_y=200, plate_w=140, plate_h=30)


@pytest.fixture
def blank_image() -> bytes:
    """Solid grey image — no plate. Should return success=False."""
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def make_plate_image():
    """Factory fixture — call with any arguments."""
    return _make_plate_image
