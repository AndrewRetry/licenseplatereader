"""
Debug visualisation.

Draws detected plate regions (and the winning candidate label) onto a copy
of the original image. Returns JPEG bytes.

Used exclusively by POST /api/plate/debug — never called in production flow.
"""

import io

import cv2
import numpy as np
from PIL import Image

from .plate_detector import PlateRegion

# Colour palette per detection method (BGR)
_COLOURS = {
    "contour": (0, 220, 90),    # green
    "haar":    (0, 160, 255),   # orange
}
_STRIP_COLOUR = (180, 180, 60)  # yellow-ish for strip fallbacks


def draw_regions(
    image_bytes: bytes,
    regions: list[PlateRegion],
    best_plate: str | None = None,
) -> bytes:
    """
    Draw bounding boxes for each detected region.

    Args:
        image_bytes:  raw input image
        regions:      list of PlateRegion from plate_detector
        best_plate:   plate string to label the highest-score region (optional)

    Returns:
        JPEG bytes of the annotated image
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image for debug visualisation")

    img_h, img_w = img.shape[:2]

    # Scale font and line thickness to image size
    scale = max(0.4, img_w / 1280)
    thickness = max(1, int(scale * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, region in enumerate(regions):
        colour = _COLOURS.get(region.method, _STRIP_COLOUR)

        # Draw bounding box
        cv2.rectangle(
            img,
            (region.x, region.y),
            (region.x + region.w, region.y + region.h),
            colour,
            thickness,
        )

        # Label: method + score
        label = f"{region.method}  {region.score:.3f}"
        if i == 0 and best_plate:
            label = f"{best_plate}  ({region.method})"

        label_y = max(region.y - 6, 14)

        # Shadow for legibility on any background
        cv2.putText(img, label, (region.x + 1, label_y + 1), font, scale * 0.55, (0, 0, 0), thickness + 1)
        cv2.putText(img, label, (region.x, label_y),          font, scale * 0.55, colour,    thickness)

    # Encode back to JPEG
    success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise RuntimeError("Failed to encode debug image")

    return buf.tobytes()
