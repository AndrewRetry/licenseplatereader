"""
test_reader.py — Debug-first test for the Singapore plate reader.

Prints every pipeline stage so you can see exactly where things break.

Usage:
  python test_reader.py plate_model.pt test_car.jpg
  python test_reader.py plate_model.pt            # uses webcam frame
"""

import re
import sys
import logging

import cv2
import numpy as np

from plate_reader import PlateReader

logging.basicConfig(level=logging.WARNING)   # suppress library noise


def test_with_image(model_path: str, image_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Image: {image_path}")
    print(f"{'='*60}")

    reader = PlateReader(model_path, detect_conf=0.25)

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image at '{image_path}'")
        return

    h, w = image.shape[:2]
    print(f"\n[Image]  {w} x {h} px")

    # ----------------------------------------------------------------
    # STEP 1: YOLO detection
    # ----------------------------------------------------------------
    print("\n[Step 1] YOLO detection (conf >= 0.25)")
    detections = reader._detect_plates(image)

    if not detections:
        print("  ✗ YOLO found NO plates.")
        print("    → Try a clearer image, or swap to the v1s/v1m model variant.")
        return

    print(f"  ✓ YOLO found {len(detections)} region(s):")
    for i, (bbox, conf) in enumerate(detections):
        print(f"    [{i+1}] conf={conf:.3f}  bbox={[round(v) for v in bbox]}")

    # ----------------------------------------------------------------
    # STEP 2–4: Walk through every detected region
    # ----------------------------------------------------------------
    final_plates = []

    for i, (bbox, conf) in enumerate(detections):
        x1, y1, x2, y2 = bbox
        print(f"\n[Step 2–4] Region {i+1}  (conf={conf:.3f})")

        # Crop
        crop = reader._crop_plate(image, x1, y1, x2, y2)
        crop_path = f"debug_crop_{i+1}.jpg"
        cv2.imwrite(crop_path, crop)
        print(f"  Crop: {crop.shape[1]} x {crop.shape[0]} px  → saved {crop_path}")

        # Colour normalisation
        normalised = reader._normalise_colour_scheme(crop)
        mean_brightness = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
        scheme = "white-on-black (inverted)" if mean_brightness < 100 else "black-on-white/yellow"
        print(f"  Colour scheme: {scheme}  (mean brightness={mean_brightness:.0f})")
        norm_path = f"debug_normalised_{i+1}.jpg"
        cv2.imwrite(norm_path, normalised)
        print(f"  Normalised → {norm_path}  (this is what TrOCR sees)")

        # TrOCR
        raw_text = reader._ocr_read(normalised)
        print(f"  TrOCR raw:  '{raw_text}'")

        if not raw_text.strip():
            print("  ✗ TrOCR returned empty — crop may be too small or blurry")
            continue

        # Clean
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper().strip())
        print(f"  Cleaned:    '{cleaned}'")

        if cleaned:
            final_plates.append({
                "text":       cleaned,
                "confidence": round(float(conf), 3),
                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
            })

    # ----------------------------------------------------------------
    # Summary + annotated output
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    if not final_plates:
        print("  No plates returned.")
    else:
        for p in final_plates:
            print(f"  ✓  '{p['text']}'  conf={p['confidence']}  bbox={p['bbox']}")

    annotated = image.copy()
    for p in final_plates:
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated, p["text"], (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

    out_path = "test_output.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\n  Annotated image → {out_path}")
    print()


def test_with_webcam(model_path: str) -> None:
    print("\n--- Capturing from webcam ---")

    # Try DirectShow first (more reliable on Windows), fall back to default
    for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(1, backend)
        if cap.isOpened():
            break
    else:
        print("No webcam available. Provide an image path instead.")
        return

    # Warm up — discard frames while camera auto-adjusts
    print("  Warming up camera...")
    for _ in range(30):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None or float(np.mean(frame)) < 5:
        print("Failed to capture a usable frame.")
        return

    capture_path = "webcam_capture.jpg"
    cv2.imwrite(capture_path, frame)
    print(f"  Saved {capture_path}")
    test_with_image(model_path, capture_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_reader.py <model.pt> [image.jpg]")
        sys.exit(1)

    model = sys.argv[1]
    if len(sys.argv) >= 3:
        test_with_image(model, sys.argv[2])
    else:
        test_with_webcam(model)