"""
test_reader.py — Debug-first test for the Singapore plate reader.

Prints every pipeline stage so you can see exactly where things break.

Saved debug images per detected region:
  debug_crop_N.jpg        — raw YOLO crop
  debug_normalised_N.jpg  — after colour scheme inversion (what used to go to TrOCR)
  debug_processed_N.jpg   — after CLAHE preprocessing (what actually goes to TrOCR)
  test_output.jpg         — annotated final result

Usage:
  python test_reader.py plate_model.pt test_car.jpg
  python test_reader.py plate_model.pt            # uses webcam / DroidCam frame

DroidCam note:
  DroidCam registers as a DirectShow virtual device — MSMF (Windows default backend)
  cannot see it and will always return black frames. This script uses CAP_DSHOW
  exclusively and scans indices 0–_WEBCAM_MAX_INDEX to find whatever index
  DroidCam was assigned.
"""

import re
import sys
import time
import logging

import cv2
import numpy as np

from plate_reader import PlateReader

logging.basicConfig(level=logging.WARNING)   # suppress library noise

# Scan this many indices looking for any working camera (DroidCam can land anywhere)
_WEBCAM_MAX_INDEX = 6

# After open(), wait up to this long for a non-black frame (real cameras need time)
_WEBCAM_RETRY_DELAY  = 0.1   # seconds between reads
_WEBCAM_RETRIES      = 15    # 15 × 0.1 s = 1.5 s max per index
_WEBCAM_MIN_BRIGHTNESS = 10.0


def _scan_dshow_cameras() -> None:
    """
    Print which DirectShow indices open successfully.
    Useful for diagnosing which index DroidCam is on.
    """
    print("  Scanning DirectShow camera indices...")
    for i in range(_WEBCAM_MAX_INDEX):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            brightness = float(np.mean(frame)) if (ret and frame is not None) else 0.0
            print(f"    index {i}: opened  brightness={brightness:.1f}")
            cap.release()
        else:
            print(f"    index {i}: not available")


def _try_camera_dshow(index: int) -> tuple[cv2.VideoCapture | None, np.ndarray | None]:
    """
    Open a DirectShow camera at the given index and wait for a usable frame.

    Uses CAP_DSHOW exclusively — DroidCam and other virtual cameras are
    DirectShow devices and are invisible to MSMF (Windows Media Foundation).

    Returns (cap, frame) on success, (None, None) on failure.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None, None

    # Ask for a standard resolution — prompts DirectShow to negotiate format
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for attempt in range(_WEBCAM_RETRIES):
        ret, frame = cap.read()
        if ret and frame is not None:
            brightness = float(np.mean(frame))
            if brightness > _WEBCAM_MIN_BRIGHTNESS:
                return cap, frame
            print(
                f"    index {index}: attempt {attempt + 1:02d}/{_WEBCAM_RETRIES}"
                f"  brightness={brightness:.1f}"
            )
        time.sleep(_WEBCAM_RETRY_DELAY)

    cap.release()
    return None, None


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
    # STEP 2–5: Walk through every detected region
    # ----------------------------------------------------------------
    final_plates = []

    for i, (bbox, conf) in enumerate(detections):
        x1, y1, x2, y2 = bbox
        print(f"\n[Steps 2–5] Region {i+1}  (conf={conf:.3f})")

        # Step 2: Crop
        crop = reader._crop_plate(image, x1, y1, x2, y2)
        crop_path = f"debug_crop_{i+1}.jpg"
        cv2.imwrite(crop_path, crop)
        print(f"  [2] Crop:       {crop.shape[1]} x {crop.shape[0]} px  → {crop_path}")

        # Step 3: Colour normalisation
        normalised = reader._normalise_colour_scheme(crop)
        mean_brightness = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
        scheme = "white-on-black (inverted)" if mean_brightness < 100 else "black-on-white/yellow"
        norm_path = f"debug_normalised_{i+1}.jpg"
        cv2.imwrite(norm_path, normalised)
        print(f"  [3] Normalised: {scheme}  (mean brightness={mean_brightness:.0f})  → {norm_path}")

        # Step 4: CLAHE preprocessing — what TrOCR actually receives
        processed_rgb = reader._preprocess_for_ocr(normalised)
        proc_path = f"debug_processed_{i+1}.jpg"
        cv2.imwrite(proc_path, cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR))
        print(f"  [4] Processed:  CLAHE enhanced  → {proc_path}  (this is what TrOCR sees)")

        # Step 5: TrOCR
        raw_text = reader._ocr_read(normalised)
        print(f"  [5] TrOCR raw:  '{raw_text}'")

        if not raw_text.strip():
            print("      ✗ TrOCR returned empty — crop may be too small or blurry")
            continue

        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper().strip())
        print(f"      Cleaned:    '{cleaned}'")

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
    print("\n--- Capturing from webcam (DirectShow only) ---")

    # Show a full scan first so the user knows exactly what's visible
    _scan_dshow_cameras()

    cap = None
    frame = None
    used_index = None

    for index in range(_WEBCAM_MAX_INDEX):
        cap, frame = _try_camera_dshow(index)
        if cap is not None:
            used_index = index
            break

    if cap is None:
        print(
            "\nNo usable camera found via DirectShow.\n"
            "  If using DroidCam:\n"
            "    - Open the DroidCam Windows client and confirm it shows 'connected'\n"
            "    - The phone and PC must be on the same Wi-Fi, or connected via USB\n"
            "    - Try the DroidCam app's built-in test — if that shows black, it's a DroidCam issue\n"
            "  Otherwise:\n"
            f"    - Increase _WEBCAM_MAX_INDEX (currently {_WEBCAM_MAX_INDEX})\n"
            "    - Pass an image directly:  python test_reader.py plate_model.pt image.jpg"
        )
        return

    cap.release()

    capture_path = "webcam_capture.jpg"
    cv2.imwrite(capture_path, frame)
    print(f"\n  Saved {capture_path}  (camera index={used_index}, backend=DirectShow)")
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