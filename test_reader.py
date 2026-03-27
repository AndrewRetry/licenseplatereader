"""
test_reader.py — Quick test for the plate reader (no server needed).

Usage:
  python test_reader.py plate_model.pt test_car.jpg
  python test_reader.py plate_model.pt            # uses webcam frame
"""

import sys
import cv2
import logging
from plate_reader import PlateReader

logging.basicConfig(level=logging.INFO)


def test_with_image(model_path: str, image_path: str):
    """Test plate reader on a single image file."""
    print(f"\n--- Testing with image: {image_path} ---")
    reader = PlateReader(model_path)
    results = reader.read_from_path(image_path)

    if not results:
        print("No plates detected. Try:")
        print("  - A clearer image with visible plate")
        print("  - Lowering DETECT_CONF (e.g., 0.3)")
        return

    for i, plate in enumerate(results):
        print(f"  Plate {i+1}: {plate['text']}")
        print(f"    Confidence: {plate['confidence']}")
        print(f"    BBox:       {plate['bbox']}")

    # Optional: show annotated image
    image = cv2.imread(image_path)
    for plate in results:
        x1, y1, x2, y2 = plate["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, plate["text"], (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
        )

    out_path = "test_output.jpg"
    cv2.imwrite(out_path, image)
    print(f"\n  Annotated image saved to: {out_path}")


def test_with_webcam(model_path: str):
    """Capture a single frame from webcam and test."""
    print("\n--- Capturing from webcam ---")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam available. Provide an image path instead.")
        print("Usage: python test_reader.py plate_model.pt test_car.jpg")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture frame.")
        return

    cv2.imwrite("webcam_capture.jpg", frame)
    print("  Saved webcam_capture.jpg")
    test_with_image(model_path, "webcam_capture.jpg")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_reader.py <model.pt> [image.jpg]")
        sys.exit(1)

    model = sys.argv[1]
    if len(sys.argv) >= 3:
        test_with_image(model, sys.argv[2])
    else:
        test_with_webcam(model)
