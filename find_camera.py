"""
Run this to find your working camera index and backend on Windows.

    python find_camera.py

It will open a preview window for each camera it finds so you can
confirm which index is your webcam.
"""

import cv2

BACKENDS = [
    (cv2.CAP_MSMF,  "MSMF (recommended on Windows)"),
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_ANY,   "Auto"),
]

print("Scanning camera indices 0–4 across backends...\n")

found = []

for index in range(5):
    for backend, label in BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  ✓ index={index}  backend={label}  resolution={w}x{h}")
                found.append((index, backend, label))

                # Show a preview so you can visually confirm
                cv2.imshow(f"Camera {index} — {label} (press any key)", frame)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
            cap.release()
            break   # found a working backend for this index, skip the rest

if not found:
    print("\nNo cameras found at all.")
    print("Check: Device Manager > Cameras — is your webcam listed and enabled?")
else:
    print(f"\nAdd the best match to your .env:")
    idx, _, label = found[0]
    print(f"  CAMERA_INDEX={idx}")
    print(f"  # Backend: {label}")
    print(f"\nIf that index isn't your webcam, try the others listed above.")