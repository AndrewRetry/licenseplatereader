"""
download_model.py — Download a pretrained license plate detection model.

No training required. Downloads directly from HuggingFace.

Model: morsetechlab/yolov11-license-plate-detection
  - Trained on 10,125 annotated images (CC BY 4.0)
  - 300 epochs on NVIDIA A100
  - mAP@50: 0.981  |  Precision: 0.989  |  Recall: 0.951
  - Works globally — detects rectangular plate regions regardless of country

Run:
  python download_model.py

Output:
  plate_model.pt   (saved in the current directory, ~6 MB)
"""

from huggingface_hub import hf_hub_download
import shutil
import sys


REPO_ID  = "morsetechlab/yolov11-license-plate-detection"
FILENAME = "yolov11n-license-plate.pt"   # nano variant: lightest, fast on CPU
OUTPUT   = "plate_model.pt"


def download() -> None:
    print(f"Downloading pretrained model from HuggingFace...")
    print(f"  Repo:   {REPO_ID}")
    print(f"  File:   {FILENAME}")
    print()

    try:
        cached = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    except Exception as e:
        print(f"Download failed: {e}")
        print()
        print("Check your internet connection, or try the larger variants:")
        print("  yolov11s-license-plate.pt  (small)")
        print("  yolov11m-license-plate.pt  (medium)")
        sys.exit(1)

    shutil.copy(cached, OUTPUT)
    print(f"Saved to: {OUTPUT}")
    print()
    print("Verify it works:")
    print("  python -c \"from ultralytics import YOLO; m = YOLO('plate_model.pt'); print('OK:', m.names)\"")


if __name__ == "__main__":
    download()