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

HuggingFace token (optional — only needed if the repo is gated):
  Set the HF_TOKEN environment variable:
    Windows:   set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
    Mac/Linux: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
  Or paste your token into the HF_TOKEN variable below.
  Get a free token at: https://huggingface.co/settings/tokens
"""

import os
import shutil
import sys

from huggingface_hub import hf_hub_download


REPO_ID  = "morsetechlab/yolov11-license-plate-detection"
FILENAME = "license-plate-finetune-v1n.pt"   # nano variant: lightest, fast on CPU
OUTPUT   = "plate_model.pt"

# Paste your HuggingFace token here if needed, or leave as None to use
# the HF_TOKEN environment variable (recommended — keeps secrets out of code).
HF_TOKEN: str | None = None


def download() -> None:
    token = HF_TOKEN or os.getenv("HF_TOKEN") or None

    print("Downloading pretrained model from HuggingFace...")
    print(f"  Repo:   {REPO_ID}")
    print(f"  File:   {FILENAME}")
    print(f"  Auth:   {'token provided' if token else 'no token (public access)'}")
    print()

    try:
        cached = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, token=token)
    except Exception as e:
        print(f"Download failed: {e}")
        print()
        print("If you see a 401 or 403 error:")
        print("  1. Get a free token at https://huggingface.co/settings/tokens")
        print("  2. Set it as an environment variable:")
        print("       Windows:   set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx")
        print("       Mac/Linux: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx")
        print("  3. Re-run this script.")
        print()
        print("Or try a different model variant by editing FILENAME:")
        print("  license-plate-finetune-v1s.pt  (small,  ~19 MB)")
        print("  license-plate-finetune-v1m.pt  (medium, ~40 MB)")
        sys.exit(1)

    shutil.copy(cached, OUTPUT)
    print(f"Saved to: {OUTPUT}")
    print()
    print("Verify it works:")
    print("  python -c \"from ultralytics import YOLO; m = YOLO('plate_model.pt'); print('OK:', m.names)\"")


if __name__ == "__main__":
    download()