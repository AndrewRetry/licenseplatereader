"""
train_colab.py — Google Colab training script for Singapore license plate detection.

SINGAPORE PLATE CONTEXT
-----------------------
Singapore plates use standard rectangular shapes with Latin alphanumeric characters
(same as US/EU). The YOLO model only needs to detect WHERE the plate is — it does
not read the characters. Because of this, a globally-sourced dataset works well for
detection, even without Singapore-specific images.

The critical Singapore-specific handling lives in plate_reader.py:
  - Colour scheme normalisation (white-on-black vs black-on-white/yellow)
  - Position-aware OCR corrections for the SBA-1234-A format
  - Plate format validation (regex + valid checksum letter range)

For best accuracy, supplement the global dataset with your own Singapore footage
(see CELL 3b below).

STEPS
-----
  1. Open https://colab.research.google.com → New Notebook
  2. Runtime → Change runtime type → T4 GPU (free tier is sufficient)
  3. Copy each CELL block into a separate Colab cell and run in order
  4. Download plate_model.pt at the end and place it in your project folder
"""


# === CELL 1: Install dependencies ===

# !pip install ultralytics roboflow


# === CELL 2: Download the primary license plate dataset ===
#
# Dataset: "License Plate Recognition" by Roboflow Universe Projects
# URL:     https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
# Images:  ~10,000+ annotated images from global locations (CC BY 4.0 licence)
# Format:  YOLOv8 — bounding boxes around license plates (one class: "License_Plate")
#
# WHY this dataset works for Singapore:
#   YOLO is only trained to find WHERE the plate is (rectangle detection).
#   It does not learn country-specific characters. The model transfers well to
#   Singapore plates because all plates are rectangular regions with text.
#   Singapore-specific character reading is handled by EasyOCR + post-processing.
#
# You will be prompted to log in with a free Roboflow account.

# from roboflow import Roboflow
# rf = Roboflow()
# project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
#
# # Use the latest available version.
# # v4 has ~10k images; newer versions may have more. Check the Roboflow page above.
# dataset = project.version(4).download("yolov8")
# print(f"Primary dataset downloaded to: {dataset.location}")
# DATASET_YAML = f"{dataset.location}/data.yaml"


# === CELL 3a (Optional): Add a second global dataset to increase variety ===
#
# Merging a second dataset improves robustness for different plate angles,
# lighting conditions, and plate styles similar to Singapore's (rectangular, single-line).
#
# Dataset: "License Plates" from Open Images (subset)
# URL:     https://universe.roboflow.com/samrat-sahoo/license-plates-f8vsn
# Images:  ~350 additional images (CC BY 2.0)
#
# NOTE: You can merge datasets in Roboflow's UI (free), then re-download the merged
# version. Alternatively, keep training on the primary dataset alone — it is sufficient.


# === CELL 3b (Recommended for production): Collect your own Singapore footage ===
#
# For best real-world accuracy at your specific gantry, collect 100–500 images of
# Singapore vehicles at your site. This is the single biggest accuracy improvement.
#
# Steps:
#   1. Record a video at your gantry during daylight and at night (~5 min each)
#   2. Use vid2img.py (in this repo) to extract frames:
#      python vid2img.py --input gantry_day.mp4 --output sg_frames/ --fps 2
#   3. Annotate the plates using LabelImg (free):
#      pip install labelImg
#      labelImg sg_frames/
#      → Draw bounding boxes, save as YOLO format (.txt per image)
#   4. Upload annotated images to Roboflow → fork the primary dataset → merge
#   5. Re-download and use the merged dataset for training (Cell 3b)
#
# Even 100 Singapore-specific images added to the 10k global set noticeably
# improves accuracy on white-on-black plates and ERP gantry angles.


# === CELL 4: Train YOLOv8n (nano) on the dataset ===
#
# YOLOv8n = nano variant: ~6 MB, adequate for gantry/edge deployment.
# Training takes ~1–2 hours on a free T4 GPU.

# from ultralytics import YOLO
#
# model = YOLO("yolov8n.pt")   # Start from pretrained COCO weights (transfer learning)
#
# results = model.train(
#     data=DATASET_YAML,
#     epochs=50,        # 50 is a good starting point; try 80 if mAP is below 0.85
#     imgsz=640,        # Standard YOLO input size
#     batch=16,         # Fits in T4 16 GB VRAM; reduce to 8 if you hit OOM errors
#     name="sg_plate_detector",
#     patience=15,      # Stop early if validation loss doesn't improve for 15 epochs
#     save=True,
#     plots=True,       # Generates training curves (loss, mAP over epochs)
# )
#
# print("Training complete!")
# print("Best weights saved to: runs/detect/sg_plate_detector/weights/best.pt")
# print(f"Final mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'see plots')}")


# === CELL 5: Evaluate the trained model ===
#
# Target metrics for Singapore gantry use:
#   mAP@0.5      > 0.88  (how well it finds plate regions)
#   Precision    > 0.90  (low false positives — important for gantry)
#   Recall       > 0.85  (low missed plates)
#
# If recall is low, reduce detect_conf in plate_reader.py (try 0.35).
# If precision is low, increase detect_conf (try 0.60) or collect more SG images.

# from ultralytics import YOLO
# model = YOLO("runs/detect/sg_plate_detector/weights/best.pt")
# metrics = model.val()
# print(f"mAP@0.5:      {metrics.box.map50:.3f}")
# print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
# print(f"Precision:    {metrics.box.mp:.3f}")
# print(f"Recall:       {metrics.box.mr:.3f}")


# === CELL 6: Quick visual test on a sample image ===
#
# from ultralytics import YOLO
# from google.colab.patches import cv2_imshow
# import glob, cv2
#
# model = YOLO("runs/detect/sg_plate_detector/weights/best.pt")
# test_images = glob.glob(f"{dataset.location}/test/images/*")
#
# if test_images:
#     result = model(test_images[0])[0]
#     annotated = result.plot()
#     cv2_imshow(annotated)
#     print(f"Detected {len(result.boxes)} plate(s)")


# === CELL 7: Download the trained model ===
#
# Saves the best checkpoint locally as plate_model.pt.
# Place this file in your project folder alongside plate_reader.py.

# import shutil
# shutil.copy(
#     "runs/detect/sg_plate_detector/weights/best.pt",
#     "plate_model.pt",
# )
#
# from google.colab import files
# files.download("plate_model.pt")
# print("Done! Place plate_model.pt in your project folder.")


# === CELL 8 (Optional): Export to ONNX for edge deployment ===
#
# ONNX runs on CPU without the full PyTorch stack — useful for gantry edge devices.
#
# from ultralytics import YOLO
# model = YOLO("plate_model.pt")
# model.export(format="onnx", imgsz=640)
# # Downloads plate_model.onnx — load with YOLO("plate_model.onnx") in plate_reader.py