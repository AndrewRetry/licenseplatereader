"""
train_colab.py — Copy-paste this into Google Colab cells to train your model.

Steps:
  1. Open https://colab.research.google.com
  2. Runtime → Change runtime type → T4 GPU
  3. Copy each "# === CELL N ===" block into separate Colab cells
  4. Run them in order
  5. Download the trained plate_model.pt at the end
"""

# === CELL 1: Install dependencies ===
# !pip install ultralytics roboflow


# === CELL 2: Download the Roboflow license plate dataset ===
# This dataset has ~24,000 annotated plate images (CC BY 4.0 license).
# You'll need a free Roboflow account — it will prompt you to log in.
#
# from roboflow import Roboflow
# rf = Roboflow()
# project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
# version = project.version(4)
# dataset = version.download("yolov8")
# print(f"Dataset downloaded to: {dataset.location}")


# === CELL 3: Train YOLOv8n on the dataset ===
# YOLOv8n = "nano" — smallest, fastest, ~6MB model file.
# 50 epochs takes ~1-2 hours on a T4 GPU.
#
# from ultralytics import YOLO
#
# model = YOLO("yolov8n.pt")  # start from pretrained COCO weights
#
# results = model.train(
#     data=f"{dataset.location}/data.yaml",
#     epochs=50,            # more epochs = better accuracy (diminishing returns after ~80)
#     imgsz=640,            # YOLO resizes all images to this
#     batch=16,             # 16 works on T4's 16GB VRAM
#     name="plate_detector",
#     patience=10,          # early stopping if no improvement for 10 epochs
#     save=True,
#     plots=True,           # generates training charts
# )
#
# print("Training complete!")
# print(f"Best weights: runs/detect/plate_detector/weights/best.pt")


# === CELL 4: Evaluate the model ===
# from ultralytics import YOLO
# model = YOLO("runs/detect/plate_detector/weights/best.pt")
# metrics = model.val()
# print(f"mAP@0.5:     {metrics.box.map50:.3f}")
# print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
# print(f"Precision:    {metrics.box.mp:.3f}")
# print(f"Recall:       {metrics.box.mr:.3f}")


# === CELL 5: Quick test on an image ===
# import cv2
# from google.colab.patches import cv2_imshow
#
# model = YOLO("runs/detect/plate_detector/weights/best.pt")
#
# # Test on a validation image
# import glob
# test_images = glob.glob(f"{dataset.location}/test/images/*")
# if test_images:
#     results = model(test_images[0])
#     annotated = results[0].plot()
#     cv2_imshow(annotated)


# === CELL 6: Download the trained model ===
# This saves best.pt to your local machine as plate_model.pt.
#
# import shutil
# shutil.copy(
#     "runs/detect/plate_detector/weights/best.pt",
#     "plate_model.pt"
# )
#
# from google.colab import files
# files.download("plate_model.pt")
# print("Done! Place plate_model.pt in your project folder.")
