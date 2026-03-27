# Gantry License Plate Reader — Beginner Tutorial

A lightweight license plate detection + OCR system exposed as a REST API.
Returns the plate string from an image.

## Architecture

```
Camera/Image ──► YOLOv8n (detect plate) ──► Crop ──► EasyOCR (read text) ──► "SGX1234A"
                    ~6 MB model              OpenCV       PyTorch-based
```

**Why this stack (not the repo's YOLOv7 + PaddleOCR)?**

| Choice        | Why                                                        |
|---------------|------------------------------------------------------------|
| YOLOv8n       | Same accuracy, 1-line pip install, no cloning repos needed |
| EasyOCR       | PyTorch-based (same as YOLO), simpler install than Paddle  |
| FastAPI       | Async, auto-docs at `/docs`, production-ready              |

Everything is **free and open-source**. No API keys needed.

---

## PHASE 1 — Set Up Your Machine (30 min)

### Prerequisites
- Python 3.9 – 3.11 (3.12 can have issues with some CV libs)
- Git
- ~4 GB free disk space (for model weights + dependencies)

### Step 1: Create the project
```bash
mkdir license-plate-reader && cd license-plate-reader
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 2: Install dependencies
```bash
pip install ultralytics easyocr fastapi uvicorn python-multipart opencv-python-headless Pillow
```

That's it. No compiling. No CUDA required (GPU optional, CPU works fine for gantry).

---

## PHASE 2 — Get a Trained Plate Detection Model (1–3 hrs)

YOLOv8's default model detects 80 COCO classes (car, dog, etc.) but **NOT license plates**.
You need a model fine-tuned on license plate images.

### Option A: Train on Google Colab (FREE GPU) — Recommended

1. Go to [Google Colab](https://colab.research.google.com) → New Notebook
2. Runtime → Change runtime type → **T4 GPU**
3. Paste this into cells and run:

```python
# Cell 1 — Install
!pip install ultralytics roboflow

# Cell 2 — Download dataset from Roboflow (free, 24k images, CC BY 4.0)
from roboflow import Roboflow
rf = Roboflow()  # Will prompt you to log in (free account)
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(4)
dataset = version.download("yolov8")

# Cell 3 — Train (takes ~1-2 hours on T4 GPU)
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # nano = lightweight
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="plate_detector"
)

# Cell 4 — Download your trained model
from google.colab import files
files.download("runs/detect/plate_detector/weights/best.pt")
```

4. Save the downloaded `best.pt` into your project folder as `plate_model.pt`

### Option B: Use a Community Pre-trained Model (faster, less accurate)

Search [Roboflow Universe](https://universe.roboflow.com/search?q=class:%22license+plate%22)
for "license plate" → filter by YOLOv8 → download weights.

### Verify your model works
```bash
python -c "
from ultralytics import YOLO
model = YOLO('plate_model.pt')
print('Model loaded ✓  Classes:', model.names)
"
```
Expected output: `{0: 'License_Plate'}` or similar.

---

## PHASE 3 — Build the Plate Reader (the code)

Your project should look like this:
```
license-plate-reader/
├── plate_model.pt        ← your trained YOLO weights
├── plate_reader.py       ← core detection + OCR logic
├── server.py             ← FastAPI REST endpoint
├── test_reader.py        ← quick local test script
└── requirements.txt
```

### plate_reader.py — Core Logic
See the file in this project. Key flow:
1. YOLO detects bounding boxes of plates in the image
2. Crop each plate region with a small padding
3. Preprocess the crop (grayscale, contrast enhancement)
4. EasyOCR reads the text
5. Post-process: strip spaces, uppercase, keep only alphanumeric

### server.py — FastAPI Endpoint
Exposes `POST /read-plate` that accepts an image and returns:
```json
{
  "success": true,
  "plates": [
    {
      "text": "SGX1234A",
      "confidence": 0.94,
      "bbox": [120, 340, 380, 420]
    }
  ]
}
```

---

## PHASE 4 — Run It

### Start the API server
```bash
python server.py
```
Server starts at `http://localhost:8000`

### Test with curl
```bash
curl -X POST http://localhost:8000/read-plate \
  -F "file=@test_car.jpg"
```

### Test with Python
```python
import requests
resp = requests.post(
    "http://localhost:8000/read-plate",
    files={"file": open("test_car.jpg", "rb")}
)
print(resp.json())
```

### Interactive docs
Open `http://localhost:8000/docs` in your browser — you can upload images directly.

---

## PHASE 5 — Integrate with Your Gantry System

Your gantry service calls this API whenever a vehicle arrives:

```python
import requests

def detect_plate(image_path: str) -> str | None:
    """Call the plate reader API and return the plate string."""
    with open(image_path, "rb") as f:
        resp = requests.post(
            "http://plate-reader-host:8000/read-plate",
            files={"file": f}
        )
    data = resp.json()
    if data["success"] and data["plates"]:
        return data["plates"][0]["text"]
    return None
```

---

## Tips for Gantry Deployment

1. **Camera**: Position at 1–2m height, slight downward angle toward plates
2. **Resolution**: 1080p minimum. 640×640 is what YOLO resizes to internally
3. **Lighting**: IR illuminator helps at night (plates are retro-reflective)
4. **Frame selection**: Don't OCR every frame — pick the sharpest one when vehicle is stationary
5. **Confidence threshold**: Set `DETECT_CONF=0.5` or higher to avoid false positives
6. **Multiple reads**: Read 3–5 frames and pick the most common result (majority vote)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Make sure venv is activated |
| YOLO finds no plates | Lower `DETECT_CONF` to 0.3, check image quality |
| OCR reads garbage | Check the crop preview, improve lighting/angle |
| Slow on CPU | Use `yolov8n` (nano), set `DETECT_CONF=0.5` to reduce processing |
| Wrong characters | Add post-processing rules for your plate format (see `_clean_plate_text`) |
