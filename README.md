# Gantry License Plate Reader — Singapore

A lightweight license plate detection + OCR system tuned for **Singapore LTA plates**,
exposed as a REST API. Returns the plate string from a single image.

## Architecture

```
Camera/Image ──► YOLOv8n ──────► Crop ──► Colour ──► EasyOCR ──► SG Validate ──► "SBA1234A"
                  Detect plate    OpenCV    Normalise    PyTorch      Regex + fix
```

## Singapore Plate Format

Singapore plates follow the LTA-mandated `SBA 1234 A` format:

| Part        | Detail                                                         |
|-------------|----------------------------------------------------------------|
| Prefix      | 1–3 letters (I and O never used; no vowel in 2nd char of 3-letter prefix) |
| Numbers     | 1–4 digits                                                     |
| Checksum    | 1 letter (F, I, N, O, Q, V, W never used as check digits)     |
| Max length  | 8 characters — e.g. `SBA1234A`                                |

**Two LTA-approved colour schemes** (both handled automatically):
- **White-on-black** — white text on black background (most common from dealerships)
- **Black-on-white** (front) / **black-on-yellow** (rear) — Euro reflective scheme

This system auto-detects the colour scheme on each plate crop and normalises it
before OCR, so both schemes work without any configuration.

---

## Why This Stack

| Choice    | Why                                                                      |
|-----------|--------------------------------------------------------------------------|
| YOLOv8n   | Detects plate *location* only — works globally regardless of plate style |
| EasyOCR   | PyTorch-based, English support is perfect for SG's Latin alphanumerics   |
| FastAPI   | Async, auto-docs at `/docs`, production-ready                            |

No API keys. No cloud fees. Everything runs locally.

---

## PHASE 1 — Set Up Your Machine

### Prerequisites
- Python 3.9–3.11
- Git
- ~4 GB free disk space

### Step 1: Create the project
```bash
mkdir sg-plate-reader && cd sg-plate-reader
python -m venv venv

# Activate
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

---

## PHASE 2 — Download the Pretrained Model (no training required)

No training required. A fully pretrained model (~6 MB) is available for free on HuggingFace.

```bash
python download_model.py
```

That's it. This saves `plate_model.pt` in your project folder.

**What you're getting:**

| Detail        | Value                                                           |
|---------------|-----------------------------------------------------------------|
| Model         | YOLO11n (nano) from `morsetechlab/yolov11-license-plate-detection` |
| Training data | 10,125 annotated plate images (CC BY 4.0)                      |
| Trained for   | 300 epochs on NVIDIA A100                                       |
| mAP@50        | 0.981                                                           |
| Precision     | 0.989 / Recall: 0.951                                           |
| Licence       | AGPLv3 — free for open-source projects                         |

### Verify the download
```bash
python -c "from ultralytics import YOLO; m = YOLO('plate_model.pt'); print('OK:', m.names)"
```
Expected output: `OK: {0: 'license-plate'}` or similar.

---

## PHASE 3 — Project Structure

```
sg-plate-reader/
├── plate_model.pt        ← downloaded YOLO weights (after running download_model.py)
├── download_model.py     ← one-shot model download script (run this first)
├── plate_reader.py       ← core detection + OCR + Singapore validation
├── server.py             ← FastAPI REST endpoint
├── test_reader.py        ← local test without server
└── requirements.txt
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

### Test locally (no server)
```bash
python test_reader.py plate_model.pt test_car.jpg
```

### Interactive docs
Open `http://localhost:8000/docs` — upload images directly in the browser.

---

## PHASE 5 — Integrate with Your Gantry System

```python
import requests

def detect_sg_plate(image_path: str) -> str | None:
    """Call the plate reader API and return the Singapore plate string."""
    with open(image_path, "rb") as f:
        resp = requests.post(
            "http://plate-reader-host:8000/read-plate",
            files={"file": f},
            timeout=10,
        )
    data = resp.json()
    if data["success"] and data["plates"]:
        return data["plates"][0]["text"]   # e.g. "SBA1234A"
    return None
```

---

## Singapore Gantry Tips

| Topic             | Recommendation                                                                        |
|-------------------|---------------------------------------------------------------------------------------|
| Camera angle      | 1–2 m height, slight downward angle toward the rear plate                            |
| Colour scheme     | System auto-detects; no config needed                                                 |
| Frame selection   | Don't OCR every frame — pick the sharpest frame when the vehicle is stationary        |
| Multiple reads    | Read 3–5 frames and take the majority-vote result for reliability                     |
| Confidence        | Start with `DETECT_CONF=0.5`; lower to `0.35` if plates are missed                   |
| Night / rain      | IR illuminator recommended — SG plates are retro-reflective (LTA requirement)         |
| Off-peak cars     | Red plate, same alphanumeric format — detected correctly                              |
| EV/PHEV plates    | LTA announced green plates (2026) — same format, no code changes needed               |

---

## Troubleshooting

| Problem                        | Fix                                                                         |
|--------------------------------|-----------------------------------------------------------------------------|
| No plates detected             | Lower `DETECT_CONF` to 0.35; check image is at least 640 px wide           |
| Valid plate rejected           | Check the plate uses standard LTA format; off-peak/special plates may vary  |
| OCR reads wrong characters     | Ensure plate is well-lit and not at extreme angle (>30° horizontal skew)    |
| White-on-black plate garbled   | Auto-inversion is in `_normalise_colour_scheme`; check mean brightness      |
| `ModuleNotFoundError`          | Ensure venv is activated: `source venv/bin/activate`                        |
| Slow on CPU                    | YOLOv8n is already the fastest; set `DETECT_CONF=0.6` to reduce candidates  |
| Model not found (503)          | Set `MODEL_PATH` env var or place `plate_model.pt` in the project folder    |