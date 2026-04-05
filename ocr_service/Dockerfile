FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Bake YOLO plate detector weights ──────────────────────────────────────
ARG HF_TOKEN
RUN python - <<'EOF'
from huggingface_hub import hf_hub_download
import os, shutil
path = hf_hub_download(
    repo_id="morsetechlab/yolov11-license-plate-detection",
    filename="best.pt",
    local_dir="/app",
    token=os.environ.get("HF_TOKEN"),
)
shutil.move("/app/best.pt", "/app/plate_model.pt")
print("YOLO model ready at /app/plate_model.pt")
EOF

# ── Pre-warm fast-plate-ocr ONNX model ───────────────────────────────────
# Runs the recogniser once so it downloads + caches the ONNX weights
# into the image layer — no network needed at runtime
RUN python - <<'EOF'
from fast_plate_ocr import LicensePlateRecognizer
import numpy as np
print("Downloading fast-plate-ocr ONNX model …")
ocr = LicensePlateRecognizer("cct-xs-v1-global-model")
# Dummy inference to force full initialisation
dummy = np.zeros((64, 128, 3), dtype=np.uint8)
ocr.run(dummy)
print("fast-plate-ocr ready.")
EOF

COPY main.py plate_reader.py ./

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]