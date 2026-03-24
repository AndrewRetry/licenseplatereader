FROM python:3.11-slim

WORKDIR /app

# System deps for opencv-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first (avoids pulling CUDA builds, saves ~2 GB)
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining deps
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Pre-download HuggingFace models at build time so runtime startup is instant
# Models are cached to /root/.cache/huggingface
RUN python -c "\
from transformers import YolosImageProcessor, YolosForObjectDetection, \
                         TrOCRProcessor, VisionEncoderDecoderModel; \
YolosImageProcessor.from_pretrained('nickmuchi/yolos-small-rego-plates-detection'); \
YolosForObjectDetection.from_pretrained('nickmuchi/yolos-small-rego-plates-detection'); \
TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed'); \
VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed'); \
print('Models cached OK')"

COPY . .

ENV PORT=3001
EXPOSE 3001

CMD ["python", "server.py"]
