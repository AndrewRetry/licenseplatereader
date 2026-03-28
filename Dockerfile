# Dockerfile — Lightweight plate reader container
# Build:  docker build -t plate-reader .
# Run:    docker run -p 8000:8000 -v ./plate_model.pt:/app/plate_model.pt plate-reader

FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY plate_reader.py server.py ./

# TrOCR downloads ~400 MB from HuggingFace on first run — bake into image
# so container startup is instant and works fully offline
RUN python -c "\
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel; \
    TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed'); \
    VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')"

EXPOSE 8000

CMD ["python", "server.py"]