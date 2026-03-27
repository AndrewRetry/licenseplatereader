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

# EasyOCR downloads models on first run — pre-download them
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

EXPOSE 8000

CMD ["python", "server.py"]