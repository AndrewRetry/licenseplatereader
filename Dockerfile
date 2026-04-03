# Dockerfile — Gantry License Plate Reader Microservice
#
# Build:  docker build -t gantry-lpr .
# Run:    docker run -p 8000:8000 \
#           -v ./plate_model.pt:/app/plate_model.pt \
#           -e AMQP_URL=amqp://guest:guest@rabbitmq:5672/ \
#           -e STREAM_URL=http://host.docker.internal:8080/video \
#           gantry-lpr

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libglib2.0-0 libgl1 libsm6 libice6 libxext6 \
       libxrender1 libxcb1 libx11-6 libxau6 libxdmcp6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "\
from transformers import TrOCRProcessor, VisionEncoderDecoderModel; \
TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed'); \
VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed'); \
print('TrOCR cached.')"

COPY plate_reader.py server.py \
     event_publisher.py stream_processor.py \
     ./

EXPOSE 8000

CMD ["python", "server.py"]