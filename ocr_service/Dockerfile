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

COPY main.py plate_reader.py ./

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
