FROM python:3.11-slim

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    ultralytics \
    transformers \
    accelerate \
    psycopg2-binary \
    boto3 \
    python-dotenv \
    pycocotools \
    nuscenes-devkit \
    numpy \
    pillow \
    fastapi \
    uvicorn

RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/sam2.git"

WORKDIR /opt/airflow