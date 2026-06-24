FROM apache/airflow:2.9.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir \
    nuscenes-devkit \
    boto3 \
    psycopg2-binary \
    python-dotenv \
    pycocotools \
    mlflow