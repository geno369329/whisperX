# Use a base image with CUDA and PyTorch
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    python3-venv \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1

# ✅ Let Railway use the Start Command from its settings, not from here
