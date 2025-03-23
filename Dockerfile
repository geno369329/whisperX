# Use NVIDIA PyTorch base image with CUDA support for GPU
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    libsndfile1 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files into container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    torchaudio \
    librosa \
    pandas \
    transformers \
    nltk \
    pyannote.audio \
    faster-whisper \
    git+https://github.com/m-bain/whisperx.git

# Pre-download punkt tokenizer so nltk works at runtime
RUN python3 -c "import nltk; nltk.download('punkt')"

# Run WhisperX when container starts
CMD ["python3", "-m", "whisperx"]
