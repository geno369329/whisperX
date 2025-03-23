# Use a CUDA-compatible PyTorch base image for GPU support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# System-level dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsndfile1 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Hugging Face model caching
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Create working directory
WORKDIR /app

# Copy local files to container
COPY . /app

# Install pip and Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Explicitly install extra dependencies needed by WhisperX
RUN pip install flask transformers nltk librosa pandas torchaudio

# Download NLTK tokenizers (needed for diarization)
RUN python3 -m nltk.downloader punkt

# Expose Flask default port
EXPOSE 5000

# Run Flask server
CMD ["python3", "app.py"]
