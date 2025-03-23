# Use a base image with CUDA and PyTorch
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set timezone non-interactively to avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    python3-venv \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set environment variables for HuggingFace
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1

# Start the Flask app
CMD ["python3", "app.py"]
