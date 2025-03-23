# Use NVIDIA PyTorch base image with CUDA support for GPU
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    libsndfile1 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install pipx and uv globally
RUN pip install --no-cache-dir pipx && \
    pipx install uv

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Use full path to uv since pipx doesn’t add it to PATH in this context
RUN /root/.local/bin/uv pip install --upgrade pip --system
RUN /root/.local/bin/uv sync --no-dev

# Start WhisperX
CMD ["python3", "-m", "whisperx"]
