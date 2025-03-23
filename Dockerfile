FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TZ=America/Chicago

# Install system dependencies including python3-venv
RUN apt-get update && apt-get install -y \
  tzdata \
  ffmpeg \
  libsndfile1 \
  git \
  curl \
  pipx \
  python3-venv \
  && rm -rf /var/lib/apt/lists/*

# Install uv globally using pipx
RUN pipx install uv

WORKDIR /app

COPY . /app

# Install Python dependencies with uv
RUN uv pip install --upgrade pip
RUN uv sync --no-dev

CMD ["python3", "-m", "whisperx"]
