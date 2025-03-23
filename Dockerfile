FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TZ=America/Chicago

# Install system dependencies including python3-venv and pipx
RUN apt-get update && apt-get install -y \
  tzdata \
  ffmpeg \
  libsndfile1 \
  git \
  curl \
  python3-venv \
  pipx \
  && rm -rf /var/lib/apt/lists/*

# Install uv globally using pipx
RUN pipx install uv

WORKDIR /app

COPY . /app

# Use full path to uv since pipx doesn’t add it to PATH in this context
RUN /root/.local/bin/uv pip install --upgrade pip
RUN /root/.local/bin/uv sync --no-dev

CMD ["python3", "-m", "whisperx"]
