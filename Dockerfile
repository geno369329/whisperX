FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  tzdata \
  ffmpeg \
  libsndfile1 \
  git \
  curl && \
  rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -Ls https://astral.sh/uv/install.sh | bash

WORKDIR /app

COPY . /app

# Install Python deps with uv
RUN ~/.cargo/bin/uv pip install --upgrade pip
RUN ~/.cargo/bin/uv sync --no-dev

CMD ["python3", "-m", "whisperx"]
