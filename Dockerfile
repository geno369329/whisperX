FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
  tzdata \
  ffmpeg \
  libsndfile1 \
  git \
  curl && \
  rm -rf /var/lib/apt/lists/*

# Install uv using the official script
RUN curl -Ls https://astral.sh/uv/install.sh | bash

WORKDIR /app

COPY . /app

# Install Python dependencies with uv
RUN bash -c "source $HOME/.cargo/env && uv pip install --upgrade pip"
RUN bash -c "source $HOME/.cargo/env && uv sync --no-dev"

CMD ["python3", "-m", "whisperx"]
