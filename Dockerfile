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

# Install uv globally
RUN curl -Ls https://astral.sh/uv/install.sh | bash && \
    mv ~/.cargo/bin/uv /usr/local/bin/uv

WORKDIR /app

COPY . /app

# Use uv to install dependencies from pyproject.toml
RUN uv pip install --upgrade pip
RUN uv sync --no-dev

CMD ["python3", "-m", "whisperx"]
