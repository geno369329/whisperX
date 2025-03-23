FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Prevent interactive prompts (like tzdata asking for region)
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
  tzdata \
  ffmpeg \
  libsndfile1 \
  git && \
  rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy app files into container
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Start WhisperX
CMD ["python3", "-m", "whisperx"]
