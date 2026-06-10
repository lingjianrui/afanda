# AFANDA local-stream stack: CUDA renderer + WebRTC streamer.
# Requires NVIDIA Container Toolkit on the host (Docker Desktop + WSL2 on Windows).
#
# Build:
#   docker compose build
#
# Run (after extracting model artifacts — see docker-compose.yml):
#   docker compose up

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/opt/pixi \
    PATH="/opt/pixi/bin:${PATH}" \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Pixi manages both the renderer (CUDA/TRT) and streamer (WebRTC) venvs.
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME="${PIXI_HOME}" bash

WORKDIR /app

COPY pixi.toml pyproject.toml ./
COPY src ./src
COPY scripts ./scripts

# Resolve and install both environments (no pixi.lock — first build needs network).
RUN pixi install -e renderer && pixi install -e streamer

EXPOSE 7860 8000

CMD ["pixi", "run", "-e", "streamer", "python", "scripts/run_local_stream.py"]
