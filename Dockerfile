# syntax=docker/dockerfile:1

# ---- builder stage ----
FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install all dependencies — pip resolves compatible versions from PyPI
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/sanitune && \
    echo '__version__ = "0.1.0"' > src/sanitune/__init__.py && \
    pip install --no-cache-dir ".[lyrics]"

# Swap CUDA torch packages for CPU-only builds (same base versions, much smaller)
# Strip +cuXXX suffix — CPU index uses +cpu for the same base version
RUN TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])") && \
    AUDIO_VER=$(python -c "import torchaudio; print(torchaudio.__version__.split('+')[0])") && \
    VISION_VER=$(python -c "import torchvision; print(torchvision.__version__.split('+')[0])") && \
    pip install --no-cache-dir --force-reinstall --no-deps \
        "torch==${TORCH_VER}" "torchaudio==${AUDIO_VER}" "torchvision==${VISION_VER}" \
        --index-url https://download.pytorch.org/whl/cpu

# Remove leftover CUDA packages
RUN rm -rf /usr/local/lib/python3.12/site-packages/nvidia/ \
           /usr/local/lib/python3.12/site-packages/triton/

# Copy actual source and reinstall (deps already cached)
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

# ---- runtime stage ----
FROM python:3.12-slim

LABEL maintainer="GeiserX <9169332+GeiserX@users.noreply.github.com>"
LABEL version="0.1.0"
LABEL license="GPL-3.0-only"
LABEL description="Phase 1 CLI for song cleaning: separate vocals, detect profanity, and mute or bleep flagged words"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

RUN mkdir -p input output

ENTRYPOINT ["python", "-m", "sanitune"]
