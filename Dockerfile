# syntax=docker/dockerfile:1

# ---- builder stage ----
FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install all dependencies — pip resolves compatible versions from PyPI
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/sanitune && \
    echo '__version__ = "0.5.0"' > src/sanitune/__init__.py && \
    pip install --no-cache-dir "setuptools<80" && \
    pip install --no-cache-dir ".[lyrics,voice,web,ai]"

# Clone Seed-VC for singing voice conversion (GPL-3.0, archived but stable)
RUN git clone --depth 1 https://github.com/Plachtaa/seed-vc.git /opt/seed-vc && \
    sed -i 's/proxies: Optional\[Dict\],/proxies: Optional[Dict] = None,/' /opt/seed-vc/modules/bigvgan/bigvgan.py && \
    sed -i 's/resume_download: bool,/resume_download: bool = False,/' /opt/seed-vc/modules/bigvgan/bigvgan.py

# Swap CUDA torch packages for CPU-only builds (same base versions, much smaller)
# Strip +cuXXX suffix — CPU index uses +cpu for the same base version
# Only swap packages that are actually installed (torchvision may not be present)
RUN TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])") && \
    AUDIO_VER=$(python -c "import torchaudio; print(torchaudio.__version__.split('+')[0])") && \
    PKGS="torch==${TORCH_VER} torchaudio==${AUDIO_VER}" && \
    VISION_VER=$(python -c "import torchvision; print(torchvision.__version__.split('+')[0])" 2>/dev/null) && \
    PKGS="${PKGS} torchvision==${VISION_VER}" || true && \
    pip install --no-cache-dir --force-reinstall --no-deps \
        ${PKGS} \
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
LABEL version="0.5.0"
LABEL license="GPL-3.0-only"
LABEL description="AI-powered song cleaning: separate vocals, detect profanity, and mute, bleep, or replace flagged words with the singer's voice"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy Seed-VC for singing voice conversion
COPY --from=builder /opt/seed-vc /opt/seed-vc
ENV PYTHONPATH="/opt/seed-vc"

WORKDIR /app

RUN mkdir -p input output

# Health check for web UI mode (skipped in CLI mode)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" 2>/dev/null || exit 0

EXPOSE 7860

ENTRYPOINT ["python", "-m", "sanitune"]
