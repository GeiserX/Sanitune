# syntax=docker/dockerfile:1

# ---- builder stage ----
FROM python:3.13.15-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install all dependencies — pip resolves compatible versions from PyPI
#
# The `voice` extra is deliberately NOT installed here: it cannot be resolved on
# Python 3.13 at all. voice pulls descript-audio-codec -> descript-audiotools,
# which pins protobuf<3.20, and no protobuf below 3.20 ships a cp313 wheel, while
# whisperx pulls protobuf 7.x through onnxruntime/grpcio. pip backtracks looking
# for a way out and dies on a kiwisolver 1.4.5 sdist (its legacy pyproject has no
# project.version, which modern setuptools rejects) — a fatal
# metadata-generation-failed that points at the wrong package entirely. That is
# what broke this image from 2026-08-24 with no change on our side.
#
# Nothing under src/ imports the voice packages, and voice_converter.py already
# raises a clear "install sanitune[voice]" ImportError, so the CLI, the pipeline
# and the web UI all work without it. Add voice back when descript-audiotools
# drops the protobuf<3.20 pin.
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/sanitune && \
    echo '__version__ = "0.5.2"' > src/sanitune/__init__.py && \
    pip install --no-cache-dir "setuptools<80" && \
    pip install --no-cache-dir ".[lyrics,web,ai]"

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
RUN rm -rf /usr/local/lib/python3.13/site-packages/nvidia/ \
           /usr/local/lib/python3.13/site-packages/triton/

# Copy actual source and reinstall (deps already cached)
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

# ---- runtime stage ----
FROM python:3.13.15-slim

LABEL maintainer="GeiserX <9169332+GeiserX@users.noreply.github.com>"
LABEL version="0.5.2"
LABEL license="GPL-3.0-only"
LABEL description="AI-powered song cleaning: separate vocals, detect profanity, and mute, bleep, or replace flagged words with the singer's voice"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
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
