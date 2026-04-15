# syntax=docker/dockerfile:1

# ---- builder stage ----
FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install PyTorch CPU-only first (smaller than full CUDA bundle)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy source and install the package
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir ".[lyrics]"

# ---- runtime stage ----
FROM python:3.12-slim

LABEL maintainer="GeiserX <9169332+GeiserX@users.noreply.github.com>"
LABEL version="0.1.0"
LABEL license="GPL-3.0-only"
LABEL description="AI-powered song cleaning — remove or replace explicit words while preserving the original singer's voice"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1000 sanitune && \
    useradd --uid 1000 --gid sanitune --create-home sanitune

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Create input/output directories owned by non-root user
RUN mkdir -p input output && chown -R sanitune:sanitune /app

USER sanitune

ENTRYPOINT ["python", "-m", "sanitune"]
