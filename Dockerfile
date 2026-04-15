# syntax=docker/dockerfile:1

# ---- builder stage ----
FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install PyTorch CPU-only first (smaller than full CUDA bundle)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Pin torch versions so the next pip install cannot upgrade to CUDA variants
RUN pip freeze | grep -iE "^(torch|torchaudio)==" > /tmp/torch-pin.txt

# Install dependencies separately so source changes don't bust the cache
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/sanitune && \
    echo '__version__ = "0.1.0"' > src/sanitune/__init__.py && \
    pip install --no-cache-dir ".[lyrics]" -c /tmp/torch-pin.txt && \
    # Remove CUDA/GPU packages pulled transitively — not needed for CPU
    rm -rf /usr/local/lib/python3.12/site-packages/nvidia/ \
           /usr/local/lib/python3.12/site-packages/triton/ && \
    # Strip test suites and bytecode caches from all packages
    find /usr/local/lib/python3.12/site-packages \
        \( -type d -name "__pycache__" -o -type d -name "tests" -o -type d -name "test" \) \
        -exec rm -rf {} + 2>/dev/null; true

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
