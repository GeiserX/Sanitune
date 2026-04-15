<p align="center">
  <img src="docs/images/banner.svg" alt="Sanitune Banner" width="900"/>
</p>

<p align="center">
  <a href="https://github.com/GeiserX/Sanitune/blob/main/LICENSE"><img src="https://img.shields.io/github/license/GeiserX/Sanitune" alt="License"></a>
  <a href="https://github.com/GeiserX/Sanitune/releases"><img src="https://img.shields.io/github/v/release/GeiserX/Sanitune" alt="Release"></a>
  <a href="https://github.com/GeiserX/Sanitune/stargazers"><img src="https://img.shields.io/github/stars/GeiserX/Sanitune" alt="Stars"></a>
  <a href="https://hub.docker.com/r/drumsergio/sanitune"><img src="https://img.shields.io/docker/pulls/drumsergio/sanitune" alt="Docker Pulls"></a>
</p>

---

**Sanitune** creates clean versions of songs by removing or replacing explicit words using AI. Unlike simple audio bleepers, Sanitune separates vocals from instrumentals first — so when a word is removed, the music keeps playing naturally. For the ultimate clean edit, it can even replace bad words with clean alternatives **sung in the original singer's voice**.

## How It Works

```
Upload Song → Separate Vocals & Instrumentals → Transcribe Lyrics
    → Detect Profanity → Choose Action Per Word → Process → Download Clean Version
```

1. **Source separation** — [Demucs v4](https://github.com/adefossez/demucs) (Meta) isolates vocals from the instrumental track
2. **Transcription** — [WhisperX](https://github.com/m-bain/whisperX) transcribes lyrics with precise word-level timestamps
3. **Detection** — Configurable profanity word lists flag explicit content (English + Spanish)
4. **Processing** — Per-word action:
   - **Mute**: Silence the word in vocals only (instrumentals keep playing)
   - **Bleep**: Overlay a tone on the word
   - **Voice Replace**: AI generates the clean word in the singer's voice using [RVC](https://github.com/IAHispano/Applio)
5. **Remix** — Processed vocals are merged back with the untouched instrumental

## Features

- **Three cleaning modes**: Mute (silence), bleep (tone), or voice replacement (AI-generated)
- **Auto-detection**: Built-in profanity lists for English and Spanish
- **Manual selection**: Review transcript and toggle words on/off
- **Voice cloning**: Replace words with clean alternatives in the original singer's voice
- **Smart replacement**: Built-in word mappings, custom overrides, or AI-contextual suggestions (BYO API key)
- **Hardware flexible**: Runs on CPU (slower), NVIDIA GPU, Apple Silicon (MPS), or Intel iGPU
- **Multiple interfaces**: Web UI (Gradio), CLI tool, Docker container
- **Format support**: MP3, WAV, FLAC, AAC, OGG, AIFF, M4A (via ffmpeg)
- **Privacy first**: All processing happens locally. No audio leaves your machine

## Quick Start

### CLI

```bash
pip install sanitune

# Mute explicit words
sanitune process song.mp3 --mode mute --language en

# Replace with singer's voice
sanitune process song.mp3 --mode replace

# Specify GPU device
sanitune process song.mp3 --mode mute --device cuda
```

### Web UI

```bash
sanitune web --port 7860
# Open http://localhost:7860
```

### Docker

```bash
# CPU only
docker run -p 7860:7860 -v ./music:/data drumsergio/sanitune:0.1.0

# NVIDIA GPU
docker run --gpus all -p 7860:7860 -v ./music:/data drumsergio/sanitune:0.1.0

# Intel iGPU
docker run --device=/dev/dri -p 7860:7860 -v ./music:/data drumsergio/sanitune:0.1.0
```

### Docker Compose

```yaml
services:
  sanitune:
    image: drumsergio/sanitune:0.1.0
    ports:
      - "7860:7860"
    volumes:
      - ./music:/data
      - sanitune-models:/app/models
    environment:
      - SANITUNE_DEVICE=auto        # auto, cpu, cuda, mps
      - SANITUNE_LANGUAGE=en        # en, es
      - SANITUNE_DEFAULT_MODE=mute  # mute, bleep, replace
    # Uncomment for NVIDIA GPU:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - capabilities: [gpu]

volumes:
  sanitune-models:
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SANITUNE_DEVICE` | `auto` | Processing device: `auto`, `cpu`, `cuda`, `mps` |
| `SANITUNE_LANGUAGE` | `en` | Profanity detection language: `en`, `es` |
| `SANITUNE_DEFAULT_MODE` | `mute` | Default cleaning mode: `mute`, `bleep`, `replace` |
| `SANITUNE_MODEL_DIR` | `./models` | Directory for AI model storage |
| `SANITUNE_MAX_FILE_SIZE` | `200` | Maximum upload file size in MB |
| `SANITUNE_BLEEP_FREQ` | `1000` | Bleep tone frequency in Hz |
| `SANITUNE_LLM_API_KEY` | — | Optional API key for AI-contextual word suggestions |

## Hardware Requirements

| Mode | Minimum | Recommended |
|------|---------|-------------|
| Mute/Bleep | 4 GB RAM, any CPU | 8 GB RAM, 4+ cores |
| Voice Replace | 8 GB RAM, any CPU | 16 GB RAM, GPU with 4+ GB VRAM |

Processing times (per 3-minute song, approximate):

| Hardware | Mute/Bleep | Voice Replace |
|----------|-----------|---------------|
| CPU (4 cores) | ~3 min | ~10 min |
| NVIDIA RTX 3060 | ~30 sec | ~2 min |
| Apple M2 | ~45 sec | ~3 min |
| Intel iGPU (UHD 770) | ~2 min | ~5 min |

## Technology

| Component | Library | Purpose |
|-----------|---------|---------|
| Source Separation | [Demucs v4](https://github.com/adefossez/demucs) | Isolate vocals from instrumentals |
| Transcription | [WhisperX](https://github.com/m-bain/whisperX) | Word-level lyrics transcription |
| Voice Conversion | [RVC v2 / Applio](https://github.com/IAHispano/Applio) | Clone singer's voice for replacements |
| Web UI | [Gradio](https://github.com/gradio-app/gradio) | Browser-based interface |
| Audio Processing | [pydub](https://github.com/jiaaro/pydub) + [ffmpeg](https://ffmpeg.org/) | Format conversion and manipulation |

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan.

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

## License

[GPL-3.0](LICENSE)
