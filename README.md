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

**Sanitune** is a Phase 1 local CLI for creating clean versions of songs. It separates vocals from instrumentals, transcribes the vocal track, detects explicit words, and then mutes or bleeps those words before remixing the song.

The current release is intentionally narrow:

- `sanitune process` is the only shipped command
- `mute` and `bleep` are the only supported edit modes
- output is written as `.wav`
- optional lyrics lookup can improve detection heuristics, but it is not required

Voice replacement, a web UI, and richer self-hosting flows are planned in later phases and are tracked in `ROADMAP.md`.

## How Phase 1 Works

```
Upload Song → Separate Vocals & Instrumentals → Transcribe Lyrics
    → Detect Profanity → Mute/Bleep Flagged Words → Remix → Write Clean WAV
```

1. **Source separation** — [Demucs v4](https://github.com/adefossez/demucs) (Meta) isolates vocals from the instrumental track
2. **Transcription** — [WhisperX](https://github.com/m-bain/whisperX) transcribes lyrics with precise word-level timestamps
3. **Detection** — Configurable profanity word lists flag explicit content (English + Spanish)
4. **Processing** — Flagged words are muted or replaced with a tone in the vocal track only
5. **Remix** — Processed vocals are merged back with the untouched instrumental

## Features

- **Phase 1 CLI**: `sanitune process <file>`
- **Two edit modes**: `mute` or `bleep`
- **Word-level editing**: WhisperX timestamps let edits target individual words instead of whole regions
- **Built-in profanity lists**: English and Spanish wordlists with custom additions/exclusions
- **Optional lyrics reference**: `pip install -e ".[lyrics]"` enables syncedlyrics/Genius lookup when you pass `--artist` and `--title`
- **Hardware-aware runtime**: automatic selection of CUDA, MPS, or CPU when available
- **Docker-friendly batch workflow**: CPU-only CLI container for one-off processing jobs
- **Local audio processing**: audio never leaves your machine unless you explicitly enable lyrics lookup

## Quick Start

### CLI

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# Basic mute flow
sanitune process song.mp3 --mode mute --language en

# Bleep explicit words instead
sanitune process song.mp3 --mode bleep --language es --bleep-freq 1200

# Override device or output location
sanitune process song.mp3 --device cuda --output song_clean.wav

# Raise the input size limit when needed
sanitune process long-song.flac --max-file-size 500
```

### Optional Lyrics Lookup

```bash
pip install -e ".[lyrics]"

sanitune process song.mp3 \
  --artist "Artist Name" \
  --title "Song Title"
```

When `--artist` and `--title` are provided, Sanitune may query external lyrics providers to cross-reference the transcription. Only song metadata and optional provider API tokens are sent; audio stays local.

### Docker

The current Docker image is a **CPU-only CLI container**, not a web app.

```bash
docker build -t sanitune-local .

# Process a file from the current directory
docker run --rm \
  -v "$PWD:/work" \
  sanitune-local process /work/song.mp3 --output /work/song_clean.wav
```

### Docker Compose

```bash
docker compose run --rm sanitune process input/song.mp3 --output output/song_clean.wav
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SANITUNE_DEVICE` | `auto` | Processing device: `auto`, `cpu`, `cuda`, `mps` |
| `SANITUNE_LANGUAGE` | `en` | Profanity detection language: `en`, `es` |
| `SANITUNE_DEFAULT_MODE` | `mute` | Default cleaning mode: `mute`, `bleep` |
| `SANITUNE_MAX_FILE_SIZE` | `200` | Maximum upload file size in MB |
| `SANITUNE_BLEEP_FREQ` | `1000` | Bleep tone frequency in Hz |

## Hardware Requirements

| Mode | Minimum | Recommended |
|------|---------|-------------|
| Mute/Bleep | 4 GB RAM, any CPU | 8 GB RAM, 4+ cores |

Processing times (per 3-minute song, approximate):

| Hardware | Mute/Bleep |
|----------|-------------|
| CPU (4 cores) | ~3 min |
| NVIDIA RTX 3060 | ~30 sec |
| Apple M2 | ~45 sec |

## Technology

| Component | Library | Purpose |
|-----------|---------|---------|
| Source Separation | [Demucs v4](https://github.com/adefossez/demucs) | Isolate vocals from instrumentals |
| Transcription | [WhisperX](https://github.com/m-bain/whisperX) | Word-level lyrics transcription |
| Optional Lyrics Lookup | [syncedlyrics](https://github.com/arran4/syncedlyrics), [lyricsgenius](https://github.com/johnwmillr/LyricsGenius) | Cross-reference lyrics when explicitly enabled |
| Audio I/O | [soundfile](https://github.com/bastibe/python-soundfile) + [ffmpeg](https://ffmpeg.org/) | Audio reading and writing |

## Current Limitations

- Output is written as `.wav` in Phase 1
- There is no web UI yet
- There is no voice replacement mode yet
- The Docker image is CPU-only for now
- Optional lyrics lookup requires extra dependencies and outbound network access

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan.

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

## License

[GPL-3.0](LICENSE)
