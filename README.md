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

**Sanitune** is an AI-powered tool for creating clean versions of songs. It separates vocals from instrumentals, transcribes the vocal track, detects explicit words, and then mutes, bleeps, or replaces those words before remixing the song.

Key features:

- **Three edit modes**: `mute`, `bleep`, and `replace` (voice replacement with pitch/timbre matching)
- **Web UI**: Gradio-based interface with drag-and-drop upload, transcript highlighting, and audio preview
- **AI-powered suggestions**: BYO API key (Anthropic/OpenAI) for context-aware replacement word selection
- **Voice synthesis**: Edge-TTS (speech) or Bark (singing) for replacement word generation
- **Cloud voice conversion**: Optional Kits.ai integration for singer voice matching
- **Format preservation**: Output matches input format (MP3, FLAC, WAV, OGG, M4A, etc.)
- **CLI + Docker**: Local processing, audio never leaves your machine (except optional AI/cloud features)

## How It Works

```text
Upload Song → Separate Vocals & Instrumentals → Transcribe Lyrics
    → Detect Profanity → [AI Suggestions] → Mute/Bleep/Replace → Remix → Clean Song
```

1. **Source separation** — [Demucs v4](https://github.com/adefossez/demucs) (Meta) isolates vocals from the instrumental track
2. **Transcription** — [WhisperX](https://github.com/m-bain/whisperX) transcribes lyrics with precise word-level timestamps
3. **Detection** — Configurable profanity word lists flag explicit content (English + Spanish)
4. **AI suggestions** (optional) — LLM picks context-aware clean replacements that match rhyme, syllable count, and tone
5. **Processing** — Flagged words are muted, bleeped, or replaced with synthesized clean alternatives
6. **Remix** — Processed vocals are merged back with the untouched instrumental

## Quick Start

### CLI

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[lyrics,web,ai]"

# Basic mute flow
sanitune process song.mp3 --mode mute --language en

# Bleep explicit words instead
sanitune process song.mp3 --mode bleep --language es --bleep-freq 1200

# Replace with voice synthesis
sanitune process song.mp3 --mode replace --synth-engine bark -l es

# AI-powered contextual replacements (BYO API key)
sanitune process song.mp3 --mode replace \
  --ai-provider anthropic --ai-api-key sk-ant-...

# Override device or output location
sanitune process song.mp3 --device cuda --output song_clean.wav
```

### Web UI

```bash
pip install -e ".[web]"
sanitune web --port 7860
```

Then open http://localhost:7860 in your browser.

### Optional Lyrics Lookup

```bash
pip install -e ".[lyrics]"

sanitune process song.mp3 \
  --artist "Artist Name" \
  --title "Song Title"
```

When `--artist` and `--title` are provided, Sanitune may query external lyrics providers to cross-reference the transcription. Only song metadata and optional provider API tokens are sent; audio stays local.

### Docker

```bash
# CLI mode
docker compose run --rm sanitune process input/song.mp3 -o output/song_clean.wav

# Web UI mode (uncomment sanitune-web service in docker-compose.yml)
docker compose up sanitune-web
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SANITUNE_DEVICE` | `auto` | Processing device: `auto`, `cpu`, `cuda`, `mps` |
| `SANITUNE_LANGUAGE` | `en` | Profanity detection language: `en`, `es` |
| `SANITUNE_DEFAULT_MODE` | `mute` | Default cleaning mode: `mute`, `bleep` |
| `SANITUNE_MAX_FILE_SIZE` | `200` | Maximum upload file size in MB |
| `SANITUNE_BLEEP_FREQ` | `1000` | Bleep tone frequency in Hz |
| `SANITUNE_AI_API_KEY` | | API key for AI replacement suggestions (Anthropic/OpenAI) |
| `KITS_API_KEY` | | Kits.ai API key for cloud voice conversion |
| `KITS_VOICE_MODEL_ID` | | Kits.ai voice model ID |

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

- Voice replacement quality is experimental — TTS-based synthesis doesn't perfectly match singing voices yet
- Bark singing output varies in quality depending on language and speaker preset
- Kits.ai voice conversion requires a paid account and has a 1 request/minute rate limit
- AI suggestions require a BYO API key (Anthropic or OpenAI)
- GPU recommended for faster processing (CPU works but is slower)

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan.

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

## License

[GPL-3.0](LICENSE)
