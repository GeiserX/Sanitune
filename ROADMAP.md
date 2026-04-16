# Sanitune Roadmap

## Phase 1: Core Engine (v0.1.0)

The foundation — audio processing pipeline without UI.

- [x] **Project scaffolding**: Python package structure, pyproject.toml, CLI entry point
- [x] **Audio source separation**: Demucs v4 (`htdemucs_ft`) integration — split any song into vocals + instrumentals
- [x] **Transcription with word timestamps**: WhisperX integration — word-level aligned transcription of isolated vocals
- [x] **Lyrics reference alignment**: Optional Genius API / LRCLIB integration — align Whisper output against known lyrics to correct transcription errors (especially for heavily produced tracks)
- [x] **Profanity detection engine**: Configurable word lists for English and Spanish, with exact/partial/fuzzy matching modes
- [x] **Mute mode**: Silence flagged words in the vocal track only, with smooth 10-20ms crossfades at edit boundaries. Instrumentals play through untouched
- [x] **Bleep mode**: Replace flagged words with a configurable tone (frequency, duration) overlaid on the vocal track
- [x] **Audio remix**: Merge processed vocals back with original instrumentals, preserving original quality
- [x] **Format handling**: Input/output via ffmpeg — support MP3, WAV, FLAC, AAC, OGG, AIFF, M4A
- [x] **Hardware auto-detection**: PyTorch runtime detection of CUDA, MPS (Apple Silicon), or CPU fallback. User-configurable override via env var or CLI flag
- [x] **CLI**: `sanitune process <file> [--mode mute|bleep] [--language en|es] [--device cpu|cuda|mps]`

## Phase 2: Voice Replacement (v0.2.0)

The differentiator — replace words with natural-sounding alternatives.

- [x] **Replacement word mapping**: Built-in profanity-to-clean-word dictionaries (EN: 43 entries, ES: 47 entries). E.g., fuck → heck, shit → crap, picha → cosa, joder → jolines
- [x] **TTS-based voice replacement**: edge-tts (Microsoft Edge Neural voices) generates replacement words with disk caching for fast re-runs
- [x] **Pitch and timing matching**: librosa pyin F0 extraction + pitch shifting to match singer's pitch; time-stretch to match original word duration
- [x] **Loudness matching**: RMS-based level alignment between replacement and original audio
- [x] **Spectral smoothing**: Hann-windowed overlap-add at splice boundaries for seamless transitions
- [x] **Surgical editing**: Only modify flagged word regions — rest of song preserved byte-for-byte from original. Crossfade at region boundaries (configurable margin + fade)
- [x] **Format preservation**: Output matches input format (MP3→MP3, FLAC→FLAC) with same bitrate/sample rate via ffprobe detection + ffmpeg encoding. Supports WAV, MP3, FLAC, OGG, M4A, AAC, OPUS
- [x] **Mute fallback**: Words without a mapping entry are silenced as fallback (no crash on unmapped words)
- [x] **Multi-language TTS voices**: Default neural voices for EN, ES, FR, DE, IT, PT, JA, KO, ZH
- [x] **User word override**: Custom mapping JSON via `--mapping custom.json` overrides/extends built-in dictionaries
- [x] **CLI extension**: `sanitune process <file> --mode replace [--mapping custom.json] [--tts-voice voice-name]`

### Deferred to future phases
- **RVC v2 / singing voice cloning**: Too heavy for CPU Docker (~200MB+ model per song). Current TTS approach is lighter and works well for single-word replacements. May revisit when GPU-first deployment is standard

## Phase 3: Web UI (v0.4.0)

Gradio-based web interface for non-technical users.

- [x] **Upload interface**: Drag-and-drop or file picker for audio files, with format validation and size limits
- [x] **Processing status**: Status message showing pipeline results and timing
- [x] **Interactive transcript**: Display transcribed lyrics with flagged words highlighted in red, with timestamps on hover
- [ ] **Per-word action selector**: Dropdown per flagged word — mute, bleep, or voice replace. Visual preview of replacement word
- [x] **Audio preview**: In-browser playback of clean version
- [x] **Download**: One-click download of the clean version in the original format
- [x] **Settings panel**: Language selection, device selection, synth engine, bleep frequency, word overrides, lyrics lookup, AI suggestions
- [ ] **Processing history**: Session-based history of processed songs (not persisted across restarts)

## Phase 4: Docker & Self-Hosting (v0.4.0)

Production-ready containerized deployment.

- [x] **Dockerfile**: Multi-stage build — slim base with Python, ffmpeg, and all model dependencies. Single container
- [x] **Model management**: Auto-download models on first run (Demucs ~300MB, Whisper ~1.5GB). Cache in a volume via docker-compose
- [x] **GPU passthrough**: Docker Compose examples for NVIDIA (nvidia-container-toolkit), Intel iGPU (/dev/dri), and CPU-only
- [x] **Environment configuration**: All settings via environment variables (SANITUNE_DEVICE, SANITUNE_LANGUAGE, KITS_API_KEY, SANITUNE_AI_API_KEY, etc.)
- [x] **Health check endpoint**: Dockerfile HEALTHCHECK for web UI mode
- [x] **Resource limits**: Configurable max file size via SANITUNE_MAX_FILE_SIZE
- [x] **Docker Hub publishing**: `drumsergio/sanitune` with versioned tags (never `latest`)
- [ ] **Unraid Community App template**: XML template in GeiserX/docker-templates

## Phase 5: AI Contextual Replacement (v0.4.0)

LLM-powered smart word selection.

- [x] **BYO API key support**: User provides their own LLM API key (OpenAI, Anthropic). Key used only for current session, never stored
- [x] **Context-aware suggestions**: Send surrounding lyrics + flagged word to LLM, receive contextually appropriate clean replacement
- [ ] **Replacement preview**: Show AI-suggested word alongside default mapping. User picks which to use
- [x] **Fallback chain**: AI suggestion -> built-in mapping -> mute (if all else fails)
- [x] **Prompt engineering**: Tuned prompt that respects rhyme scheme, syllable count, and emotional tone of the original lyric
- [x] **Rate limiting**: Cap LLM calls per song (max 20) to prevent abuse / cost overruns

## Phase 6: Quality & Polish (v0.6.0)

Refinement pass for production quality.

- [ ] **A/B quality testing framework**: Automated comparison of original vs. clean output. Measure SNR, spectral similarity, and perceptual quality (PESQ/POLQA)
- [ ] **Crossfade tuning**: Adaptive crossfade duration based on vocal context (longer for sustained notes, shorter for percussive consonants)
- [ ] **Reverb matching**: Analyze original recording's reverb profile and apply matching reverb to voice-replaced segments
- [ ] **Multi-occurrence handling**: When the same word appears multiple times, process all occurrences with consistent settings
- [ ] **Edge case handling**: Overlapping vocals, backing vocals, ad-libs, harmonies — detect and warn user when confidence is low
- [ ] **Processing speed optimization**: Model quantization (INT8/FP16), batch inference where possible, memory optimization for CPU-only environments

## Future Considerations (Unscheduled)

These are documented ideas for future evaluation. Not committed to any version.

### Guardrails & Abuse Prevention
- **Time-based edit limit**: Cap total editable duration at 10-15% of song length. Prevents full lyric rewriting while allowing repeated bad words
- **Word count limit**: Max N unique flagged words per song (e.g., 20). Each word's instances are auto-found
- **Single-word restriction**: Only individual words can be targeted, never phrases or full sentences
- **Dictionary enforcement**: Replacement words must exist in a real dictionary. No arbitrary text injection
- **Rate limiting (SaaS)**: Per-user processing quotas for the hosted web app version
- **All-of-the-above layered approach**: Combine time + word count + single-word + dictionary for maximum protection

### Batch Processing
- Upload multiple songs, sequential queue processing
- Per-song settings or apply same profile to all
- Zip download of all clean versions
- Progress tracking per song in queue

### Additional Languages
- Expand profanity detection beyond EN/ES
- Community-contributed word lists with moderation
- Language auto-detection from audio (Whisper already supports this)

### Lyrics Database Integration
- Fetch synced lyrics from LRCLIB, Musixmatch, or Genius
- Pre-populate transcript for faster/more accurate detection
- Fallback to Whisper when lyrics not available

### Advanced Voice Synthesis
- DiffSinger integration for more precise pitch/phoneme control
- Amphion/Vevo2 for zero-shot singing voice editing (research frontier)
- Per-word quality scoring — automatically fall back to mute if voice replacement confidence is below threshold

### Streaming Filter Mode
- Real-time playback filter (like Verso Music) for personal use
- Integration with music players or streaming services
- Low-latency processing pipeline

### Mobile & Desktop
- Progressive Web App (PWA) for mobile access
- Tauri-based desktop app if demand exists

### API & Integrations
- REST API for label-scale batch processing
- Integration with music distribution platforms (DistroKid, TuneCore, CD Baby)
- Webhook notifications on processing completion
- MCP server for AI agent integration

### Monitoring & Analytics (SaaS)
- Processing time metrics and queue depth
- Popular flagged words analytics (anonymized)
- Error rate tracking per pipeline stage
- User feedback on output quality
