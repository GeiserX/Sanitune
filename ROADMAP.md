# Sanitune Roadmap

## Phase 1: Core Engine (v0.1.0)

The foundation — audio processing pipeline without UI.

- [ ] **Project scaffolding**: Python package structure, pyproject.toml, CLI entry point
- [ ] **Audio source separation**: Demucs v4 (`htdemucs_ft`) integration — split any song into vocals + instrumentals
- [ ] **Transcription with word timestamps**: WhisperX integration — word-level aligned transcription of isolated vocals
- [ ] **Lyrics reference alignment**: Optional Genius API / LRCLIB integration — align Whisper output against known lyrics to correct transcription errors (especially for heavily produced tracks)
- [ ] **Profanity detection engine**: Configurable word lists for English and Spanish, with exact/partial/fuzzy matching modes
- [ ] **Mute mode**: Silence flagged words in the vocal track only, with smooth 10-20ms crossfades at edit boundaries. Instrumentals play through untouched
- [ ] **Bleep mode**: Replace flagged words with a configurable tone (frequency, duration) overlaid on the vocal track
- [ ] **Audio remix**: Merge processed vocals back with original instrumentals, preserving original quality
- [ ] **Format handling**: Input/output via ffmpeg — support MP3, WAV, FLAC, AAC, OGG, AIFF, M4A
- [ ] **Hardware auto-detection**: PyTorch runtime detection of CUDA, MPS (Apple Silicon), or CPU fallback. User-configurable override via env var or CLI flag
- [ ] **CLI interface**: `sanitune process <file> [--mode mute|bleep] [--language en|es] [--device cpu|cuda|mps]`

## Phase 2: Voice Replacement (v0.2.0)

The differentiator — replace words with the singer's own voice.

- [ ] **RVC v2 integration**: Voice conversion pipeline via Applio/RVC for singing voice cloning
- [ ] **Auto voice model training**: Extract clean vocal segments from the separated track (~10 min), auto-train a per-song RVC model
- [ ] **Replacement word mapping**: Built-in profanity-to-clean-word dictionary (EN + ES). E.g., fuck -> heck, shit -> crap, damn -> darn, mierda -> rayos, joder -> jolines
- [ ] **Voice replacement pipeline**: Generate replacement word audio via TTS/recording, convert to singer's voice via RVC, splice into vocal track with crossfade
- [ ] **Pitch and timing matching**: Match the replacement word's pitch contour and duration to the original word's melody position
- [ ] **Spectral smoothing**: Post-process splice boundaries with VoiceFixer or similar to reduce artifacts
- [ ] **User word override**: Allow users to type custom replacement words (must be real dictionary words)
- [ ] **CLI extension**: `sanitune process <file> --mode replace [--mapping custom.json]`

## Phase 3: Web UI (v0.3.0)

Gradio-based web interface for non-technical users.

- [ ] **Upload interface**: Drag-and-drop or file picker for audio files, with format validation and size limits
- [ ] **Processing status**: Real-time progress bar showing pipeline stages (separating, transcribing, detecting, processing, remixing)
- [ ] **Interactive transcript**: Display transcribed lyrics with flagged words highlighted. Click to toggle words on/off
- [ ] **Per-word action selector**: Dropdown per flagged word — mute, bleep, or voice replace. Visual preview of replacement word
- [ ] **Audio preview**: In-browser playback of original vs. clean version, with A/B comparison toggle
- [ ] **Download**: One-click download of the clean version in the original format
- [ ] **Settings panel**: Language selection, device selection, model management, custom word list editor
- [ ] **Processing history**: Session-based history of processed songs (not persisted across restarts)

## Phase 4: Docker & Self-Hosting (v0.4.0)

Production-ready containerized deployment.

- [ ] **Dockerfile**: Multi-stage build — slim base with Python, ffmpeg, and all model dependencies. Single container
- [ ] **Model management**: Auto-download models on first run (Demucs ~300MB, Whisper ~1.5GB, RVC ~200MB). Cache in a volume
- [ ] **GPU passthrough**: Docker Compose examples for NVIDIA (nvidia-container-toolkit), Intel iGPU (/dev/dri), and CPU-only
- [ ] **Environment configuration**: All settings via environment variables (SANITUNE_DEVICE, SANITUNE_LANGUAGE, SANITUNE_MODEL_DIR, etc.)
- [ ] **Health check endpoint**: `/health` for container orchestrators
- [ ] **Resource limits**: Configurable max file size, max processing time, temp directory cleanup
- [ ] **Docker Hub publishing**: `drumsergio/sanitune` with versioned tags (never `latest`)
- [ ] **Unraid Community App template**: XML template in GeiserX/docker-templates

## Phase 5: AI Contextual Replacement (v0.5.0)

LLM-powered smart word selection.

- [ ] **BYO API key support**: User provides their own LLM API key (OpenAI, Anthropic, or compatible). Key used only for current session, never stored
- [ ] **Context-aware suggestions**: Send surrounding lyrics + flagged word to LLM, receive contextually appropriate clean replacement
- [ ] **Replacement preview**: Show AI-suggested word alongside default mapping. User picks which to use
- [ ] **Fallback chain**: AI suggestion -> built-in mapping -> user manual input -> mute (if all else fails)
- [ ] **Prompt engineering**: Tuned prompt that respects rhyme scheme, syllable count, and emotional tone of the original lyric
- [ ] **Rate limiting**: Cap LLM calls per song to prevent abuse / cost overruns

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
