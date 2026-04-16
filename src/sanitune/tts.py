"""Text-to-speech generation for replacement words."""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Default voices per language — Microsoft Edge Neural voices
DEFAULT_VOICES: dict[str, str] = {
    "en": "en-US-GuyNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "it": "it-IT-DiegoNeural",
    "pt": "pt-BR-AntonioNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "zh": "zh-CN-YunxiNeural",
}

_cache_dir: Path | None = None


def _get_cache_dir() -> Path:
    global _cache_dir
    if _cache_dir is None:
        _cache_dir = Path(tempfile.gettempdir()) / "sanitune_tts_cache"
        _cache_dir.mkdir(exist_ok=True)
    return _cache_dir


def synthesize(
    text: str,
    language: str = "en",
    *,
    voice: str | None = None,
    sample_rate: int = 44100,
    use_cache: bool = True,
    engine: str = "edge-tts",
    singing: bool = False,
) -> tuple[np.ndarray, int]:
    """Generate speech or singing audio for a word.

    Args:
        text: The word/phrase to synthesize.
        language: Language code for voice selection.
        voice: Override TTS voice name. If None, uses defaults for the engine/language.
        sample_rate: Target sample rate for the output.
        use_cache: Whether to cache synthesized audio on disk.
        engine: Synthesis engine — 'edge-tts' (speech) or 'bark' (can sing).
        singing: If True and engine supports it, generate singing output.

    Returns:
        Tuple of (audio_array, sample_rate). Audio is mono float32.

    Raises:
        ImportError: If required packages are not installed.
        RuntimeError: If synthesis fails.
    """
    if engine == "bark":
        return _synthesize_bark(
            text, language, voice=voice, sample_rate=sample_rate,
            use_cache=use_cache, singing=singing,
        )

    return _synthesize_edge_tts(
        text, language, voice=voice, sample_rate=sample_rate, use_cache=use_cache,
    )


# Bark speaker presets per language
BARK_SPEAKERS: dict[str, str] = {
    "en": "v2/en_speaker_6",
    "es": "v2/es_speaker_0",
    "fr": "v2/fr_speaker_1",
    "de": "v2/de_speaker_3",
    "it": "v2/it_speaker_4",
    "pt": "v2/pt_speaker_4",
    "ja": "v2/ja_speaker_4",
    "ko": "v2/ko_speaker_4",
    "zh": "v2/zh_speaker_4",
}


def _synthesize_bark(
    text: str,
    language: str = "en",
    *,
    voice: str | None = None,
    sample_rate: int = 44100,
    use_cache: bool = True,
    singing: bool = False,
) -> tuple[np.ndarray, int]:
    """Generate audio using Bark (Suno). Supports singing via musical markers."""
    # Check cache
    cache_key_src = f"bark|{text}|{language}|{voice}|{singing}|{sample_rate}"
    if use_cache:
        cache_key = hashlib.sha256(cache_key_src.encode()).hexdigest()[:16]
        cache_path = _get_cache_dir() / f"bark_{cache_key}.wav"
        if cache_path.exists():
            try:
                data, sr = sf.read(str(cache_path), dtype="float32")
                logger.debug("Bark cache hit for '%s'", text)
                return data, sr
            except (OSError, RuntimeError) as exc:
                logger.warning("Ignoring bad Bark cache entry %s: %s", cache_path, exc)
                cache_path.unlink(missing_ok=True)

    try:
        # Bark uses torch.load internally — PyTorch 2.6+ defaults weights_only=True
        # which fails on Bark's numpy-containing checkpoints. Monkey-patch it.
        import torch
        _original_torch_load = torch.load
        torch.load = lambda *a, **kw: _original_torch_load(
            *a, **{**kw, "weights_only": kw.get("weights_only", False)},
        )

        import os
        os.environ.setdefault("SUNO_USE_SMALL_MODELS", "1")

        from bark import SAMPLE_RATE as BARK_SR
        from bark import generate_audio
        from bark.generation import preload_models
    except ImportError:
        raise ImportError(
            "Bark is required for the bark synthesis engine. "
            "Install with: pip install suno-bark"
        )

    preload_models()

    speaker = voice or BARK_SPEAKERS.get(language, BARK_SPEAKERS["en"])

    # Wrap text with singing markers if requested
    prompt = f"\u266a {text} \u266a" if singing else text

    logger.debug("Bark generating '%s' (speaker=%s, singing=%s)", prompt, speaker, singing)

    audio = generate_audio(prompt, history_prompt=speaker)

    # Bark outputs at BARK_SR (24000 Hz) — resample if needed
    data = audio.astype(np.float32)
    data = _trim_silence(data, threshold_db=-40)

    if BARK_SR != sample_rate:
        import librosa
        data = librosa.resample(data, orig_sr=BARK_SR, target_sr=sample_rate).astype(np.float32)
        sr = sample_rate
    else:
        sr = BARK_SR

    # Cache result
    if use_cache:
        try:
            cache_key = hashlib.sha256(cache_key_src.encode()).hexdigest()[:16]
            cache_path = _get_cache_dir() / f"bark_{cache_key}.wav"
            sf.write(str(cache_path), data, sr)
            logger.debug("Bark cached '%s' → %s", text, cache_path.name)
        except (OSError, RuntimeError) as exc:
            logger.warning("Failed to cache Bark audio for '%s': %s", text, exc)

    return data, sr


def _synthesize_edge_tts(
    text: str,
    language: str = "en",
    *,
    voice: str | None = None,
    sample_rate: int = 44100,
    use_cache: bool = True,
) -> tuple[np.ndarray, int]:
    """Generate speech audio using edge-tts (Microsoft Neural voices)."""
    try:
        import edge_tts
    except ImportError:
        raise ImportError(
            "edge-tts is required for voice replacement mode. "
            "Install with: pip install sanitune[replace]"
        )

    resolved_voice = voice or DEFAULT_VOICES.get(language, DEFAULT_VOICES["en"])

    # Check cache
    if use_cache:
        cache_key = hashlib.sha256(f"{text}|{resolved_voice}|{sample_rate}".encode()).hexdigest()[:16]
        cache_path = _get_cache_dir() / f"{cache_key}.wav"
        if cache_path.exists():
            try:
                data, sr = sf.read(str(cache_path), dtype="float32")
                logger.debug("TTS cache hit for '%s'", text)
                return data, sr
            except (OSError, RuntimeError) as exc:
                logger.warning("Ignoring bad TTS cache entry %s: %s", cache_path, exc)
                cache_path.unlink(missing_ok=True)

    # Generate via edge-tts (async API)
    async def _generate() -> Path:
        communicate = edge_tts.Communicate(text, resolved_voice)
        tmp_path = _get_cache_dir() / f"tmp_{hashlib.sha256(text.encode()).hexdigest()[:12]}.mp3"
        await communicate.save(str(tmp_path))
        return tmp_path

    try:
        mp3_path = asyncio.run(_generate())
    except Exception as exc:
        raise RuntimeError(f"TTS synthesis failed for '{text}': {exc}") from exc

    # Convert MP3 to WAV via ffmpeg
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-i", str(mp3_path),
                "-f", "wav", "-acodec", "pcm_f32le",
                "-ar", str(sample_rate), "-ac", "1",
                "-",
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(f"TTS audio conversion failed for '{text}': {exc}") from exc
    finally:
        mp3_path.unlink(missing_ok=True)

    # Trim leading/trailing silence (below -40 dB)
    data = _trim_silence(data, threshold_db=-40)

    # Cache result (best-effort — never fail synthesis for a cache write error)
    if use_cache:
        try:
            sf.write(str(cache_path), data, sr)
            logger.debug("TTS cached '%s' → %s", text, cache_path.name)
        except (OSError, RuntimeError) as exc:
            logger.warning("Failed to cache TTS audio for '%s': %s", text, exc)

    return data, sr


def _trim_silence(audio: np.ndarray, threshold_db: float = -40) -> np.ndarray:
    """Trim leading and trailing silence from audio."""
    if len(audio) == 0:
        return audio

    threshold = 10 ** (threshold_db / 20)
    abs_audio = np.abs(audio)

    # Find first sample above threshold
    above = np.where(abs_audio > threshold)[0]
    if len(above) == 0:
        return audio

    start = max(0, above[0] - 100)  # Keep ~2ms lead-in at 44100
    end = min(len(audio), above[-1] + 100)
    return audio[start:end]
