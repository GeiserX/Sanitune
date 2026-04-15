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
) -> tuple[np.ndarray, int]:
    """Generate speech audio for a word using edge-tts.

    Args:
        text: The word/phrase to synthesize.
        language: Language code for voice selection.
        voice: Override TTS voice name. If None, uses DEFAULT_VOICES for the language.
        sample_rate: Target sample rate for the output.
        use_cache: Whether to cache synthesized audio on disk.

    Returns:
        Tuple of (audio_array, sample_rate). Audio is mono float32.

    Raises:
        ImportError: If edge-tts is not installed.
        RuntimeError: If TTS synthesis fails.
    """
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

    # Cache result
    if use_cache:
        sf.write(str(cache_path), data, sr)
        logger.debug("TTS cached '%s' → %s", text, cache_path.name)

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
