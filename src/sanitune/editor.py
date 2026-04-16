"""Audio editing — mute, bleep, or replace flagged words in the vocal track."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sanitune.detector import FlaggedWord

logger = logging.getLogger(__name__)


def _generate_tone(duration_seconds: float, sample_rate: int, frequency: int = 1000) -> np.ndarray:
    """Generate a sine wave tone."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * frequency * t)
    return tone.astype(np.float32)


def _apply_crossfade(audio: np.ndarray, start: int, end: int, fade_ms: int = 10, sample_rate: int = 44100) -> None:
    """Apply a short crossfade at the boundaries of an edit region (in-place)."""
    fade_samples = min(int(sample_rate * fade_ms / 1000), (end - start) // 2)
    if fade_samples < 2:
        return

    # Fade out at start boundary
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    region_start = max(0, start - fade_samples)
    actual_fade = start - region_start
    if actual_fade > 0:
        if audio.ndim == 2:
            audio[region_start:start] *= fade_out[-actual_fade:, np.newaxis]
        else:
            audio[region_start:start] *= fade_out[-actual_fade:]

    # Fade in at end boundary
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    region_end = min(len(audio), end + fade_samples)
    actual_fade = region_end - end
    if actual_fade > 0:
        if audio.ndim == 2:
            audio[end:region_end] *= fade_in[:actual_fade, np.newaxis]
        else:
            audio[end:region_end] *= fade_in[:actual_fade]


def edit(
    vocals: np.ndarray,
    sample_rate: int,
    flagged: list[FlaggedWord],
    *,
    mode: str = "mute",
    bleep_freq: int = 1000,
    margin_ms: int = 50,
    language: str = "en",
    custom_mapping_path: Path | None = None,
    tts_voice: str | None = None,
    device: str = "cpu",
    synth_engine: str = "edge-tts",
    kits_api_key: str | None = None,
    kits_voice_model_id: int | None = None,
    ai_suggestions: dict[str, str] | None = None,
) -> np.ndarray:
    """Edit vocal track by muting, bleeping, or replacing flagged words.

    Args:
        vocals: Vocal audio array (samples,) or (samples, channels).
        sample_rate: Sample rate of the audio.
        flagged: List of flagged words with timestamps.
        mode: 'mute' to silence, 'bleep' to overlay a tone, 'replace' for voice replacement.
        bleep_freq: Frequency of the bleep tone in Hz.
        margin_ms: Extra margin in ms around each word for cleaner edits.
        language: Language code (used by replace mode for TTS and mapping selection).
        custom_mapping_path: Path to custom replacement mapping JSON (replace mode).
        tts_voice: Override TTS voice name (replace mode).

    Returns:
        Edited vocal audio array.
    """
    valid_modes = {"mute", "bleep", "replace"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown edit mode '{mode}'. Valid modes: {valid_modes}")

    if not flagged:
        logger.info("No flagged words to edit")
        return vocals.copy()

    # Delegate to replacer for replace mode
    if mode == "replace":
        from sanitune.replacer import replace_words

        result, replaced, muted = replace_words(
            vocals, sample_rate, flagged,
            language=language,
            margin_ms=margin_ms,
            custom_mapping_path=custom_mapping_path,
            tts_voice=tts_voice,
            device=device,
            synth_engine=synth_engine,
            kits_api_key=kits_api_key,
            kits_voice_model_id=kits_voice_model_id,
            ai_suggestions=ai_suggestions,
        )
        logger.info(
            "Edited %d words using 'replace' mode (%d replaced, %d muted fallback)",
            len(flagged), replaced, muted,
        )
        return result

    result = vocals.copy()
    margin_samples = int(sample_rate * margin_ms / 1000)
    total_samples = len(result)

    for fw in flagged:
        start = max(0, int(fw.word.start * sample_rate) - margin_samples)
        end = min(total_samples, int(fw.word.end * sample_rate) + margin_samples)

        if start >= end:
            continue

        if mode == "mute":
            result[start:end] = 0.0
        elif mode == "bleep":
            duration = (end - start) / sample_rate
            tone = _generate_tone(duration, sample_rate, bleep_freq)
            # Match the shape if multi-channel
            if result.ndim == 2:
                channels = result.shape[1]
                tone = np.tile(tone[:, np.newaxis], (1, channels))
            result[start:end] = tone[: end - start]

        _apply_crossfade(result, start, end, sample_rate=sample_rate)
        logger.debug("Edited '%s' (%s) at %.2f-%.2f", fw.word.text, mode, fw.word.start, fw.word.end)

    logger.info("Edited %d words using '%s' mode", len(flagged), mode)
    return result
