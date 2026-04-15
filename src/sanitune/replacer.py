"""Voice replacement pipeline — generate replacement audio matching the singer's voice characteristics."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sanitune.detector import FlaggedWord
from sanitune.mappings import get_replacement, load_mapping

logger = logging.getLogger(__name__)


def _extract_median_f0(audio: np.ndarray, sr: int) -> float:
    """Extract median fundamental frequency from audio using pyin."""
    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        audio.astype(np.float64),
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
    if len(voiced_f0) == 0:
        return 0.0
    return float(np.median(voiced_f0))


def _time_stretch(audio: np.ndarray, target_duration: float, sr: int) -> np.ndarray:
    """Time-stretch audio to match a target duration."""
    import librosa

    current_duration = len(audio) / sr
    if current_duration <= 0 or target_duration <= 0:
        return audio

    rate = current_duration / target_duration
    # Clamp rate to avoid extreme stretching
    rate = max(0.25, min(rate, 4.0))

    stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)

    target_samples = int(target_duration * sr)
    if len(stretched) > target_samples:
        stretched = stretched[:target_samples]
    elif len(stretched) < target_samples:
        stretched = np.pad(stretched, (0, target_samples - len(stretched)))

    return stretched.astype(np.float32)


def _pitch_shift(audio: np.ndarray, semitones: float, sr: int) -> np.ndarray:
    """Shift pitch by a number of semitones."""
    import librosa

    if abs(semitones) < 0.1:
        return audio
    # Clamp to avoid extreme shifts
    semitones = max(-12.0, min(semitones, 12.0))
    return librosa.effects.pitch_shift(audio.astype(np.float32), sr=sr, n_steps=semitones)


def _match_loudness(audio: np.ndarray, target_rms: float) -> np.ndarray:
    """Scale audio to match a target RMS level."""
    current_rms = float(np.sqrt(np.mean(audio**2)))
    if current_rms > 1e-8 and target_rms > 1e-8:
        audio = audio * (target_rms / current_rms)
    return audio


def _spectral_smooth(
    replacement: np.ndarray,
    original_pre: np.ndarray,
    original_post: np.ndarray,
    sr: int,
    fade_ms: int = 30,
) -> np.ndarray:
    """Apply spectral smoothing at splice boundaries.

    Uses overlap-add with a Hann window to create smooth spectral transitions
    between the original audio context and the replacement segment.
    """
    fade_samples = min(int(sr * fade_ms / 1000), len(replacement) // 4)
    if fade_samples < 4:
        return replacement

    result = replacement.copy()

    # Fade-in: blend from original_pre context into replacement start
    if len(original_pre) >= fade_samples:
        window = np.hanning(fade_samples * 2)[:fade_samples].astype(np.float32)
        pre_tail = original_pre[-fade_samples:]
        result[:fade_samples] = pre_tail * (1 - window) + result[:fade_samples] * window

    # Fade-out: blend from replacement end into original_post context
    if len(original_post) >= fade_samples:
        window = np.hanning(fade_samples * 2)[fade_samples:].astype(np.float32)
        post_head = original_post[:fade_samples]
        result[-fade_samples:] = result[-fade_samples:] * (1 - window) + post_head * window

    return result


def _clamp_bounds(start_s: float, end_s: float, sample_rate: int, total: int) -> tuple[int, int]:
    """Clamp float timestamps to valid sample indices."""
    start = max(0, min(total, int(start_s * sample_rate)))
    end = max(start, min(total, int(end_s * sample_rate)))
    return start, end


def generate_replacement(
    flagged_word: FlaggedWord,
    vocals: np.ndarray,
    sample_rate: int,
    mapping: dict[str, str],
    language: str = "en",
    *,
    tts_voice: str | None = None,
) -> np.ndarray | None:
    """Generate replacement audio for a single flagged word.

    Orchestrates: mapping lookup -> TTS -> time stretch -> pitch match ->
    loudness match -> spectral smoothing.

    Args:
        flagged_word: The word to replace.
        vocals: Full vocal track array (samples,) or (samples, channels).
        sample_rate: Audio sample rate.
        mapping: Profanity-to-clean-word dictionary.
        language: Language code for TTS voice selection.
        tts_voice: Override TTS voice name.

    Returns:
        Replacement audio array matching the original's duration and shape,
        or None if no replacement mapping exists.
    """
    from sanitune.tts import synthesize

    # Step 1: Look up replacement word
    replacement_text = get_replacement(flagged_word.matched_term, mapping)
    if replacement_text is None:
        logger.debug("No mapping for '%s', falling back to mute", flagged_word.matched_term)
        return None

    # Step 2: Extract original word segment for reference
    start_sample, end_sample = _clamp_bounds(
        flagged_word.word.start, flagged_word.word.end, sample_rate, len(vocals),
    )
    original_segment = vocals[start_sample:end_sample]

    if len(original_segment) == 0:
        return None

    # Work in mono for processing
    if original_segment.ndim == 2:
        original_mono = original_segment.mean(axis=1)
    else:
        original_mono = original_segment

    original_duration = flagged_word.word.end - flagged_word.word.start

    # Step 3: Generate TTS audio
    try:
        tts_audio, tts_sr = synthesize(
            replacement_text,
            language=language,
            voice=tts_voice,
            sample_rate=sample_rate,
        )
    except (ImportError, RuntimeError) as exc:
        logger.warning("TTS failed for '%s': %s — falling back to mute", replacement_text, exc)
        return None

    # Step 4: Time-stretch to match original duration
    tts_audio = _time_stretch(tts_audio, original_duration, sample_rate)

    # Step 5: Pitch matching — shift TTS to match singer's pitch
    try:
        orig_f0 = _extract_median_f0(original_mono.astype(np.float32), sample_rate)
        tts_f0 = _extract_median_f0(tts_audio.astype(np.float32), sample_rate)

        if orig_f0 > 0 and tts_f0 > 0:
            semitones = 12 * np.log2(orig_f0 / tts_f0)
            tts_audio = _pitch_shift(tts_audio, float(semitones), sample_rate)
            logger.debug(
                "Pitch-shifted '%s': %.1f Hz → %.1f Hz (%.1f st)",
                replacement_text, tts_f0, orig_f0, semitones,
            )
    except Exception as exc:
        logger.debug("Pitch matching failed for '%s': %s, skipping", replacement_text, exc)

    # Step 6: Match loudness
    orig_rms = float(np.sqrt(np.mean(original_mono**2)))
    tts_audio = _match_loudness(tts_audio, orig_rms)

    # Step 7: Spectral smoothing at boundaries
    context_samples = int(sample_rate * 0.05)  # 50ms context
    pre_start = max(0, start_sample - context_samples)
    post_end = min(len(vocals), end_sample + context_samples)

    if vocals.ndim == 2:
        pre_context = vocals[pre_start:start_sample].mean(axis=1)
        post_context = vocals[end_sample:post_end].mean(axis=1)
    else:
        pre_context = vocals[pre_start:start_sample]
        post_context = vocals[end_sample:post_end]

    tts_audio = _spectral_smooth(tts_audio, pre_context, post_context, sample_rate)

    # Step 8: Ensure exact length match
    target_len = end_sample - start_sample
    if len(tts_audio) > target_len:
        tts_audio = tts_audio[:target_len]
    elif len(tts_audio) < target_len:
        tts_audio = np.pad(tts_audio, (0, target_len - len(tts_audio)))

    # Step 9: Match channel layout
    if vocals.ndim == 2:
        channels = vocals.shape[1]
        tts_audio = np.tile(tts_audio[:, np.newaxis], (1, channels))

    logger.debug(
        "Generated replacement '%s' → '%s' at %.2f-%.2fs",
        flagged_word.word.text, replacement_text,
        flagged_word.word.start, flagged_word.word.end,
    )
    return tts_audio.astype(np.float32)


def replace_words(
    vocals: np.ndarray,
    sample_rate: int,
    flagged: list[FlaggedWord],
    language: str = "en",
    *,
    custom_mapping_path: Path | None = None,
    tts_voice: str | None = None,
) -> tuple[np.ndarray, int, int]:
    """Replace flagged words in the vocal track with clean alternatives.

    Words without a mapping entry are muted as a fallback.

    Args:
        vocals: Vocal audio array.
        sample_rate: Audio sample rate.
        flagged: List of flagged words.
        language: Language code.
        custom_mapping_path: Optional custom mapping JSON file.
        tts_voice: Override TTS voice.

    Returns:
        Tuple of (edited_vocals, replaced_count, muted_fallback_count).
    """
    if not flagged:
        return vocals.copy(), 0, 0

    mapping = load_mapping(language, custom_mapping_path)
    result = vocals.copy()
    replaced = 0
    muted_fallback = 0

    for fw in flagged:
        replacement = generate_replacement(
            fw, vocals, sample_rate, mapping,
            language=language, tts_voice=tts_voice,
        )

        start, end = _clamp_bounds(fw.word.start, fw.word.end, sample_rate, len(result))

        if start >= end:
            continue

        if replacement is not None:
            result[start:end] = replacement
            replaced += 1
        else:
            # Fallback: mute
            result[start:end] = 0.0
            muted_fallback += 1

    logger.info(
        "Replaced %d words, muted %d (no mapping) out of %d flagged",
        replaced, muted_fallback, len(flagged),
    )
    return result, replaced, muted_fallback
