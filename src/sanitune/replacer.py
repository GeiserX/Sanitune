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


def _match_pitch_contour(tts_audio: np.ndarray, original_audio: np.ndarray, sr: int) -> np.ndarray:
    """Warp TTS pitch to follow the original word's melody contour using WORLD vocoder.

    Extracts the F0 contour from the original sung word and applies it to
    the TTS output, making the replacement follow the exact melody.
    """
    import pyworld as pw

    # WORLD needs float64
    tts_f64 = tts_audio.astype(np.float64)
    orig_f64 = original_audio.astype(np.float64)

    # Extract F0 contour from the original sung word
    orig_f0, orig_t = pw.dio(orig_f64, sr)
    orig_f0 = pw.stonemask(orig_f64, orig_f0, orig_t, sr)

    # Analyze TTS with WORLD (F0 + spectral envelope + aperiodicity)
    tts_f0, tts_t = pw.dio(tts_f64, sr)
    tts_f0 = pw.stonemask(tts_f64, tts_f0, tts_t, sr)
    tts_sp = pw.cheaptrick(tts_f64, tts_f0, tts_t, sr)
    tts_ap = pw.d4c(tts_f64, tts_f0, tts_t, sr)

    # Resample original F0 contour to match TTS frame count
    if len(orig_f0) != len(tts_f0):
        from scipy.interpolate import interp1d

        orig_x = np.linspace(0, 1, len(orig_f0))
        tts_x = np.linspace(0, 1, len(tts_f0))
        interp = interp1d(orig_x, orig_f0, kind="linear", fill_value="extrapolate")
        target_f0 = interp(tts_x)
    else:
        target_f0 = orig_f0.copy()

    # Where original is voiced, use its F0; where unvoiced, keep TTS unvoiced
    for i in range(len(target_f0)):
        if target_f0[i] <= 0 and tts_f0[i] > 0:
            # Original unvoiced but TTS voiced — keep TTS F0
            target_f0[i] = tts_f0[i]

    # Resynthesize with the original's pitch contour but TTS spectral content
    result = pw.synthesize(target_f0, tts_sp, tts_ap, sr)
    return result.astype(np.float32)


def _time_stretch(audio: np.ndarray, target_duration: float, sr: int) -> np.ndarray:
    """Time-stretch audio to match a target duration."""
    import librosa

    current_duration = len(audio) / sr
    if current_duration <= 0 or target_duration <= 0:
        return audio

    rate = current_duration / target_duration
    # Clamp rate to avoid extreme stretching artifacts
    rate = max(0.5, min(rate, 2.0))

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
    # Clamp to avoid chipmunk effect — without voice conversion, large shifts sound unnatural
    semitones = max(-4.0, min(semitones, 4.0))
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
    reference_audio: np.ndarray | None = None,
    device: str = "cpu",
    synth_engine: str = "edge-tts",
    kits_api_key: str | None = None,
    kits_voice_model_id: int | None = None,
) -> np.ndarray | None:
    """Generate replacement audio for a single flagged word.

    Pipeline: mapping lookup → TTS → pitch match → voice conversion (Seed-VC)
    → time stretch → loudness match → spectral smoothing.

    If Seed-VC is available and reference_audio is provided, the TTS output
    is converted to match the singer's voice timbre. Otherwise falls back
    to pitch-shifted TTS.

    Args:
        flagged_word: The word to replace.
        vocals: Full vocal track array (samples,) or (samples, channels).
        sample_rate: Audio sample rate.
        mapping: Profanity-to-clean-word dictionary.
        language: Language code for TTS voice selection.
        tts_voice: Override TTS voice name.
        reference_audio: Mono float32 reference clip of the singer's voice for Seed-VC.
        device: Compute device for voice conversion.

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
            engine=synth_engine,
            singing=synth_engine == "bark",
        )
    except (ImportError, RuntimeError) as exc:
        logger.warning("TTS failed for '%s': %s — falling back to mute", replacement_text, exc)
        return None

    # Step 4: Pitch contour matching — warp TTS to follow the original melody
    # Uses WORLD vocoder to transplant the singer's F0 contour onto the TTS,
    # preserving the exact melody, inflection, and emotion of the original delivery.
    try:
        tts_audio = _match_pitch_contour(tts_audio, original_mono.astype(np.float32), sample_rate)
        logger.debug("Pitch contour matched for '%s'", replacement_text)
    except Exception as exc:
        logger.debug("Pitch contour matching failed for '%s': %s, falling back to median shift", replacement_text, exc)
        # Fallback: simple median F0 shift
        try:
            orig_f0 = _extract_median_f0(original_mono.astype(np.float32), sample_rate)
            tts_f0 = _extract_median_f0(tts_audio.astype(np.float32), sample_rate)
            if orig_f0 > 0 and tts_f0 > 0:
                semitones_shift = 12 * np.log2(orig_f0 / tts_f0)
                tts_audio = _pitch_shift(tts_audio, float(semitones_shift), sample_rate)
        except Exception:
            pass

    # Step 5: Voice conversion — Kits.ai (cloud) or Seed-VC (local)
    if kits_api_key and kits_voice_model_id:
        try:
            from sanitune.kits_client import convert_voice as kits_convert

            logger.debug("Converting voice for '%s' via Kits.ai (model=%d)...", replacement_text, kits_voice_model_id)
            tts_audio = kits_convert(
                tts_audio, sample_rate,
                voice_model_id=kits_voice_model_id,
                api_key=kits_api_key,
            )
            logger.debug("Kits.ai voice conversion complete for '%s'", replacement_text)
        except Exception as exc:
            logger.warning(
                "Kits.ai voice conversion failed for '%s': %s — falling back to local",
                replacement_text, exc,
            )
    elif reference_audio is not None:
        try:
            from sanitune.voice_converter import convert_voice, is_available

            if is_available():
                logger.debug("Converting voice for '%s' via Seed-VC...", replacement_text)
                tts_audio = convert_voice(
                    tts_audio,
                    reference_audio,
                    sample_rate,
                    device=device,
                    diffusion_steps=30,
                    f0_condition=True,
                    auto_f0_adjust=False,
                    pitch_shift=0,
                )
                logger.debug("Voice conversion complete for '%s'", replacement_text)
        except Exception as exc:
            logger.warning(
                "Voice conversion failed for '%s': %s — using pitch-shifted TTS",
                replacement_text, exc,
            )

    # Step 6: Time-stretch to match original duration
    tts_audio = _time_stretch(tts_audio, original_duration, sample_rate)

    # Step 7: Match loudness
    orig_rms = float(np.sqrt(np.mean(original_mono**2)))
    tts_audio = _match_loudness(tts_audio, orig_rms)

    # Step 8: Spectral smoothing at boundaries
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

    # Step 9: Ensure exact length match
    target_len = end_sample - start_sample
    if len(tts_audio) > target_len:
        tts_audio = tts_audio[:target_len]
    elif len(tts_audio) < target_len:
        tts_audio = np.pad(tts_audio, (0, target_len - len(tts_audio)))

    # Step 10: Match channel layout
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
    margin_ms: int = 50,
    custom_mapping_path: Path | None = None,
    tts_voice: str | None = None,
    device: str = "cpu",
    synth_engine: str = "edge-tts",
    kits_api_key: str | None = None,
    kits_voice_model_id: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Replace flagged words in the vocal track with clean alternatives.

    If Seed-VC is available, extracts a reference clip from the singer's
    clean vocals and uses voice conversion for natural-sounding replacements.
    Words without a mapping entry are muted as a fallback.

    Args:
        vocals: Vocal audio array.
        sample_rate: Audio sample rate.
        flagged: List of flagged words.
        language: Language code.
        margin_ms: Extra margin in ms around each word.
        custom_mapping_path: Optional custom mapping JSON file.
        tts_voice: Override TTS voice.
        device: Compute device for voice conversion.

    Returns:
        Tuple of (edited_vocals, replaced_count, muted_fallback_count).
    """
    if not flagged:
        return vocals.copy(), 0, 0

    mapping = load_mapping(language, custom_mapping_path)
    result = vocals.copy()
    replaced = 0
    muted_fallback = 0
    margin_samples = int(sample_rate * margin_ms / 1000)

    # Extract singer's reference audio for voice conversion (once per song)
    reference_audio = None
    try:
        from sanitune.voice_converter import extract_reference, is_available

        if is_available():
            flagged_regions = [(fw.word.start, fw.word.end) for fw in flagged]
            reference_audio = extract_reference(vocals, sample_rate, flagged_regions)
            logger.info(
                "Extracted %.1fs reference audio for voice conversion",
                len(reference_audio) / sample_rate,
            )
    except ImportError:
        logger.debug("Seed-VC not available, using TTS-only replacement")

    for fw in flagged:
        replacement = generate_replacement(
            fw, vocals, sample_rate, mapping,
            language=language, tts_voice=tts_voice,
            reference_audio=reference_audio, device=device,
            synth_engine=synth_engine,
            kits_api_key=kits_api_key,
            kits_voice_model_id=kits_voice_model_id,
        )

        word_start, word_end = _clamp_bounds(fw.word.start, fw.word.end, sample_rate, len(result))
        margin_start = max(0, word_start - margin_samples)
        margin_end = min(len(result), word_end + margin_samples)

        if word_start >= word_end:
            continue

        if replacement is not None:
            # Write replacement at exact word bounds (not margin-expanded)
            word_len = word_end - word_start
            result[word_start:word_end] = replacement[:word_len]
            replaced += 1
        else:
            # Fallback: mute the full margin-expanded region
            result[margin_start:margin_end] = 0.0
            muted_fallback += 1

    logger.info(
        "Replaced %d words, muted %d (no mapping) out of %d flagged",
        replaced, muted_fallback, len(flagged),
    )
    return result, replaced, muted_fallback
