"""Singing voice conversion using Seed-VC — convert TTS output to match the singer's voice."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Singleton wrapper — loaded once per process, reused across all replacements
_wrapper = None
_wrapper_device = None


def _get_wrapper(device: str = "cpu"):
    """Lazy-load the Seed-VC wrapper singleton."""
    global _wrapper, _wrapper_device

    if _wrapper is not None and _wrapper_device == device:
        return _wrapper

    try:
        from seed_vc_wrapper import SeedVCWrapper
    except ImportError:
        raise ImportError(
            "Seed-VC is required for high-quality voice conversion. "
            "Clone it with: git clone https://github.com/Plachtaa/seed-vc.git "
            "and add to PYTHONPATH, or install sanitune[voice]."
        )

    import torch

    logger.info("Loading Seed-VC singing voice conversion model on %s...", device)
    _wrapper = SeedVCWrapper(device=torch.device(device))
    _wrapper_device = device
    logger.info("Seed-VC loaded successfully")
    return _wrapper


def extract_reference(
    vocals: np.ndarray,
    sample_rate: int,
    flagged_regions: list[tuple[float, float]] | None = None,
    *,
    target_duration: float = 10.0,
    max_duration: float = 25.0,
) -> np.ndarray:
    """Extract a reference clip of the singer's clean vocals, avoiding flagged regions.

    Args:
        vocals: Full vocal track (samples,) or (samples, channels).
        sample_rate: Sample rate.
        flagged_regions: List of (start_s, end_s) tuples to avoid.
        target_duration: Desired reference duration in seconds.
        max_duration: Maximum reference duration (Seed-VC clips at 25s).

    Returns:
        Mono float32 reference audio array.
    """
    # Work in mono
    if vocals.ndim == 2:
        mono = vocals.mean(axis=1)
    else:
        mono = vocals.copy()

    total_samples = len(mono)
    total_duration = total_samples / sample_rate

    if flagged_regions is None or not flagged_regions:
        # No flagged regions — take the loudest segment
        dur = min(target_duration, total_duration, max_duration)
        return _extract_loudest_segment(mono, sample_rate, dur)

    # Build list of clean (non-flagged) regions with 0.2s margin around each flagged word
    margin = 0.2
    flagged_sorted = sorted(flagged_regions, key=lambda r: r[0])
    clean_regions: list[tuple[int, int]] = []

    prev_end = 0
    for start_s, end_s in flagged_sorted:
        clean_start = prev_end
        clean_end = max(0, int((start_s - margin) * sample_rate))
        if clean_end > clean_start + sample_rate:  # at least 1 second
            clean_regions.append((clean_start, clean_end))
        prev_end = min(total_samples, int((end_s + margin) * sample_rate))

    # Final region after last flagged word
    if prev_end < total_samples - sample_rate:
        clean_regions.append((prev_end, total_samples))

    if not clean_regions:
        # All vocals are flagged — use the full track as reference anyway
        dur = min(target_duration, total_duration, max_duration)
        return _extract_loudest_segment(mono, sample_rate, dur)

    # Concatenate clean regions up to target duration
    target_samples = int(min(target_duration, max_duration) * sample_rate)
    segments: list[np.ndarray] = []
    collected = 0

    # Sort by RMS energy (loudest first — better reference quality)
    clean_regions.sort(key=lambda r: -float(np.sqrt(np.mean(mono[r[0]:r[1]] ** 2))))

    for start, end in clean_regions:
        needed = target_samples - collected
        if needed <= 0:
            break
        segment = mono[start:min(end, start + needed)]
        segments.append(segment)
        collected += len(segment)

    reference = np.concatenate(segments) if segments else mono[:target_samples]
    return reference.astype(np.float32)


def _extract_loudest_segment(mono: np.ndarray, sample_rate: int, duration: float) -> np.ndarray:
    """Extract the loudest contiguous segment of the given duration."""
    target_samples = int(duration * sample_rate)
    if len(mono) <= target_samples:
        return mono.astype(np.float32)

    # Compute RMS energy in sliding windows of 1 second
    window = sample_rate
    rms = np.array([
        float(np.sqrt(np.mean(mono[i:i + window] ** 2)))
        for i in range(0, len(mono) - window, window // 2)
    ])

    if len(rms) == 0:
        return mono[:target_samples].astype(np.float32)

    # Find the center of the loudest window, then extract target_samples around it
    best_idx = int(np.argmax(rms))
    center = best_idx * (window // 2) + window // 2
    start = max(0, center - target_samples // 2)
    end = start + target_samples
    if end > len(mono):
        end = len(mono)
        start = max(0, end - target_samples)

    return mono[start:end].astype(np.float32)


def convert_voice(
    tts_audio: np.ndarray,
    reference_audio: np.ndarray,
    sample_rate: int,
    *,
    device: str = "cpu",
    diffusion_steps: int = 30,
    f0_condition: bool = True,
    auto_f0_adjust: bool = False,
    pitch_shift: int = 0,
) -> np.ndarray:
    """Convert TTS audio to match the singer's voice using Seed-VC.

    Args:
        tts_audio: Mono float32 TTS output to convert.
        reference_audio: Mono float32 reference clip of the singer's voice.
        sample_rate: Sample rate of both arrays.
        device: Compute device.
        diffusion_steps: Number of diffusion steps (higher = better quality, slower).
        f0_condition: Use F0 conditioning for singing voice conversion.
        auto_f0_adjust: Auto-adjust F0 to match reference pitch range.
        pitch_shift: Semitone shift applied to the output.

    Returns:
        Converted mono float32 audio at 44100 Hz.
    """
    wrapper = _get_wrapper(device)

    # Seed-VC expects file paths — write temp files
    with tempfile.TemporaryDirectory(prefix="sanitune_vc_") as tmp_dir:
        src_path = Path(tmp_dir) / "source.wav"
        ref_path = Path(tmp_dir) / "reference.wav"

        sf.write(str(src_path), tts_audio, sample_rate)
        sf.write(str(ref_path), reference_audio, sample_rate)

        gen = wrapper.convert_voice(
            source=str(src_path),
            target=str(ref_path),
            diffusion_steps=diffusion_steps,
            length_adjust=1.0,
            inference_cfg_rate=0.7,
            f0_condition=f0_condition,
            auto_f0_adjust=auto_f0_adjust,
            pitch_shift=pitch_shift,
            stream_output=False,
        )

        # convert_voice uses yield internally, making it always a generator.
        # With stream_output=False, the audio is returned via StopIteration.value.
        import types

        if isinstance(gen, types.GeneratorType):
            result = None
            try:
                while True:
                    result = next(gen)
            except StopIteration as e:
                if e.value is not None:
                    result = e.value
        else:
            result = gen

    # Result is (sample_rate, numpy_array) tuple or just numpy_array
    if isinstance(result, tuple):
        out_sr, out_audio = result
    else:
        out_audio = result
        out_sr = 44100  # Seed-VC SVC outputs at 44100 Hz

    # Ensure float32
    if out_audio.dtype != np.float32:
        out_audio = out_audio.astype(np.float32)

    # Resample to match input sample rate if Seed-VC output differs
    if out_sr != sample_rate:
        import librosa

        out_audio = librosa.resample(
            out_audio, orig_sr=out_sr, target_sr=sample_rate,
        ).astype(np.float32)
        logger.debug("Resampled voice conversion output from %d Hz to %d Hz", out_sr, sample_rate)

    # Normalize if needed
    if out_audio.ndim > 1:
        out_audio = out_audio.mean(axis=-1) if out_audio.shape[-1] <= 2 else out_audio.flatten()

    logger.debug(
        "Voice conversion complete: %d samples at %d Hz",
        len(out_audio), sample_rate,
    )
    return out_audio


def is_available() -> bool:
    """Check if Seed-VC is available for import."""
    try:
        from seed_vc_wrapper import SeedVCWrapper  # noqa: F401
        return True
    except ImportError:
        return False
