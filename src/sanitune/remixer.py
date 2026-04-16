"""Remix processed vocals with instrumental track, with surgical editing and format preservation."""

from __future__ import annotations

import io
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from sanitune.detector import FlaggedWord

logger = logging.getLogger(__name__)

SUPPORTED_OUTPUT_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

# Map file extensions to ffmpeg encoder names
_FFMPEG_CODECS: dict[str, str] = {
    ".mp3": "libmp3lame",
    ".flac": "flac",
    ".ogg": "libvorbis",
    ".opus": "libopus",
    ".aac": "aac",
    ".m4a": "aac",
}


def detect_audio_format(audio_path: Path) -> dict[str, str | int | None]:
    """Detect audio format parameters using ffprobe.

    Returns dict with keys: codec, sample_rate, channels, bit_rate, extension.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(audio_path),
            ],
            capture_output=True,
            check=True,
            timeout=10,
        )
        probe = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return {"codec": None, "sample_rate": 44100, "channels": 2, "bit_rate": None, "extension": ".wav"}

    audio_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "audio"),
        {},
    )
    fmt = probe.get("format", {})
    return {
        "codec": audio_stream.get("codec_name"),
        "sample_rate": int(audio_stream.get("sample_rate", 44100)),
        "channels": int(audio_stream.get("channels", 2)),
        "bit_rate": audio_stream.get("bit_rate") or fmt.get("bit_rate"),
        "extension": audio_path.suffix.lower(),
    }


def _encode_with_ffmpeg(
    audio: np.ndarray,
    sample_rate: int,
    output_path: Path,
    format_info: dict[str, str | int | None],
) -> Path:
    """Encode audio to a non-WAV format using ffmpeg."""
    ext = output_path.suffix.lower()
    codec = _FFMPEG_CODECS.get(ext)
    if codec is None:
        raise ValueError(f"No ffmpeg codec mapping for '{ext}'")

    # Write PCM to a buffer
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    wav_bytes = buf.getvalue()

    cmd = [
        "ffmpeg", "-y", "-i", "pipe:0",
        "-codec:a", codec,
        "-ar", str(sample_rate),
    ]

    # Preserve bitrate if known
    bit_rate = format_info.get("bit_rate")
    if bit_rate and codec not in ("flac",):
        cmd.extend(["-b:a", str(bit_rate)])

    cmd.append(str(output_path))

    try:
        subprocess.run(cmd, input=wav_bytes, capture_output=True, check=True, timeout=120)
    except subprocess.SubprocessError as exc:
        raise RuntimeError(f"ffmpeg encoding to '{ext}' failed: {exc}") from exc

    return output_path


def surgical_remix(
    original: np.ndarray,
    edited_vocals: np.ndarray,
    instrumentals: np.ndarray,
    sample_rate: int,
    flagged: list[FlaggedWord],
    *,
    margin_ms: int = 50,
    crossfade_ms: int = 20,
) -> np.ndarray:
    """Surgically replace only flagged regions while preserving original audio.

    For flagged regions: uses instrumentals + edited_vocals.
    For everything else: keeps the original audio byte-for-byte unchanged.
    """
    result = original.copy()
    total = len(result)
    cf_samples = int(sample_rate * crossfade_ms / 1000)
    margin_samples = int(sample_rate * margin_ms / 1000)

    for fw in flagged:
        start = max(0, int(fw.word.start * sample_rate) - margin_samples)
        end = min(total, int(fw.word.end * sample_rate) + margin_samples)

        if start >= end:
            continue

        # Build the edited content for this region
        region_len = end - start
        edited_region = instrumentals[start:end] + edited_vocals[start:end]

        # Crossfade at boundaries for seamless transitions
        cf = min(cf_samples, region_len // 4)
        if cf >= 2:
            fade_in = np.linspace(0, 1, cf, dtype=np.float32)
            fade_out = np.linspace(1, 0, cf, dtype=np.float32)
            if result.ndim == 2:
                fade_in = fade_in[:, np.newaxis]
                fade_out = fade_out[:, np.newaxis]

            # Blend original → edited at start
            edited_region[:cf] = original[start:start + cf] * (1 - fade_in) + edited_region[:cf] * fade_in
            # Blend edited → original at end
            edited_region[-cf:] = edited_region[-cf:] * fade_out + original[end - cf:end] * (1 - fade_out)

        result[start:end] = edited_region

    # Prevent clipping
    peak = float(np.abs(result).max())
    if peak > 1.0:
        result /= peak
        logger.info("Normalized output to prevent clipping (peak was %.2f)", peak)

    return result


def remix(
    vocals: np.ndarray,
    instrumentals: np.ndarray,
    sample_rate: int,
    output_path: Path,
    *,
    vocal_gain: float = 1.0,
    instrumental_gain: float = 1.0,
    original: np.ndarray | None = None,
    flagged: list[FlaggedWord] | None = None,
    input_format: dict[str, str | int | None] | None = None,
) -> Path:
    """Mix processed vocals with instrumentals and write to file.

    If `original` and `flagged` are provided, uses surgical mode — only modifying
    flagged regions while preserving the original audio everywhere else.
    Supports format-preserving output (MP3, FLAC, OGG, etc.) via ffmpeg.
    """
    if output_path.suffix.lower() not in SUPPORTED_OUTPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported output file type '{output_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_OUTPUT_EXTENSIONS))}"
        )

    # Ensure both arrays have the same number of dimensions
    if vocals.ndim != instrumentals.ndim:
        if vocals.ndim == 1:
            vocals = vocals[:, np.newaxis]
        if instrumentals.ndim == 1:
            instrumentals = instrumentals[:, np.newaxis]

    # Match channel counts
    if vocals.ndim == 2 and instrumentals.ndim == 2:
        if vocals.shape[1] != instrumentals.shape[1]:
            vocal_ch = vocals.shape[1]
            instr_ch = instrumentals.shape[1]
            if vocal_ch == 1:
                vocals = np.tile(vocals, (1, instr_ch))
            elif instr_ch == 1:
                instrumentals = np.tile(instrumentals, (1, vocal_ch))
            else:
                raise ValueError(
                    f"Channel mismatch: vocals={vocal_ch}, instrumentals={instr_ch}. "
                    "Only mono-to-multichannel upmix is supported."
                )

    # Pad shorter track with silence
    len_diff = len(vocals) - len(instrumentals)
    if len_diff > 0:
        pad_shape = (len_diff,) if instrumentals.ndim == 1 else (len_diff, instrumentals.shape[1])
        instrumentals = np.concatenate([instrumentals, np.zeros(pad_shape, dtype=instrumentals.dtype)])
    elif len_diff < 0:
        pad_shape = (-len_diff,) if vocals.ndim == 1 else (-len_diff, vocals.shape[1])
        vocals = np.concatenate([vocals, np.zeros(pad_shape, dtype=vocals.dtype)])

    # Choose remix strategy
    if original is not None and flagged is not None:
        # Surgical mode: only modify flagged regions
        # Match original dimensions/length to separated tracks
        orig = original
        if orig.ndim != vocals.ndim:
            if orig.ndim == 1:
                orig = orig[:, np.newaxis]
        if orig.ndim == 2 and vocals.ndim == 2 and orig.shape[1] != vocals.shape[1]:
            if orig.shape[1] == 1:
                orig = np.tile(orig, (1, vocals.shape[1]))
        orig_len = len(orig)
        target_len = len(vocals)
        if orig_len < target_len:
            pad = np.zeros((target_len - orig_len,) + orig.shape[1:], dtype=orig.dtype)
            orig = np.concatenate([orig, pad])
        elif orig_len > target_len:
            orig = orig[:target_len]

        mixed = surgical_remix(orig, vocals, instrumentals, sample_rate, flagged)
        logger.info("Surgical remix: %d regions edited, rest preserved from original", len(flagged))
    else:
        # Full remix (legacy mode)
        mixed = (vocals * vocal_gain) + (instrumentals * instrumental_gain)
        peak = float(np.abs(mixed).max())
        if peak > 1.0:
            mixed /= peak
            logger.info("Normalized output to prevent clipping (peak was %.2f)", peak)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".wav":
        sf.write(str(output_path), mixed, sample_rate)
    else:
        fmt = input_format or {}
        _encode_with_ffmpeg(mixed, sample_rate, output_path, fmt)

    logger.info("Wrote output to %s (%d samples, %d Hz)", output_path, len(mixed), sample_rate)
    return output_path
