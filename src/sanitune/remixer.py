"""Remix processed vocals with instrumental track."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def remix(
    vocals: np.ndarray,
    instrumentals: np.ndarray,
    sample_rate: int,
    output_path: Path,
    *,
    vocal_gain: float = 1.0,
    instrumental_gain: float = 1.0,
) -> Path:
    """Mix processed vocals with instrumentals and write to file.

    Args:
        vocals: Processed vocal track (samples,) or (samples, channels).
        instrumentals: Instrumental track (samples,) or (samples, channels).
        sample_rate: Sample rate for the output file.
        output_path: Path to write the output audio file.
        vocal_gain: Volume multiplier for vocals.
        instrumental_gain: Volume multiplier for instrumentals.

    Returns:
        Path to the written output file.
    """
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

    # Mix
    mixed = (vocals * vocal_gain) + (instrumentals * instrumental_gain)

    # Prevent clipping
    peak = np.abs(mixed).max()
    if peak > 1.0:
        mixed /= peak
        logger.info("Normalized output to prevent clipping (peak was %.2f)", peak)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mixed, sample_rate)
    logger.info("Wrote output to %s (%d samples, %d Hz)", output_path, len(mixed), sample_rate)
    return output_path
