"""Vocal/instrumental separation using Demucs v4."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SeparationResult:
    vocals: np.ndarray
    instrumentals: np.ndarray
    sample_rate: int


def separate(audio_path: Path, *, device: str = "cpu", model_name: str = "htdemucs_ft") -> SeparationResult:
    """Separate a song into vocals and instrumentals using Demucs.

    Args:
        audio_path: Path to the input audio file.
        device: Compute device (cpu, cuda, mps).
        model_name: Demucs model to use.

    Returns:
        SeparationResult with vocals, instrumentals, and sample rate.
    """
    from demucs.api import Separator

    logger.info("Separating vocals from instrumentals with %s on %s...", model_name, device)

    sep = Separator(model=model_name, device=torch.device(device))
    origin, separated = sep.separate_audio_file(str(audio_path))

    sr = sep.samplerate
    vocals = separated["vocals"].cpu().numpy()
    # Sum all non-vocal stems into instrumentals
    instrumental_stems = [name for name in separated if name != "vocals"]
    instrumentals = sum(separated[name] for name in instrumental_stems).cpu().numpy()

    # Convert from (channels, samples) to (samples, channels) if needed
    if vocals.ndim == 2:
        vocals = vocals.T
    elif vocals.ndim == 3:
        # Demucs returns (batch, channels, samples)
        vocals = vocals.squeeze(0).T

    if instrumentals.ndim == 2:
        instrumentals = instrumentals.T
    elif instrumentals.ndim == 3:
        instrumentals = instrumentals.squeeze(0).T

    logger.info("Separation complete. Vocals: %s, Instrumentals: %s, SR: %d", vocals.shape, instrumentals.shape, sr)
    return SeparationResult(vocals=vocals, instrumentals=instrumentals, sample_rate=sr)
