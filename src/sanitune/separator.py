"""Vocal/instrumental separation using Demucs v4."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    logger.info("Separating vocals from instrumentals with %s on %s...", model_name, device)

    model = get_model(model_name)
    model.to(torch.device(device))
    sr = model.samplerate

    # Load and resample audio to model's expected sample rate
    wav, orig_sr = torchaudio.load(str(audio_path))
    if orig_sr != sr:
        logger.info("Resampling from %d to %d Hz...", orig_sr, sr)
        wav = torchaudio.transforms.Resample(orig_sr, sr)(wav)

    # apply_model expects (batch, channels, samples) → returns (batch, sources, channels, samples)
    sources = apply_model(model, wav[None], device=torch.device(device))

    # Extract vocals and sum remaining stems as instrumentals
    vocal_idx = model.sources.index("vocals")
    vocals = sources[0, vocal_idx].cpu().numpy()  # (channels, samples)

    non_vocal_indices = [i for i, name in enumerate(model.sources) if name != "vocals"]
    if not non_vocal_indices:
        instrumentals = np.zeros_like(vocals)
    else:
        instrumentals = sources[0, non_vocal_indices].sum(dim=0).cpu().numpy()

    # Convert from (channels, samples) to (samples, channels)
    vocals = vocals.T
    instrumentals = instrumentals.T

    logger.info("Separation complete. Vocals: %s, Instrumentals: %s, SR: %d", vocals.shape, instrumentals.shape, sr)
    return SeparationResult(vocals=vocals, instrumentals=instrumentals, sample_rate=sr)
