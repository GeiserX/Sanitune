"""Shared test fixtures for Sanitune tests."""

from __future__ import annotations

import numpy as np
import pytest

from sanitune.transcriber import Word


@pytest.fixture
def sample_words() -> list[Word]:
    """A list of transcribed words with some profanity."""
    return [
        Word(text="This", start=0.0, end=0.3),
        Word(text="is", start=0.3, end=0.5),
        Word(text="a", start=0.5, end=0.6),
        Word(text="damn", start=0.6, end=0.9),
        Word(text="good", start=0.9, end=1.2),
        Word(text="fucking", start=1.2, end=1.6),
        Word(text="song", start=1.6, end=2.0),
    ]


@pytest.fixture
def mono_audio() -> np.ndarray:
    """1 second of mono silence at 16kHz."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def stereo_audio() -> np.ndarray:
    """1 second of stereo silence at 44100Hz."""
    return np.zeros((44100, 2), dtype=np.float32)
