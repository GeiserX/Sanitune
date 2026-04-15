"""Tests for the TTS module."""

from __future__ import annotations

import numpy as np
import pytest

from sanitune.tts import DEFAULT_VOICES, _trim_silence


def test_default_voices_has_en_and_es():
    assert "en" in DEFAULT_VOICES
    assert "es" in DEFAULT_VOICES


def test_trim_silence_removes_leading_trailing():
    # Create audio with silence-signal-silence
    audio = np.zeros(1000, dtype=np.float32)
    audio[200:800] = 0.5
    trimmed = _trim_silence(audio, threshold_db=-40)
    assert len(trimmed) < len(audio)
    assert np.abs(trimmed).max() > 0


def test_trim_silence_empty_audio():
    audio = np.array([], dtype=np.float32)
    result = _trim_silence(audio)
    assert len(result) == 0


def test_trim_silence_all_silence():
    audio = np.zeros(1000, dtype=np.float32)
    result = _trim_silence(audio)
    assert len(result) == 1000  # nothing to trim, returns as-is


def test_synthesize_raises_without_edge_tts(monkeypatch):
    """Synthesize should raise ImportError when edge-tts is not installed."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "edge_tts":
            raise ImportError("No module named 'edge_tts'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from sanitune.tts import synthesize

    with pytest.raises(ImportError, match="edge-tts is required"):
        synthesize("hello")
