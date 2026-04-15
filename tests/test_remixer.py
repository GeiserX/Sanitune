"""Tests for the remixer module."""

import numpy as np
import pytest

from sanitune.remixer import remix


def test_remix_basic(tmp_path):
    sr = 16000
    vocals = np.random.randn(sr).astype(np.float32) * 0.3
    instrumentals = np.random.randn(sr).astype(np.float32) * 0.3
    out = tmp_path / "output.wav"

    result = remix(vocals, instrumentals, sr, out)
    assert result.exists()
    assert result == out


def test_remix_different_lengths(tmp_path):
    sr = 16000
    vocals = np.zeros(sr, dtype=np.float32)
    instrumentals = np.zeros(sr * 2, dtype=np.float32)
    out = tmp_path / "output.wav"

    result = remix(vocals, instrumentals, sr, out)
    assert result.exists()


def test_remix_stereo(tmp_path):
    sr = 44100
    vocals = np.random.randn(sr, 2).astype(np.float32) * 0.3
    instrumentals = np.random.randn(sr, 2).astype(np.float32) * 0.3
    out = tmp_path / "output.wav"

    result = remix(vocals, instrumentals, sr, out)
    assert result.exists()


def test_remix_prevents_clipping(tmp_path):
    import soundfile as sf

    sr = 16000
    vocals = np.ones(sr, dtype=np.float32) * 0.8
    instrumentals = np.ones(sr, dtype=np.float32) * 0.8
    out = tmp_path / "output.wav"

    remix(vocals, instrumentals, sr, out)

    data, _ = sf.read(str(out))
    assert np.abs(data).max() <= 1.0


def test_remix_creates_parent_dirs(tmp_path):
    sr = 16000
    vocals = np.zeros(sr, dtype=np.float32)
    instrumentals = np.zeros(sr, dtype=np.float32)
    out = tmp_path / "subdir" / "deep" / "output.wav"

    result = remix(vocals, instrumentals, sr, out)
    assert result.exists()


def test_remix_mono_to_stereo_upmix(tmp_path):
    """Mono vocals + stereo instrumentals should upmix vocals to stereo."""
    sr = 16000
    vocals = np.ones((sr, 1), dtype=np.float32) * 0.3
    instrumentals = np.ones((sr, 2), dtype=np.float32) * 0.3
    out = tmp_path / "output.wav"

    import soundfile as sf

    remix(vocals, instrumentals, sr, out)
    data, _ = sf.read(str(out))
    assert data.ndim == 2
    assert data.shape[1] == 2


def test_remix_rejects_multichannel_mismatch(tmp_path):
    """Non-mono channel mismatch (e.g. 2 vs 6) should raise ValueError."""
    sr = 16000
    vocals = np.zeros((sr, 2), dtype=np.float32)
    instrumentals = np.zeros((sr, 6), dtype=np.float32)
    out = tmp_path / "output.wav"

    with pytest.raises(ValueError, match="Channel mismatch"):
        remix(vocals, instrumentals, sr, out)


def test_remix_rejects_unsupported_output_extension(tmp_path):
    sr = 16000
    vocals = np.zeros(sr, dtype=np.float32)
    instrumentals = np.zeros(sr, dtype=np.float32)
    out = tmp_path / "output.m4a"

    with pytest.raises(ValueError, match="Unsupported output file type"):
        remix(vocals, instrumentals, sr, out)
