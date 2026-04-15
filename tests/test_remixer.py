"""Tests for the remixer module."""

import numpy as np

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
