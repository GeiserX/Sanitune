"""Tests for the remixer module."""

import numpy as np
import pytest

from sanitune.detector import FlaggedWord
from sanitune.remixer import remix, surgical_remix
from sanitune.transcriber import Word


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
    out = tmp_path / "output.wma"

    with pytest.raises(ValueError, match="Unsupported output file type"):
        remix(vocals, instrumentals, sr, out)


def test_surgical_remix_preserves_unflagged_regions():
    """Unflagged regions should be identical to original."""
    sr = 16000
    # Use low amplitudes to avoid triggering anti-clipping normalization
    original = np.random.randn(sr * 2).astype(np.float32) * 0.05
    edited_vocals = np.random.randn(sr * 2).astype(np.float32) * 0.02
    instrumentals = np.random.randn(sr * 2).astype(np.float32) * 0.02

    # Flag only a small region (0.5s - 0.7s)
    flagged = [
        FlaggedWord(
            word=Word(text="bad", start=0.5, end=0.7),
            matched_term="bad",
            index=0,
        ),
    ]

    result = surgical_remix(original, edited_vocals, instrumentals, sr, flagged)

    # Check that a region far from the flagged area is unchanged
    # Region 1.5s-1.8s should be untouched (well outside flagged + margin)
    check_start = int(1.5 * sr)
    check_end = int(1.8 * sr)
    np.testing.assert_array_equal(result[check_start:check_end], original[check_start:check_end])


def test_surgical_remix_modifies_flagged_region():
    """Flagged regions should differ from original."""
    sr = 16000
    original = np.ones(sr, dtype=np.float32) * 0.5
    edited_vocals = np.ones(sr, dtype=np.float32) * 0.1
    instrumentals = np.ones(sr, dtype=np.float32) * 0.1

    flagged = [
        FlaggedWord(
            word=Word(text="bad", start=0.2, end=0.4),
            matched_term="bad",
            index=0,
        ),
    ]

    result = surgical_remix(original, edited_vocals, instrumentals, sr, flagged)

    # The center of the flagged region should be edited (instrumentals + edited_vocals = 0.2, not 0.5)
    center = int(0.3 * sr)
    assert result[center] != original[center]


def test_remix_surgical_mode(tmp_path):
    """Remix with original + flagged should use surgical mode."""
    sr = 16000
    vocals = np.random.randn(sr, 2).astype(np.float32) * 0.3
    instrumentals = np.random.randn(sr, 2).astype(np.float32) * 0.3
    original = np.random.randn(sr, 2).astype(np.float32) * 0.3
    out = tmp_path / "output.wav"

    flagged = [
        FlaggedWord(
            word=Word(text="bad", start=0.1, end=0.3),
            matched_term="bad",
            index=0,
        ),
    ]

    result = remix(vocals, instrumentals, sr, out, original=original, flagged=flagged)
    assert result.exists()
