"""Tests for the remixer module."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from sanitune.detector import FlaggedWord
from sanitune.remixer import (
    _FFMPEG_CODECS,
    SUPPORTED_OUTPUT_EXTENSIONS,
    _encode_with_ffmpeg,
    detect_audio_format,
    remix,
    surgical_remix,
)
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


# --- Additional tests for uncovered paths ---


def test_detect_audio_format_success(tmp_path):
    """detect_audio_format should return format info using ffprobe."""
    wav_file = tmp_path / "test.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(str(wav_file), audio, 16000)

    ffprobe_output = json.dumps({
        "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "sample_rate": "16000", "channels": "1"}],
        "format": {"bit_rate": "256000"},
    })
    with patch("sanitune.remixer.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=ffprobe_output.encode())
        result = detect_audio_format(wav_file)
    assert isinstance(result, dict)
    assert result["codec"] == "pcm_s16le"
    assert result["sample_rate"] == 16000
    assert result["channels"] == 1
    assert result["extension"] == ".wav"


def test_detect_audio_format_fallback(tmp_path):
    """Should return defaults when ffprobe fails."""
    fake_file = tmp_path / "nonexistent.wav"
    fake_file.write_bytes(b"not audio")

    import subprocess as _sp
    with patch("sanitune.remixer.subprocess.run", side_effect=_sp.SubprocessError("no ffprobe")):
        result = detect_audio_format(fake_file)
    assert isinstance(result, dict)
    assert result["sample_rate"] == 44100


def test_supported_output_extensions():
    """Verify expected extensions are supported."""
    assert ".wav" in SUPPORTED_OUTPUT_EXTENSIONS
    assert ".mp3" in SUPPORTED_OUTPUT_EXTENSIONS
    assert ".flac" in SUPPORTED_OUTPUT_EXTENSIONS
    assert ".ogg" in SUPPORTED_OUTPUT_EXTENSIONS


def test_ffmpeg_codecs_mapping():
    """Verify expected codec mappings exist."""
    assert _FFMPEG_CODECS[".mp3"] == "libmp3lame"
    assert _FFMPEG_CODECS[".flac"] == "flac"
    assert _FFMPEG_CODECS[".ogg"] == "libvorbis"


def test_encode_with_ffmpeg_no_codec(tmp_path):
    """Should raise ValueError for unknown extension."""
    audio = np.zeros(1000, dtype=np.float32)
    out = tmp_path / "output.xyz"

    with pytest.raises(ValueError, match="No ffmpeg codec mapping"):
        _encode_with_ffmpeg(audio, 16000, out, {})


@patch("sanitune.remixer.subprocess")
def test_encode_with_ffmpeg_failure(mock_subprocess, tmp_path):
    """Should raise RuntimeError when ffmpeg fails."""
    import subprocess

    mock_subprocess.run.side_effect = subprocess.SubprocessError("ffmpeg died")
    mock_subprocess.SubprocessError = subprocess.SubprocessError

    audio = np.zeros(1000, dtype=np.float32)
    out = tmp_path / "output.mp3"

    with pytest.raises(RuntimeError, match="ffmpeg encoding"):
        _encode_with_ffmpeg(audio, 16000, out, {})


def test_remix_mono_vocals_1d_stereo_instrumentals(tmp_path):
    """1D mono vocals + 2D stereo instrumentals should work with dimension promotion."""
    sr = 16000
    vocals = np.zeros(sr, dtype=np.float32)  # 1D
    instrumentals = np.zeros((sr, 2), dtype=np.float32)  # 2D stereo
    out = tmp_path / "output.wav"

    result = remix(vocals, instrumentals, sr, out)
    assert result.exists()


def test_remix_vocals_longer_than_instrumentals(tmp_path):
    """When vocals are longer, instrumentals should be padded."""
    sr = 16000
    vocals = np.zeros(sr * 2, dtype=np.float32)
    instrumentals = np.zeros(sr, dtype=np.float32)
    out = tmp_path / "output.wav"

    result = remix(vocals, instrumentals, sr, out)
    assert result.exists()


def test_surgical_remix_stereo(tmp_path):
    """Surgical remix should handle stereo arrays correctly."""
    sr = 16000
    original = np.random.randn(sr, 2).astype(np.float32) * 0.05
    edited_vocals = np.random.randn(sr, 2).astype(np.float32) * 0.02
    instrumentals = np.random.randn(sr, 2).astype(np.float32) * 0.02

    flagged = [
        FlaggedWord(
            word=Word(text="bad", start=0.2, end=0.4),
            matched_term="bad",
            index=0,
        ),
    ]

    result = surgical_remix(original, edited_vocals, instrumentals, sr, flagged)
    assert result.shape == original.shape


def test_surgical_remix_clipping_prevention():
    """Should normalize if peak > 1.0."""
    sr = 16000
    original = np.ones(sr, dtype=np.float32) * 0.9
    edited_vocals = np.ones(sr, dtype=np.float32) * 0.9
    instrumentals = np.ones(sr, dtype=np.float32) * 0.9

    flagged = [
        FlaggedWord(
            word=Word(text="bad", start=0.0, end=1.0),
            matched_term="bad",
            index=0,
        ),
    ]

    result = surgical_remix(original, edited_vocals, instrumentals, sr, flagged)
    assert np.abs(result).max() <= 1.0


def test_surgical_remix_skip_invalid_region():
    """When start >= end (e.g. very short word), region should be skipped."""
    sr = 16000
    original = np.ones(sr, dtype=np.float32) * 0.5
    edited_vocals = np.ones(sr, dtype=np.float32) * 0.1
    instrumentals = np.ones(sr, dtype=np.float32) * 0.1

    # Create a word that after margin subtraction might have start >= end
    flagged = [
        FlaggedWord(
            word=Word(text="x", start=0.003, end=0.003),  # zero-length
            matched_term="x",
            index=0,
        ),
    ]

    result = surgical_remix(original, edited_vocals, instrumentals, sr, flagged, margin_ms=0)
    # Should complete without modification
    np.testing.assert_array_equal(result, original)


def test_remix_with_non_wav_format(tmp_path):
    """Remix to non-WAV format should call _encode_with_ffmpeg."""
    sr = 16000
    vocals = np.zeros(sr, dtype=np.float32)
    instrumentals = np.zeros(sr, dtype=np.float32)
    out = tmp_path / "output.mp3"

    with patch("sanitune.remixer._encode_with_ffmpeg", return_value=out) as mock_encode:
        result = remix(vocals, instrumentals, sr, out, input_format={"bit_rate": "320000"})
        mock_encode.assert_called_once()
        assert result == out


def test_remix_surgical_original_mono_vocals_stereo(tmp_path):
    """Surgical mode with mono original and stereo vocals should handle dimension mismatch."""
    sr = 16000
    vocals = np.zeros((sr, 2), dtype=np.float32)
    instrumentals = np.zeros((sr, 2), dtype=np.float32)
    original = np.zeros(sr, dtype=np.float32)  # mono 1D
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
