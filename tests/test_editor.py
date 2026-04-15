"""Tests for the audio editing module."""

import numpy as np
import pytest

from sanitune.detector import FlaggedWord
from sanitune.editor import edit
from sanitune.transcriber import Word


def _make_flagged(text: str, start: float, end: float, matched: str | None = None) -> FlaggedWord:
    return FlaggedWord(
        word=Word(text=text, start=start, end=end),
        matched_term=matched or text.lower(),
        index=0,
    )


def test_edit_mute_silences_region():
    sr = 16000
    audio = np.ones(sr, dtype=np.float32)  # 1 second of ones
    flagged = [_make_flagged("fuck", 0.3, 0.5)]

    result = edit(audio, sr, flagged, mode="mute", margin_ms=0)

    # Region around 0.3-0.5s should be silenced
    start = int(0.3 * sr)
    end = int(0.5 * sr)
    assert np.allclose(result[start:end], 0.0, atol=0.01)
    # Rest should be mostly unchanged
    assert result[0] != 0.0


def test_edit_bleep_replaces_with_tone():
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    flagged = [_make_flagged("shit", 0.3, 0.5)]

    result = edit(audio, sr, flagged, mode="bleep", bleep_freq=1000, margin_ms=0)

    start = int(0.3 * sr)
    end = int(0.5 * sr)
    # Bleep region should have non-zero values
    assert np.abs(result[start:end]).max() > 0


def test_edit_no_flagged_returns_copy():
    audio = np.ones(16000, dtype=np.float32)
    result = edit(audio, 16000, [], mode="mute")
    np.testing.assert_array_equal(result, audio)
    assert result is not audio  # should be a copy


def test_edit_stereo_audio():
    sr = 16000
    audio = np.ones((sr, 2), dtype=np.float32)
    flagged = [_make_flagged("damn", 0.2, 0.4)]

    result = edit(audio, sr, flagged, mode="mute", margin_ms=0)
    assert result.shape == audio.shape

    start = int(0.2 * sr)
    end = int(0.4 * sr)
    assert np.allclose(result[start:end], 0.0, atol=0.01)


def test_edit_does_not_modify_original():
    sr = 16000
    audio = np.ones(sr, dtype=np.float32)
    original = audio.copy()
    flagged = [_make_flagged("fuck", 0.3, 0.5)]

    edit(audio, sr, flagged, mode="mute")
    np.testing.assert_array_equal(audio, original)


def test_edit_rejects_invalid_mode():
    sr = 16000
    audio = np.ones(sr, dtype=np.float32)
    flagged = [_make_flagged("fuck", 0.3, 0.5)]

    with pytest.raises(ValueError, match="Unknown edit mode"):
        edit(audio, sr, flagged, mode="replace")

    with pytest.raises(ValueError, match="Unknown edit mode"):
        edit(audio, sr, flagged, mode="Mute")
