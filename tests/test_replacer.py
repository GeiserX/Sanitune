"""Tests for the voice replacement pipeline."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from sanitune.detector import FlaggedWord
from sanitune.replacer import (
    _match_loudness,
    _spectral_smooth,
    generate_replacement,
    replace_words,
)
from sanitune.transcriber import Word


def _make_flagged(text: str, start: float, end: float, matched: str | None = None) -> FlaggedWord:
    return FlaggedWord(
        word=Word(text=text, start=start, end=end),
        matched_term=matched or text.lower(),
        index=0,
    )


class TestMatchLoudness:
    def test_scales_to_target(self):
        audio = np.ones(100, dtype=np.float32) * 0.1
        result = _match_loudness(audio, target_rms=0.5)
        result_rms = float(np.sqrt(np.mean(result**2)))
        assert abs(result_rms - 0.5) < 0.01

    def test_silent_audio_unchanged(self):
        audio = np.zeros(100, dtype=np.float32)
        result = _match_loudness(audio, target_rms=0.5)
        assert np.allclose(result, 0.0)

    def test_zero_target_unchanged(self):
        audio = np.ones(100, dtype=np.float32) * 0.5
        result = _match_loudness(audio, target_rms=0.0)
        np.testing.assert_array_equal(result, audio)


class TestSpectralSmooth:
    def test_smooths_boundaries(self):
        sr = 16000
        replacement = np.ones(1600, dtype=np.float32) * 0.5
        pre = np.ones(800, dtype=np.float32) * 0.2
        post = np.ones(800, dtype=np.float32) * 0.3

        result = _spectral_smooth(replacement, pre, post, sr, fade_ms=30)
        fade_samples = int(sr * 30 / 1000)
        # Start region should be blended toward pre value
        assert result[0] != 0.5
        # Mid-fade region at end should be blended toward post value
        assert result[-fade_samples // 2] != 0.5
        # Middle of audio should be unchanged
        assert result[len(result) // 2] == 0.5

    def test_short_audio_no_crash(self):
        sr = 16000
        replacement = np.ones(10, dtype=np.float32)
        pre = np.ones(10, dtype=np.float32)
        post = np.ones(10, dtype=np.float32)

        result = _spectral_smooth(replacement, pre, post, sr, fade_ms=30)
        assert len(result) == 10


class TestGenerateReplacement:
    def test_returns_none_for_unmapped_word(self):
        vocals = np.ones(16000, dtype=np.float32) * 0.3
        fw = _make_flagged("unknownword", 0.3, 0.5, matched="unknownword")
        result = generate_replacement(fw, vocals, 16000, {}, language="en")
        assert result is None

    @patch("sanitune.tts.synthesize")
    @patch("sanitune.replacer._pitch_shift", side_effect=lambda a, s, sr: a)
    @patch("sanitune.replacer._time_stretch")
    @patch("sanitune.replacer._extract_median_f0")
    def test_generates_replacement_audio(self, mock_f0, mock_stretch, mock_pitch, mock_synth):
        sr = 16000
        duration = 0.3  # 0.8 - 0.5
        target_len = int(duration * sr)
        vocals = np.random.randn(sr * 2).astype(np.float32) * 0.3
        fw = _make_flagged("fuck", 0.5, 0.8, matched="fuck")
        mapping = {"fuck": "fudge"}

        tts_audio = np.random.randn(int(sr * 0.5)).astype(np.float32) * 0.1
        mock_synth.return_value = (tts_audio, sr)
        mock_stretch.return_value = np.random.randn(target_len).astype(np.float32) * 0.1
        mock_f0.return_value = 200.0

        result = generate_replacement(fw, vocals, sr, mapping, language="en")

        assert result is not None
        assert len(result) == target_len
        mock_synth.assert_called_once()

    @patch("sanitune.tts.synthesize")
    @patch("sanitune.replacer._pitch_shift", side_effect=lambda a, s, sr: a)
    @patch("sanitune.replacer._time_stretch")
    @patch("sanitune.replacer._extract_median_f0")
    def test_matches_stereo_shape(self, mock_f0, mock_stretch, mock_pitch, mock_synth):
        sr = 16000
        duration = 0.3
        target_len = int(duration * sr)
        vocals = np.random.randn(sr * 2, 2).astype(np.float32) * 0.3
        fw = _make_flagged("damn", 0.5, 0.8, matched="damn")
        mapping = {"damn": "darn"}

        tts_audio = np.random.randn(int(sr * 0.5)).astype(np.float32) * 0.1
        mock_synth.return_value = (tts_audio, sr)
        mock_stretch.return_value = np.random.randn(target_len).astype(np.float32) * 0.1
        mock_f0.return_value = 200.0

        result = generate_replacement(fw, vocals, sr, mapping, language="en")

        assert result is not None
        assert result.ndim == 2
        assert result.shape[1] == 2

    @patch("sanitune.tts.synthesize")
    def test_tts_failure_returns_none(self, mock_synth):
        sr = 16000
        vocals = np.ones(sr, dtype=np.float32) * 0.3
        fw = _make_flagged("fuck", 0.3, 0.5, matched="fuck")
        mapping = {"fuck": "fudge"}

        mock_synth.side_effect = RuntimeError("Network error")

        result = generate_replacement(fw, vocals, sr, mapping, language="en")
        assert result is None


class TestReplaceWords:
    @patch("sanitune.replacer.generate_replacement")
    def test_replaces_and_counts(self, mock_gen):
        sr = 16000
        vocals = np.ones(sr * 3, dtype=np.float32) * 0.5
        flagged = [
            _make_flagged("fuck", 0.5, 0.8, matched="fuck"),
            _make_flagged("shit", 1.5, 1.8, matched="shit"),
        ]

        # First word gets replacement, second doesn't
        replacement = np.ones(int(0.3 * sr), dtype=np.float32) * 0.2
        mock_gen.side_effect = [replacement, None]

        result, replaced, muted = replace_words(vocals, sr, flagged, language="en")

        assert replaced == 1
        assert muted == 1
        assert result.shape == vocals.shape

    def test_empty_flagged_returns_copy(self):
        vocals = np.ones(16000, dtype=np.float32)
        result, replaced, muted = replace_words(vocals, 16000, [], language="en")
        np.testing.assert_array_equal(result, vocals)
        assert result is not vocals
        assert replaced == 0
        assert muted == 0
