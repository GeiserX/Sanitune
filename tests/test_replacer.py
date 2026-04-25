"""Tests for the voice replacement pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from sanitune.detector import FlaggedWord
from sanitune.replacer import (
    _clamp_bounds,
    _match_loudness,
    _spectral_smooth,
    generate_replacement,
    replace_words,
)
from sanitune.transcriber import Word


def _make_flagged(text: str, start: float, end: float, matched: str | None = None, index: int = 0) -> FlaggedWord:
    return FlaggedWord(
        word=Word(text=text, start=start, end=end),
        matched_term=matched or text.lower(),
        index=index,
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

    def test_short_pre_context(self):
        """When pre context is shorter than fade_samples, no fade-in is applied."""
        sr = 16000
        replacement = np.ones(1600, dtype=np.float32) * 0.5
        pre = np.ones(2, dtype=np.float32) * 0.2  # too short
        post = np.ones(800, dtype=np.float32) * 0.3

        result = _spectral_smooth(replacement, pre, post, sr, fade_ms=30)
        # Start should be unchanged since pre is too short
        assert result[0] == 0.5

    def test_short_post_context(self):
        """When post context is shorter than fade_samples, no fade-out is applied."""
        sr = 16000
        replacement = np.ones(1600, dtype=np.float32) * 0.5
        pre = np.ones(800, dtype=np.float32) * 0.2
        post = np.ones(2, dtype=np.float32) * 0.3  # too short

        result = _spectral_smooth(replacement, pre, post, sr, fade_ms=30)
        # End should be unchanged since post is too short
        assert result[-1] == 0.5


class TestClampBounds:
    def test_normal_bounds(self):
        start, end = _clamp_bounds(0.5, 1.0, 16000, 32000)
        assert start == 8000
        assert end == 16000

    def test_clamps_to_zero(self):
        start, end = _clamp_bounds(-0.5, 0.5, 16000, 32000)
        assert start == 0
        assert end == 8000

    def test_clamps_to_total(self):
        start, end = _clamp_bounds(1.5, 3.0, 16000, 32000)
        assert start == 24000
        assert end == 32000

    def test_start_after_end(self):
        start, end = _clamp_bounds(2.0, 1.0, 16000, 32000)
        assert start <= end


class TestGenerateReplacement:
    def test_returns_none_for_unmapped_word(self):
        vocals = np.ones(16000, dtype=np.float32) * 0.3
        fw = _make_flagged("unknownword", 0.3, 0.5, matched="unknownword")
        result = generate_replacement(fw, vocals, 16000, {}, language="en")
        assert result is None

    @patch("sanitune.tts.synthesize")
    @patch("sanitune.replacer._match_pitch_contour")
    @patch("sanitune.replacer._time_stretch")
    def test_generates_replacement_audio(self, mock_stretch, mock_pitch, mock_synth):
        sr = 16000
        duration = 0.3  # 0.8 - 0.5
        target_len = int(duration * sr)
        vocals = np.random.randn(sr * 2).astype(np.float32) * 0.3
        fw = _make_flagged("fuck", 0.5, 0.8, matched="fuck")
        mapping = {"fuck": "fudge"}

        tts_audio = np.random.randn(int(sr * 0.5)).astype(np.float32) * 0.1
        mock_synth.return_value = (tts_audio, sr)
        mock_pitch.return_value = tts_audio
        mock_stretch.return_value = np.random.randn(target_len).astype(np.float32) * 0.1

        result = generate_replacement(fw, vocals, sr, mapping, language="en")

        assert result is not None
        assert len(result) == target_len
        mock_synth.assert_called_once()

    @patch("sanitune.tts.synthesize")
    @patch("sanitune.replacer._match_pitch_contour")
    @patch("sanitune.replacer._time_stretch")
    def test_matches_stereo_shape(self, mock_stretch, mock_pitch, mock_synth):
        sr = 16000
        duration = 0.3
        target_len = int(duration * sr)
        vocals = np.random.randn(sr * 2, 2).astype(np.float32) * 0.3
        fw = _make_flagged("damn", 0.5, 0.8, matched="damn")
        mapping = {"damn": "darn"}

        tts_audio = np.random.randn(int(sr * 0.5)).astype(np.float32) * 0.1
        mock_synth.return_value = (tts_audio, sr)
        mock_pitch.return_value = tts_audio
        mock_stretch.return_value = np.random.randn(target_len).astype(np.float32) * 0.1

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

    def test_zero_length_segment_returns_none(self):
        """When start == end, empty segment should return None."""
        sr = 16000
        vocals = np.ones(sr, dtype=np.float32) * 0.3
        fw = _make_flagged("fuck", 0.5, 0.5, matched="fuck")
        mapping = {"fuck": "fudge"}

        result = generate_replacement(fw, vocals, sr, mapping, language="en")
        assert result is None

    @patch("sanitune.tts.synthesize")
    @patch("sanitune.replacer._match_pitch_contour")
    @patch("sanitune.replacer._time_stretch")
    def test_pitch_contour_fallback(self, mock_stretch, mock_pitch, mock_synth):
        """When pitch contour matching fails, should fall back to median F0 shift."""
        sr = 16000
        duration = 0.3
        target_len = int(duration * sr)
        vocals = np.random.randn(sr * 2).astype(np.float32) * 0.3
        fw = _make_flagged("fuck", 0.5, 0.8, matched="fuck")
        mapping = {"fuck": "fudge"}

        tts_audio = np.random.randn(int(sr * 0.5)).astype(np.float32) * 0.1
        mock_synth.return_value = (tts_audio, sr)
        mock_pitch.side_effect = RuntimeError("pyworld failed")
        mock_stretch.return_value = np.random.randn(target_len).astype(np.float32) * 0.1

        # Patch median f0 and pitch shift
        with patch("sanitune.replacer._extract_median_f0", return_value=200.0), \
             patch("sanitune.replacer._pitch_shift", side_effect=lambda a, s, sr: a):
            result = generate_replacement(fw, vocals, sr, mapping, language="en")

        assert result is not None
        assert len(result) == target_len

    @patch("sanitune.tts.synthesize")
    @patch("sanitune.replacer._match_pitch_contour")
    @patch("sanitune.replacer._time_stretch")
    def test_kits_voice_conversion(self, mock_stretch, mock_pitch, mock_synth):
        """When kits_api_key and voice model are provided, Kits.ai conversion should be attempted."""
        sr = 16000
        duration = 0.3
        target_len = int(duration * sr)
        vocals = np.random.randn(sr * 2).astype(np.float32) * 0.3
        fw = _make_flagged("fuck", 0.5, 0.8, matched="fuck")
        mapping = {"fuck": "fudge"}

        tts_audio = np.random.randn(int(sr * 0.5)).astype(np.float32) * 0.1
        mock_synth.return_value = (tts_audio, sr)
        mock_pitch.return_value = tts_audio
        mock_stretch.return_value = np.random.randn(target_len).astype(np.float32) * 0.1

        with patch("sanitune.kits_client.convert_voice", return_value=tts_audio) as mock_kits:
            result = generate_replacement(
                fw, vocals, sr, mapping, language="en",
                kits_api_key="test-key", kits_voice_model_id=123,
            )
            mock_kits.assert_called_once()

        assert result is not None

    @patch("sanitune.tts.synthesize")
    @patch("sanitune.replacer._match_pitch_contour")
    @patch("sanitune.replacer._time_stretch")
    def test_kits_failure_falls_back(self, mock_stretch, mock_pitch, mock_synth):
        """Kits.ai failure should not prevent replacement generation."""
        sr = 16000
        duration = 0.3
        target_len = int(duration * sr)
        vocals = np.random.randn(sr * 2).astype(np.float32) * 0.3
        fw = _make_flagged("fuck", 0.5, 0.8, matched="fuck")
        mapping = {"fuck": "fudge"}

        tts_audio = np.random.randn(int(sr * 0.5)).astype(np.float32) * 0.1
        mock_synth.return_value = (tts_audio, sr)
        mock_pitch.return_value = tts_audio
        mock_stretch.return_value = np.random.randn(target_len).astype(np.float32) * 0.1

        with patch("sanitune.kits_client.convert_voice", side_effect=RuntimeError("API error")):
            result = generate_replacement(
                fw, vocals, sr, mapping, language="en",
                kits_api_key="test-key", kits_voice_model_id=123,
            )

        assert result is not None

    @patch("sanitune.tts.synthesize")
    @patch("sanitune.replacer._match_pitch_contour")
    @patch("sanitune.replacer._time_stretch")
    def test_seed_vc_conversion(self, mock_stretch, mock_pitch, mock_synth):
        """When reference_audio is provided and Seed-VC is available, voice conversion should be used."""
        sr = 16000
        duration = 0.3
        target_len = int(duration * sr)
        vocals = np.random.randn(sr * 2).astype(np.float32) * 0.3
        fw = _make_flagged("fuck", 0.5, 0.8, matched="fuck")
        mapping = {"fuck": "fudge"}
        reference = np.random.randn(sr).astype(np.float32) * 0.3

        tts_audio = np.random.randn(int(sr * 0.5)).astype(np.float32) * 0.1
        mock_synth.return_value = (tts_audio, sr)
        mock_pitch.return_value = tts_audio
        mock_stretch.return_value = np.random.randn(target_len).astype(np.float32) * 0.1

        with patch("sanitune.voice_converter.is_available", return_value=True), \
             patch("sanitune.voice_converter.convert_voice", return_value=tts_audio):
            result = generate_replacement(
                fw, vocals, sr, mapping, language="en",
                reference_audio=reference,
            )

        assert result is not None

    @patch("sanitune.tts.synthesize")
    @patch("sanitune.replacer._match_pitch_contour")
    @patch("sanitune.replacer._time_stretch")
    def test_seed_vc_failure_falls_back(self, mock_stretch, mock_pitch, mock_synth):
        """Seed-VC failure should not prevent replacement generation."""
        sr = 16000
        duration = 0.3
        target_len = int(duration * sr)
        vocals = np.random.randn(sr * 2).astype(np.float32) * 0.3
        fw = _make_flagged("fuck", 0.5, 0.8, matched="fuck")
        mapping = {"fuck": "fudge"}
        reference = np.random.randn(sr).astype(np.float32) * 0.3

        tts_audio = np.random.randn(int(sr * 0.5)).astype(np.float32) * 0.1
        mock_synth.return_value = (tts_audio, sr)
        mock_pitch.return_value = tts_audio
        mock_stretch.return_value = np.random.randn(target_len).astype(np.float32) * 0.1

        with patch("sanitune.voice_converter.is_available", return_value=True), \
             patch("sanitune.voice_converter.convert_voice", side_effect=RuntimeError("VC failed")):
            result = generate_replacement(
                fw, vocals, sr, mapping, language="en",
                reference_audio=reference,
            )

        assert result is not None

    @patch("sanitune.tts.synthesize")
    @patch("sanitune.replacer._match_pitch_contour")
    @patch("sanitune.replacer._time_stretch")
    def test_replacement_padded_when_short(self, mock_stretch, mock_pitch, mock_synth):
        """When TTS generates shorter audio than the target, it should be padded."""
        sr = 16000
        duration = 0.3
        target_len = int(duration * sr)
        vocals = np.random.randn(sr * 2).astype(np.float32) * 0.3
        fw = _make_flagged("fuck", 0.5, 0.8, matched="fuck")
        mapping = {"fuck": "fudge"}

        tts_audio = np.random.randn(int(sr * 0.5)).astype(np.float32) * 0.1
        mock_synth.return_value = (tts_audio, sr)
        mock_pitch.return_value = tts_audio
        # Return shorter audio than target
        mock_stretch.return_value = np.random.randn(target_len // 2).astype(np.float32) * 0.1

        result = generate_replacement(fw, vocals, sr, mapping, language="en")

        assert result is not None
        assert len(result) == target_len


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

    @patch("sanitune.replacer.generate_replacement")
    def test_ai_suggestions_merged(self, mock_gen):
        """AI suggestions should be merged into the mapping."""
        sr = 16000
        vocals = np.ones(sr * 3, dtype=np.float32) * 0.5
        flagged = [_make_flagged("fuck", 0.5, 0.8, matched="fuck")]
        replacement = np.ones(int(0.3 * sr), dtype=np.float32) * 0.2
        mock_gen.return_value = replacement

        result, replaced, muted = replace_words(
            vocals, sr, flagged, language="en",
            ai_suggestions={"fuck": "fudge"},
        )

        assert replaced == 1
        assert muted == 0

    @patch("sanitune.replacer.generate_replacement")
    def test_skip_invalid_bounds(self, mock_gen):
        """Words with start >= end after clamping should be skipped."""
        sr = 16000
        vocals = np.ones(sr, dtype=np.float32) * 0.5
        # Create a flagged word with identical start/end
        flagged = [_make_flagged("fuck", 0.5, 0.5, matched="fuck")]
        mock_gen.return_value = None

        result, replaced, muted = replace_words(vocals, sr, flagged, language="en")
        # Should be skipped (start >= end after clamping)
        assert replaced == 0
        assert muted == 0
