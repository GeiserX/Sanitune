"""Tests for the voice_converter module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sanitune.voice_converter import (
    _extract_loudest_segment,
    extract_reference,
    is_available,
)


class TestIsAvailable:
    def test_returns_false_without_seed_vc(self, monkeypatch):
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "seed_vc_wrapper":
                raise ImportError
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        assert is_available() is False


class TestExtractLoudestSegment:
    def test_short_audio_returns_all(self):
        sr = 16000
        mono = np.random.randn(sr).astype(np.float32)
        result = _extract_loudest_segment(mono, sr, duration=2.0)
        assert len(result) == sr  # Audio shorter than duration

    def test_extracts_correct_length(self):
        sr = 16000
        mono = np.random.randn(sr * 5).astype(np.float32)
        result = _extract_loudest_segment(mono, sr, duration=2.0)
        assert len(result) == sr * 2

    def test_finds_loudest_region(self):
        sr = 16000
        mono = np.zeros(sr * 5, dtype=np.float32)
        # Make region at 3-4s loud
        mono[sr * 3:sr * 4] = 1.0
        result = _extract_loudest_segment(mono, sr, duration=2.0)
        assert np.abs(result).max() > 0.5

    def test_empty_rms(self):
        sr = 16000
        mono = np.zeros(100, dtype=np.float32)
        result = _extract_loudest_segment(mono, sr, duration=2.0)
        assert len(result) == 100

    def test_end_clamp_when_near_end(self):
        """When loudest segment is near end, clamp to end of audio."""
        sr = 16000
        mono = np.zeros(sr * 3, dtype=np.float32)
        # Make the very end loud
        mono[sr * 2:sr * 3] = 1.0
        result = _extract_loudest_segment(mono, sr, duration=2.0)
        assert len(result) == sr * 2
        assert result.dtype == np.float32


class TestExtractReference:
    def test_mono_no_flagged(self):
        sr = 16000
        mono = np.random.randn(sr * 5).astype(np.float32)
        result = extract_reference(mono, sr)
        assert result.dtype == np.float32
        assert len(result) <= sr * 10  # target_duration default

    def test_stereo_input(self):
        sr = 16000
        stereo = np.random.randn(sr * 5, 2).astype(np.float32)
        result = extract_reference(stereo, sr)
        assert result.ndim == 1  # Should be mono

    def test_with_flagged_regions(self):
        sr = 16000
        mono = np.random.randn(sr * 10).astype(np.float32)
        flagged = [(1.0, 1.5), (5.0, 5.5)]
        result = extract_reference(mono, sr, flagged)
        assert result.dtype == np.float32
        assert len(result) > 0

    def test_all_flagged_uses_full_track(self):
        sr = 16000
        mono = np.random.randn(sr * 3).astype(np.float32)
        flagged = [(0.0, 3.0)]
        result = extract_reference(mono, sr, flagged)
        assert len(result) > 0

    def test_empty_flagged_list(self):
        sr = 16000
        mono = np.random.randn(sr * 5).astype(np.float32)
        result = extract_reference(mono, sr, flagged_regions=[])
        assert len(result) > 0

    def test_respects_max_duration(self):
        sr = 16000
        mono = np.random.randn(sr * 60).astype(np.float32)
        result = extract_reference(mono, sr, target_duration=5.0, max_duration=5.0)
        assert len(result) <= sr * 5 + sr

    def test_many_small_flagged_regions(self):
        """Clean regions shorter than 1s are skipped."""
        sr = 16000
        mono = np.random.randn(sr * 5).astype(np.float32) * 0.3
        # Create many close flagged regions leaving < 1s gaps
        flagged = [(i * 0.5, i * 0.5 + 0.4) for i in range(10)]
        result = extract_reference(mono, sr, flagged)
        assert len(result) > 0


class TestGetWrapper:
    def test_raises_without_seed_vc(self, monkeypatch):
        import builtins

        import sanitune.voice_converter as vc

        vc._wrapper = None
        vc._wrapper_device = None

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "seed_vc_wrapper":
                raise ImportError("No module named 'seed_vc_wrapper'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="Seed-VC is required"):
            vc._get_wrapper("cpu")

    def test_wrapper_singleton_reused(self):
        """Same device should reuse the existing wrapper."""
        import sanitune.voice_converter as vc

        mock_wrapper = MagicMock()
        vc._wrapper = mock_wrapper
        vc._wrapper_device = "cpu"

        result = vc._get_wrapper("cpu")
        assert result is mock_wrapper

        # Clean up
        vc._wrapper = None
        vc._wrapper_device = None


class TestConvertVoice:
    def test_convert_voice_with_mocked_seed_vc(self, tmp_path):
        """Test convert_voice with mocked Seed-VC wrapper."""
        import sanitune.voice_converter as vc

        sr = 44100
        tts_audio = np.random.randn(sr).astype(np.float32) * 0.1
        ref_audio = np.random.randn(sr * 3).astype(np.float32) * 0.1
        out_audio = np.random.randn(sr).astype(np.float32) * 0.1

        mock_wrapper = MagicMock()

        # Must be a real generator (types.GeneratorType) for the isinstance check
        def fake_gen(**kwargs):
            yield (sr, out_audio)

        mock_wrapper.convert_voice.return_value = fake_gen()

        vc._wrapper = mock_wrapper
        vc._wrapper_device = "cpu"

        try:
            result = vc.convert_voice(tts_audio, ref_audio, sr, device="cpu")
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
        finally:
            vc._wrapper = None
            vc._wrapper_device = None

    def test_convert_voice_tuple_result(self):
        """When generator yields (sr, audio) tuple."""
        import sanitune.voice_converter as vc

        sr = 44100
        tts_audio = np.random.randn(sr).astype(np.float32) * 0.1
        ref_audio = np.random.randn(sr * 3).astype(np.float32) * 0.1
        out_audio = np.random.randn(sr).astype(np.float32) * 0.1

        mock_wrapper = MagicMock()

        def fake_gen(**kwargs):
            yield (sr, out_audio)

        mock_wrapper.convert_voice.return_value = fake_gen()

        vc._wrapper = mock_wrapper
        vc._wrapper_device = "cpu"

        try:
            result = vc.convert_voice(tts_audio, ref_audio, sr, device="cpu")
            assert isinstance(result, np.ndarray)
        finally:
            vc._wrapper = None
            vc._wrapper_device = None

    def test_convert_voice_non_generator(self):
        """When wrapper returns numpy directly (not a generator)."""
        import sanitune.voice_converter as vc

        sr = 44100
        tts_audio = np.random.randn(sr).astype(np.float32) * 0.1
        ref_audio = np.random.randn(sr * 3).astype(np.float32) * 0.1
        out_audio = np.random.randn(sr).astype(np.float32) * 0.1

        mock_wrapper = MagicMock()
        mock_wrapper.convert_voice.return_value = out_audio

        vc._wrapper = mock_wrapper
        vc._wrapper_device = "cpu"

        try:
            result = vc.convert_voice(tts_audio, ref_audio, sr, device="cpu")
            assert isinstance(result, np.ndarray)
        finally:
            vc._wrapper = None
            vc._wrapper_device = None

    def test_convert_voice_multi_dim_output(self):
        """Multi-dimensional output should be handled."""
        import sanitune.voice_converter as vc

        sr = 44100
        tts_audio = np.random.randn(sr).astype(np.float32) * 0.1
        ref_audio = np.random.randn(sr * 3).astype(np.float32) * 0.1
        out_audio = np.random.randn(sr, 2).astype(np.float32) * 0.1

        mock_wrapper = MagicMock()
        mock_wrapper.convert_voice.return_value = out_audio

        vc._wrapper = mock_wrapper
        vc._wrapper_device = "cpu"

        try:
            result = vc.convert_voice(tts_audio, ref_audio, sr, device="cpu")
            assert result.ndim == 1  # Should be reduced to mono
        finally:
            vc._wrapper = None
            vc._wrapper_device = None

    def test_convert_voice_dtype_conversion(self):
        """Float64 output should be converted to float32."""
        import sanitune.voice_converter as vc

        sr = 44100
        tts_audio = np.random.randn(sr).astype(np.float32) * 0.1
        ref_audio = np.random.randn(sr * 3).astype(np.float32) * 0.1
        out_audio = np.random.randn(sr).astype(np.float64) * 0.1  # float64

        mock_wrapper = MagicMock()
        mock_wrapper.convert_voice.return_value = out_audio

        vc._wrapper = mock_wrapper
        vc._wrapper_device = "cpu"

        try:
            result = vc.convert_voice(tts_audio, ref_audio, sr, device="cpu")
            assert result.dtype == np.float32
        finally:
            vc._wrapper = None
            vc._wrapper_device = None

    def test_convert_voice_resample_output(self):
        """When Seed-VC outputs at different sample rate, should resample."""
        import sanitune.voice_converter as vc

        target_sr = 16000
        out_sr = 44100
        tts_audio = np.random.randn(target_sr).astype(np.float32) * 0.1
        ref_audio = np.random.randn(target_sr * 3).astype(np.float32) * 0.1
        out_audio = np.random.randn(out_sr).astype(np.float32) * 0.1

        mock_wrapper = MagicMock()

        def fake_gen(**kwargs):
            yield (out_sr, out_audio)

        mock_wrapper.convert_voice.return_value = fake_gen()

        vc._wrapper = mock_wrapper
        vc._wrapper_device = "cpu"

        resampled = np.random.randn(target_sr).astype(np.float32) * 0.1
        mock_librosa = MagicMock()
        mock_librosa.resample.return_value = resampled

        try:
            with patch.dict("sys.modules", {"librosa": mock_librosa}):
                result = vc.convert_voice(tts_audio, ref_audio, target_sr, device="cpu")
                mock_librosa.resample.assert_called_once()
                assert result.dtype == np.float32
        finally:
            vc._wrapper = None
            vc._wrapper_device = None

    def test_convert_voice_different_device_reinits(self):
        """Changing device should reinitialize the wrapper."""
        import sanitune.voice_converter as vc

        old_wrapper = MagicMock()
        vc._wrapper = old_wrapper
        vc._wrapper_device = "cpu"

        try:
            # Request with a different device should try to reinit
            # Since seed_vc_wrapper isn't installed, it will raise ImportError
            with pytest.raises(ImportError, match="Seed-VC is required"):
                vc._get_wrapper("cuda")
        finally:
            vc._wrapper = None
            vc._wrapper_device = None
