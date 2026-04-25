"""Tests for the separator module."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from sanitune.separator import SeparationResult, _load_audio, _SOUNDFILE_FORMATS


class TestLoadAudio:
    def test_load_wav_file(self, tmp_path):
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32) * 0.3
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio, sr)

        data, loaded_sr = _load_audio(wav_path)
        assert loaded_sr == sr
        assert len(data) == len(audio)

    def test_load_flac_file(self, tmp_path):
        sr = 44100
        audio = np.random.randn(sr, 2).astype(np.float32) * 0.3
        flac_path = tmp_path / "test.flac"
        sf.write(str(flac_path), audio, sr)

        data, loaded_sr = _load_audio(flac_path)
        assert loaded_sr == sr
        assert data.ndim == 2

    @patch("sanitune.separator.subprocess.run")
    def test_load_mp3_uses_ffmpeg(self, mock_run, tmp_path):
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32) * 0.3
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")

        mock_run.return_value = SimpleNamespace(stdout=buf.getvalue(), returncode=0)

        mp3_path = tmp_path / "test.mp3"
        mp3_path.write_bytes(b"fake mp3")

        data, loaded_sr = _load_audio(mp3_path)
        assert loaded_sr == sr
        mock_run.assert_called_once()

    def test_soundfile_formats_set(self):
        assert ".wav" in _SOUNDFILE_FORMATS
        assert ".flac" in _SOUNDFILE_FORMATS
        assert ".mp3" not in _SOUNDFILE_FORMATS


class TestSeparationResult:
    def test_dataclass_fields(self):
        result = SeparationResult(
            vocals=np.zeros(100, dtype=np.float32),
            instrumentals=np.zeros(100, dtype=np.float32),
            original=np.zeros(100, dtype=np.float32),
            sample_rate=16000,
        )
        assert result.sample_rate == 16000
        assert len(result.vocals) == 100


class TestSeparate:
    def test_separate_with_mocked_demucs(self, tmp_path):
        """Test the separate function with all ML deps mocked."""
        sr = 44100
        audio = np.random.randn(sr, 2).astype(np.float32) * 0.3

        # Create mock torch
        mock_torch = MagicMock()
        mock_torch.device.return_value = MagicMock()
        mock_tensor = MagicMock()
        mock_torch.from_numpy.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.__getitem__ = MagicMock(return_value=mock_tensor)
        mock_tensor.numpy.return_value = audio.T

        # Create mock torchaudio
        mock_torchaudio = MagicMock()

        # Create mock model
        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ["drums", "bass", "other", "vocals"]

        # apply_model output
        vocal_arr = np.random.randn(2, sr).astype(np.float32) * 0.1
        instr_arr = np.random.randn(2, sr).astype(np.float32) * 0.1

        mock_sources = MagicMock()
        vocal_out = MagicMock()
        vocal_out.cpu.return_value.numpy.return_value = vocal_arr

        instr_out = MagicMock()
        instr_out.sum.return_value = instr_out
        instr_out.cpu.return_value.numpy.return_value = instr_arr

        def getitem(key):
            if isinstance(key, tuple) and key == (0, 3):
                return vocal_out
            return instr_out

        mock_sources.__getitem__ = MagicMock(side_effect=getitem)

        mock_demucs_apply = MagicMock()
        mock_demucs_apply.apply_model.return_value = mock_sources

        mock_demucs_pretrained = MagicMock()
        mock_demucs_pretrained.get_model.return_value = mock_model

        wav_path = tmp_path / "song.wav"
        sf.write(str(wav_path), audio, sr)

        with patch("sanitune.separator._load_audio", return_value=(audio, sr)), \
             patch.dict(sys.modules, {
                 "torch": mock_torch,
                 "torchaudio": mock_torchaudio,
                 "torchaudio.transforms": MagicMock(),
                 "demucs": MagicMock(),
                 "demucs.apply": mock_demucs_apply,
                 "demucs.pretrained": mock_demucs_pretrained,
             }):
            import importlib
            import sanitune.separator as sep_mod
            importlib.reload(sep_mod)

            result = sep_mod.separate(wav_path, device="cpu")
            assert result.sample_rate == sr
            mock_demucs_pretrained.get_model.assert_called_once()

    def test_separate_mono_input(self, tmp_path):
        """Mono input should be handled by expanding to (1, samples)."""
        sr = 44100
        audio = np.random.randn(sr).astype(np.float32) * 0.3

        mock_torch = MagicMock()
        mock_torch.device.return_value = MagicMock()
        mock_tensor = MagicMock()
        mock_torch.from_numpy.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.__getitem__ = MagicMock(return_value=mock_tensor)
        mock_tensor.numpy.return_value = audio[None]

        mock_torchaudio = MagicMock()

        mock_model = MagicMock()
        mock_model.samplerate = sr
        mock_model.sources = ["drums", "bass", "other", "vocals"]

        vocal_arr = np.random.randn(1, sr).astype(np.float32) * 0.1
        instr_arr = np.random.randn(1, sr).astype(np.float32) * 0.1

        mock_sources = MagicMock()
        vocal_out = MagicMock()
        vocal_out.cpu.return_value.numpy.return_value = vocal_arr
        instr_out = MagicMock()
        instr_out.sum.return_value = instr_out
        instr_out.cpu.return_value.numpy.return_value = instr_arr

        def getitem(key):
            if isinstance(key, tuple) and key == (0, 3):
                return vocal_out
            return instr_out

        mock_sources.__getitem__ = MagicMock(side_effect=getitem)

        mock_demucs_apply = MagicMock()
        mock_demucs_apply.apply_model.return_value = mock_sources
        mock_demucs_pretrained = MagicMock()
        mock_demucs_pretrained.get_model.return_value = mock_model

        wav_path = tmp_path / "song.wav"
        sf.write(str(wav_path), audio, sr)

        with patch("sanitune.separator._load_audio", return_value=(audio, sr)), \
             patch.dict(sys.modules, {
                 "torch": mock_torch,
                 "torchaudio": mock_torchaudio,
                 "torchaudio.transforms": MagicMock(),
                 "demucs": MagicMock(),
                 "demucs.apply": mock_demucs_apply,
                 "demucs.pretrained": mock_demucs_pretrained,
             }):
            import importlib
            import sanitune.separator as sep_mod
            importlib.reload(sep_mod)

            result = sep_mod.separate(wav_path, device="cpu")
            assert result.sample_rate == sr
