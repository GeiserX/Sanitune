"""Tests for the transcriber module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from sanitune.transcriber import Segment, TranscriptionResult, Word


class TestWord:
    def test_word_default_score(self):
        word = Word(text="hello", start=0.0, end=0.5)
        assert word.score == 1.0

    def test_word_custom_score(self):
        word = Word(text="world", start=0.5, end=1.0, score=0.9)
        assert word.score == 0.9

    def test_word_fields(self):
        word = Word(text="test", start=1.5, end=2.0)
        assert word.text == "test"
        assert word.start == 1.5
        assert word.end == 2.0


class TestSegment:
    def test_segment_fields(self):
        words = [Word(text="hello", start=0.0, end=0.3)]
        seg = Segment(text="hello", start=0.0, end=0.3, words=words)
        assert seg.text == "hello"
        assert len(seg.words) == 1


class TestTranscriptionResult:
    def test_default_segments(self):
        result = TranscriptionResult(words=[], language="en", full_text="")
        assert result.segments == []

    def test_with_segments(self):
        words = [Word(text="hi", start=0.0, end=0.3)]
        segs = [Segment(text="hi", start=0.0, end=0.3, words=words)]
        result = TranscriptionResult(
            words=words, language="en", full_text="hi", segments=segs
        )
        assert len(result.segments) == 1
        assert result.language == "en"
        assert result.full_text == "hi"

    def test_post_init_none_segments(self):
        result = TranscriptionResult(
            words=[], language="es", full_text="", segments=None
        )
        assert result.segments == []


class TestTranscribe:
    def test_transcribe_with_mocked_whisperx(self, tmp_path):
        """Test the full transcribe function with mocked whisperx."""
        # Build mock modules
        mock_torch = MagicMock()
        mock_torch.load = MagicMock()

        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {
            "segments": [
                {
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.3, "score": 0.95},
                        {"word": "world", "start": 0.3, "end": 0.6, "score": 0.90},
                    ],
                }
            ]
        }
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {
            "segments": [
                {
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.3, "score": 0.95},
                        {"word": "world", "start": 0.3, "end": 0.6, "score": 0.90},
                    ],
                }
            ]
        }

        audio = np.random.randn(16000).astype(np.float32) * 0.1

        with patch.dict(sys.modules, {"torch": mock_torch, "whisperx": mock_whisperx}):
            # Force fresh import
            import importlib

            import sanitune.transcriber as transcriber_mod
            importlib.reload(transcriber_mod)

            result = transcriber_mod.transcribe(audio, 16000, device="cpu", language="en")

            assert len(result.words) == 2
            assert result.words[0].text == "hello"
            assert result.words[1].text == "world"
            assert result.language == "en"
            assert len(result.segments) == 1
            assert "hello" in result.full_text

    def test_transcribe_stereo_input(self, tmp_path):
        """Stereo input should be downmixed to mono."""
        mock_torch = MagicMock()
        mock_torch.load = MagicMock()

        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"segments": []}
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {"segments": []}

        stereo_audio = np.random.randn(16000, 2).astype(np.float32) * 0.1

        with patch.dict(sys.modules, {"torch": mock_torch, "whisperx": mock_whisperx}):
            import importlib

            import sanitune.transcriber as transcriber_mod
            importlib.reload(transcriber_mod)

            result = transcriber_mod.transcribe(stereo_audio, 16000, device="cpu", language="en")
            assert result.words == []

    def test_transcribe_mps_falls_back_to_cpu(self, tmp_path):
        """MPS device should fall back to CPU for whisperx."""
        mock_torch = MagicMock()
        mock_torch.load = MagicMock()

        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"segments": []}
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {"segments": []}

        audio = np.random.randn(16000).astype(np.float32) * 0.1

        with patch.dict(sys.modules, {"torch": mock_torch, "whisperx": mock_whisperx}):
            import importlib

            import sanitune.transcriber as transcriber_mod
            importlib.reload(transcriber_mod)

            transcriber_mod.transcribe(audio, 16000, device="mps", language="en")
            # Should have fallen back to cpu
            load_call = mock_whisperx.load_model.call_args
            assert load_call[0][1] == "cpu"

    def test_transcribe_no_words_detected(self, tmp_path):
        """No words detected should return empty list and log warning."""
        mock_torch = MagicMock()
        mock_torch.load = MagicMock()

        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"segments": []}
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {"segments": []}

        audio = np.zeros(16000, dtype=np.float32)

        with patch.dict(sys.modules, {"torch": mock_torch, "whisperx": mock_whisperx}):
            import importlib

            import sanitune.transcriber as transcriber_mod
            importlib.reload(transcriber_mod)

            result = transcriber_mod.transcribe(audio, 16000, device="cpu", language="en")
            assert result.words == []
            assert result.full_text == ""

    def test_transcribe_segment_without_words_key(self):
        """Segments with 'word' key instead of 'words' should be handled."""
        mock_torch = MagicMock()
        mock_torch.load = MagicMock()

        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {
            "segments": [
                {"word": "hello", "start": 0.0, "end": 0.3, "score": 0.9}
            ]
        }
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {
            "segments": [
                {"word": "hello", "start": 0.0, "end": 0.3, "score": 0.9}
            ]
        }

        audio = np.random.randn(16000).astype(np.float32) * 0.1

        with patch.dict(sys.modules, {"torch": mock_torch, "whisperx": mock_whisperx}):
            import importlib

            import sanitune.transcriber as transcriber_mod
            importlib.reload(transcriber_mod)

            result = transcriber_mod.transcribe(audio, 16000, device="cpu", language="en")
            assert len(result.words) == 1
            assert result.words[0].text == "hello"

    def test_transcribe_cuda_uses_float16(self):
        """CUDA device should use float16 compute type."""
        mock_torch = MagicMock()
        mock_torch.load = MagicMock()

        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_whisperx.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"segments": []}
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {"segments": []}

        audio = np.random.randn(16000).astype(np.float32) * 0.1

        with patch.dict(sys.modules, {"torch": mock_torch, "whisperx": mock_whisperx}):
            import importlib

            import sanitune.transcriber as transcriber_mod
            importlib.reload(transcriber_mod)

            transcriber_mod.transcribe(audio, 16000, device="cuda", language="en")
            load_call = mock_whisperx.load_model.call_args
            assert load_call[1]["compute_type"] == "float16"
