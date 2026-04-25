"""Tests for the web module."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sanitune.web import _parse_word_mappings


class TestParseWordMappings:
    def test_basic_parsing(self):
        result = _parse_word_mappings("fuck=fudge, damn=darn")
        assert result == {"fuck": "fudge", "damn": "darn"}

    def test_newline_separated(self):
        result = _parse_word_mappings("fuck=fudge\ndamn=darn")
        assert result == {"fuck": "fudge", "damn": "darn"}

    def test_empty_string(self):
        result = _parse_word_mappings("")
        assert result == {}

    def test_whitespace_only(self):
        result = _parse_word_mappings("   \n  ")
        assert result == {}

    def test_no_equals_sign(self):
        result = _parse_word_mappings("just some text")
        assert result == {}

    def test_empty_key_or_value_skipped(self):
        result = _parse_word_mappings("=value, key=, good=word")
        assert result == {"good": "word"}

    def test_multiple_equals(self):
        result = _parse_word_mappings("key=value=extra")
        assert result == {"key": "value=extra"}

    def test_keys_lowercased(self):
        result = _parse_word_mappings("FUCK=Fudge")
        assert result == {"fuck": "Fudge"}


class TestProcessAudio:
    def test_no_file_uploaded(self):
        from sanitune.web import _process_audio

        out_path, html, status = _process_audio(
            "", "mute", "en", "edge-tts", "", "", "", True, "", "", 1000, "anthropic", ""
        )
        assert out_path is None
        assert "No file uploaded" in status

    @patch("sanitune.config.detect_device", return_value="cpu")
    def test_auto_detection_off_no_words(self, mock_device):
        from sanitune.web import _process_audio

        out_path, html, status = _process_audio(
            "/fake/path.wav", "mute", "en", "edge-tts",
            "",  # no target words
            "", "", False,  # auto detection off
            "", "", 1000, "anthropic", ""
        )
        assert out_path is None
        assert "Auto-detection is off" in status

    @patch("sanitune.config.detect_device", return_value="cpu")
    def test_replace_mode_no_mappings_no_auto(self, mock_device):
        """When replace mode with invalid mappings (no = sign) and auto detection off, error is returned."""
        from sanitune.web import _process_audio

        out_path, html, status = _process_audio(
            "/fake/path.wav", "replace", "en", "edge-tts",
            "fuck",  # has target words
            "no_equals_sign_here",  # has content but parses to empty mappings
            "", False,
            "", "", 1000, "anthropic", ""
        )
        assert out_path is None
        assert "Replace mode requires" in status

    @patch("sanitune.pipeline.process")
    @patch("sanitune.config.detect_device")
    def test_processing_failure(self, mock_device, mock_process):
        from sanitune.web import _process_audio

        mock_device.return_value = "cpu"
        mock_process.side_effect = RuntimeError("Processing failed")

        out_path, html, status = _process_audio(
            "/fake/path.wav", "mute", "en", "edge-tts",
            "", "", "", True,
            "", "", 1000, "anthropic", ""
        )
        assert out_path is None
        assert "internal error" in status

    @patch("sanitune.pipeline.process")
    @patch("sanitune.config.detect_device")
    def test_successful_processing(self, mock_device, mock_process, tmp_path):
        from sanitune.web import _process_audio
        from sanitune.detector import FlaggedWord
        from sanitune.transcriber import TranscriptionResult, Word

        mock_device.return_value = "cpu"

        words = [
            Word(text="hello", start=0.0, end=0.3),
            Word(text="damn", start=0.3, end=0.6),
            Word(text="world", start=0.6, end=1.0),
        ]
        flagged = [FlaggedWord(word=words[1], matched_term="damn", index=1)]

        mock_process.return_value = SimpleNamespace(
            output_path=tmp_path / "out.wav",
            flagged_words=flagged,
            transcription=TranscriptionResult(
                words=words, language="en", full_text="hello damn world"
            ),
            elapsed_seconds=2.5,
        )

        input_file = tmp_path / "song.wav"
        input_file.write_bytes(b"audio")

        out_path, html, status = _process_audio(
            str(input_file), "mute", "en", "edge-tts",
            "", "", "", True,
            "", "", 1000, "anthropic", ""
        )
        assert out_path is not None
        assert "2.5s" in status
        assert "1 words flagged" in status
        assert "damn" in html

    @patch("sanitune.pipeline.process")
    @patch("sanitune.config.detect_device")
    def test_sentence_deletion_flagging(self, mock_device, mock_process, tmp_path):
        from sanitune.web import _process_audio
        from sanitune.detector import FlaggedWord
        from sanitune.transcriber import TranscriptionResult, Word

        mock_device.return_value = "cpu"

        words = [
            Word(text="kill", start=0.0, end=0.3),
            Word(text="you", start=0.3, end=0.6),
        ]
        sentence_flag = FlaggedWord(
            word=Word(text="kill you", start=0.0, end=0.6, score=1.0),
            matched_term="[sentence] kill you",
            index=-1,
        )

        mock_process.return_value = SimpleNamespace(
            output_path=tmp_path / "out.wav",
            flagged_words=[sentence_flag],
            transcription=TranscriptionResult(
                words=words, language="en", full_text="kill you"
            ),
            elapsed_seconds=1.0,
        )

        input_file = tmp_path / "song.wav"
        input_file.write_bytes(b"audio")

        out_path, html, status = _process_audio(
            str(input_file), "mute", "en", "edge-tts",
            "", "", "kill you", True,
            "", "", 1000, "anthropic", ""
        )
        assert "1 sentences deleted" in status

    @patch("sanitune.pipeline.process")
    @patch("sanitune.config.detect_device")
    def test_replace_mode_with_mappings(self, mock_device, mock_process, tmp_path):
        from sanitune.web import _process_audio
        from sanitune.transcriber import TranscriptionResult, Word

        mock_device.return_value = "cpu"
        mock_process.return_value = SimpleNamespace(
            output_path=tmp_path / "out.wav",
            flagged_words=[],
            transcription=TranscriptionResult(
                words=[], language="en", full_text=""
            ),
            elapsed_seconds=0.5,
        )

        input_file = tmp_path / "song.wav"
        input_file.write_bytes(b"audio")

        out_path, html, status = _process_audio(
            str(input_file), "replace", "en", "edge-tts",
            "fuck", "fuck=fudge", "", True,
            "", "", 1000, "anthropic", ""
        )
        assert out_path is not None

    @patch("sanitune.pipeline.process")
    @patch("sanitune.config.detect_device")
    def test_ai_suggestions_passed(self, mock_device, mock_process, tmp_path):
        from sanitune.web import _process_audio
        from sanitune.transcriber import TranscriptionResult

        mock_device.return_value = "cpu"
        mock_process.return_value = SimpleNamespace(
            output_path=tmp_path / "out.wav",
            flagged_words=[],
            transcription=TranscriptionResult(words=[], language="en", full_text=""),
            elapsed_seconds=0.5,
        )

        input_file = tmp_path / "song.wav"
        input_file.write_bytes(b"audio")

        _process_audio(
            str(input_file), "mute", "en", "edge-tts",
            "", "", "", True,
            "Artist", "Title", 1000, "anthropic", "sk-test-key"
        )

        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["ai_provider"] == "anthropic"
        assert call_kwargs["ai_api_key"] == "sk-test-key"


class TestCreateApp:
    def test_create_app_raises_without_gradio(self, monkeypatch):
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "gradio":
                raise ImportError("No module named 'gradio'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from sanitune.web import create_app

        with pytest.raises(ImportError, match="Gradio is required"):
            create_app()
