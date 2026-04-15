"""Tests for the pipeline module."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sanitune.lyrics import LyricsResult
from sanitune.pipeline import process
from sanitune.transcriber import TranscriptionResult, Word


def _stub_pipeline(monkeypatch, *, words: list[Word]):
    monkeypatch.setattr("sanitune.pipeline.detect_device", lambda requested: "cpu")
    monkeypatch.setattr(
        "sanitune.pipeline.separate",
        lambda *_args, **_kwargs: SimpleNamespace(
            vocals=np.zeros((1600, 1), dtype=np.float32),
            instrumentals=np.zeros((1600, 1), dtype=np.float32),
            original=np.zeros((1600, 1), dtype=np.float32),
            sample_rate=16000,
        ),
    )
    monkeypatch.setattr(
        "sanitune.pipeline.transcribe",
        lambda *_args, **_kwargs: TranscriptionResult(
            words=words,
            language="en",
            full_text=" ".join(word.text for word in words),
        ),
    )


def test_process_rejects_missing_file(tmp_path):
    fake = tmp_path / "nonexistent.mp3"
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        process(fake)


def test_process_rejects_unsupported_extension(tmp_path):
    txt_file = tmp_path / "song.txt"
    txt_file.write_text("not audio")
    with pytest.raises(ValueError, match="Unsupported input file type"):
        process(txt_file)


def test_process_rejects_oversized_file(tmp_path):
    big_file = tmp_path / "big.wav"
    big_file.write_bytes(b"\x00" * (2 * 1024 * 1024))
    with pytest.raises(ValueError, match="exceeding the"):
        process(big_file, max_file_size_mb=1)


def test_process_defaults_output_to_input_format(tmp_path, monkeypatch):
    input_file = tmp_path / "song.mp3"
    input_file.write_bytes(b"audio")
    _stub_pipeline(monkeypatch, words=[Word(text="hello", start=0.0, end=0.5)])
    monkeypatch.setattr(
        "sanitune.pipeline.detect_audio_format",
        lambda _: {"codec": "mp3", "sample_rate": 44100, "channels": 2, "bit_rate": "320000", "extension": ".mp3"},
    )

    captured = {}

    def fake_remix(_vocals, _instrumentals, _sample_rate, output_path, **_kw):
        captured["output_path"] = output_path
        return output_path

    monkeypatch.setattr("sanitune.pipeline.remix", fake_remix)

    result = process(input_file)

    # MP3 input → MP3 output (format preservation)
    assert result.output_path == tmp_path / "song_clean.mp3"
    assert captured["output_path"].suffix == ".mp3"


def test_process_rejects_unsupported_output_extension(tmp_path, monkeypatch):
    input_file = tmp_path / "song.wav"
    input_file.write_bytes(b"audio")
    monkeypatch.setattr(
        "sanitune.pipeline.detect_audio_format",
        lambda _: {"codec": None, "sample_rate": 44100, "channels": 2, "bit_rate": None, "extension": ".wav"},
    )

    with pytest.raises(ValueError, match="Unsupported output file type"):
        process(input_file, output_path=tmp_path / "song_clean.wma")


def test_process_lyrics_alignment_adds_flagged_word(tmp_path, monkeypatch):
    input_file = tmp_path / "song.wav"
    input_file.write_bytes(b"audio")
    _stub_pipeline(
        monkeypatch,
        words=[
            Word(text="ship", start=0.0, end=0.4),
            Word(text="song", start=0.4, end=0.8),
        ],
    )
    monkeypatch.setattr(
        "sanitune.pipeline.detect_audio_format",
        lambda _: {"codec": None, "sample_rate": 44100, "channels": 2, "bit_rate": None, "extension": ".wav"},
    )
    monkeypatch.setattr(
        "sanitune.lyrics.fetch_lyrics",
        lambda *_args, **_kwargs: LyricsResult(
            text="shit song",
            provider="test",
        ),
    )
    monkeypatch.setattr("sanitune.pipeline.remix", lambda *_args, **_kwargs: input_file.with_name("song_clean.wav"))

    result = process(input_file, artist="Artist", title="Title")

    assert len(result.flagged_words) == 1
    assert result.flagged_words[0].word.text == "ship"
    assert result.flagged_words[0].matched_term == "shit"
