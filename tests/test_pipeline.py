"""Tests for the pipeline module (validation only — no ML deps needed).

Since pipeline.py transitively imports torch/demucs/whisperx via separator/transcriber,
we test the validation logic by importing just the constants and calling process()
which fails fast on validation before reaching ML code.
"""

import pytest

# Can't import pipeline directly (torch dependency), so test via subprocess or
# test the validation functions indirectly. For now, test the validation behaviors
# that don't require the ML stack by checking the file-level guards.


def test_process_rejects_missing_file(tmp_path):
    """Pipeline should reject non-existent files before touching ML code."""
    pytest.importorskip("torch", reason="torch required for pipeline import")
    from sanitune.pipeline import process

    fake = tmp_path / "nonexistent.mp3"
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        process(fake)


def test_process_rejects_unsupported_extension(tmp_path):
    pytest.importorskip("torch", reason="torch required for pipeline import")
    from sanitune.pipeline import process

    txt_file = tmp_path / "song.txt"
    txt_file.write_text("not audio")
    with pytest.raises(ValueError, match="Unsupported file type"):
        process(txt_file)


def test_process_rejects_oversized_file(tmp_path):
    pytest.importorskip("torch", reason="torch required for pipeline import")
    from sanitune.pipeline import process

    big_file = tmp_path / "big.wav"
    big_file.write_bytes(b"\x00" * (2 * 1024 * 1024))
    with pytest.raises(ValueError, match="exceeding the"):
        process(big_file, max_file_size_mb=1)
