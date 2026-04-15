"""Tests for the CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from sanitune.cli import main


def test_cli_uses_env_defaults(monkeypatch):
    captured = {}

    def fake_run_pipeline(input_file, output, **kwargs):
        captured["input_file"] = input_file
        captured["output"] = output
        captured.update(kwargs)
        return SimpleNamespace(output_path=Path("out.wav"), flagged_words=[], elapsed_seconds=1.2)

    monkeypatch.setenv("SANITUNE_DEVICE", "cpu")
    monkeypatch.setenv("SANITUNE_LANGUAGE", "es")
    monkeypatch.setenv("SANITUNE_DEFAULT_MODE", "bleep")
    monkeypatch.setenv("SANITUNE_MAX_FILE_SIZE", "321")
    monkeypatch.setenv("SANITUNE_BLEEP_FREQ", "777")
    monkeypatch.setattr("sanitune.pipeline.process", fake_run_pipeline)

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("song.wav").write_bytes(b"audio")
        result = runner.invoke(main, ["process", "song.wav"])

    assert result.exit_code == 0
    assert captured["mode"] == "bleep"
    assert captured["language"] == "es"
    assert captured["device"] == "cpu"
    assert captured["bleep_freq"] == 777
    assert captured["max_file_size_mb"] == 321


def test_cli_options_override_env_defaults(monkeypatch):
    captured = {}

    def fake_run_pipeline(input_file, output, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_path=Path("out.wav"), flagged_words=[], elapsed_seconds=1.2)

    monkeypatch.setenv("SANITUNE_DEVICE", "cpu")
    monkeypatch.setenv("SANITUNE_DEFAULT_MODE", "mute")
    monkeypatch.setattr("sanitune.pipeline.process", fake_run_pipeline)

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("song.wav").write_bytes(b"audio")
        result = runner.invoke(
            main,
            [
                "process",
                "song.wav",
                "--mode",
                "bleep",
                "--device",
                "mps",
                "--max-file-size",
                "111",
            ],
        )

    assert result.exit_code == 0
    assert captured["mode"] == "bleep"
    assert captured["device"] == "mps"
    assert captured["max_file_size_mb"] == 111


def test_cli_reports_invalid_env_config(monkeypatch):
    monkeypatch.setenv("SANITUNE_DEFAULT_MODE", "replace")

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("song.wav").write_bytes(b"audio")
        result = runner.invoke(main, ["process", "song.wav"])

    assert result.exit_code != 0
    assert "SANITUNE_DEFAULT_MODE" in result.output
