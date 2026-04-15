"""Tests for the config module."""

import pytest

from sanitune.config import VALID_DEVICES, Settings, detect_device


def test_detect_device_returns_override():
    assert detect_device("cpu") == "cpu"
    assert detect_device("cuda") == "cuda"
    assert detect_device("mps") == "mps"


def test_detect_device_rejects_invalid():
    with pytest.raises(ValueError, match="Unknown device"):
        detect_device("gpu")

    with pytest.raises(ValueError, match="Unknown device"):
        detect_device("CUDA")


def test_detect_device_auto_returns_valid():
    pytest.importorskip("torch", reason="torch required for auto device detection")
    result = detect_device("auto")
    assert result in VALID_DEVICES


def test_valid_devices_set():
    assert VALID_DEVICES == {"cpu", "cuda", "mps"}


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("SANITUNE_DEVICE", "cpu")
    monkeypatch.setenv("SANITUNE_LANGUAGE", "es")
    monkeypatch.setenv("SANITUNE_DEFAULT_MODE", "bleep")
    monkeypatch.setenv("SANITUNE_MAX_FILE_SIZE", "321")
    monkeypatch.setenv("SANITUNE_BLEEP_FREQ", "777")

    settings = Settings.from_env()

    assert settings.device == "cpu"
    assert settings.language == "es"
    assert settings.default_mode == "bleep"
    assert settings.max_file_size_mb == 321
    assert settings.bleep_freq == 777


def test_settings_from_env_rejects_invalid_mode(monkeypatch):
    monkeypatch.setenv("SANITUNE_DEFAULT_MODE", "invalid_mode")

    with pytest.raises(ValueError, match="SANITUNE_DEFAULT_MODE"):
        Settings.from_env()


def test_settings_from_env_accepts_replace_mode(monkeypatch):
    monkeypatch.setenv("SANITUNE_DEFAULT_MODE", "replace")

    settings = Settings.from_env()
    assert settings.default_mode == "replace"
