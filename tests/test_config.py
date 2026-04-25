"""Tests for the config module."""

import pytest

from sanitune.config import (
    VALID_DEVICES,
    VALID_DEVICE_OPTIONS,
    VALID_MODES,
    VALID_SYNTH_ENGINES,
    Settings,
    _parse_choice,
    _parse_language,
    _parse_positive_int,
    detect_device,
)


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


def test_valid_device_options():
    assert VALID_DEVICE_OPTIONS == {"auto", "cpu", "cuda", "mps"}


def test_valid_modes():
    assert VALID_MODES == {"mute", "bleep", "replace"}


def test_valid_synth_engines():
    assert VALID_SYNTH_ENGINES == {"edge-tts", "bark"}


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


def test_settings_defaults():
    """Default settings should have expected values."""
    s = Settings()
    assert s.device == "auto"
    assert s.language == "en"
    assert s.default_mode == "mute"
    assert s.max_file_size_mb == 200
    assert s.bleep_freq == 1000


# --- Tests for helper parsers ---


def test_parse_positive_int_valid(monkeypatch):
    monkeypatch.setenv("TEST_INT", "42")
    assert _parse_positive_int("TEST_INT", "10") == 42


def test_parse_positive_int_default(monkeypatch):
    monkeypatch.delenv("TEST_INT", raising=False)
    assert _parse_positive_int("TEST_INT", "10") == 10


def test_parse_positive_int_invalid_string(monkeypatch):
    monkeypatch.setenv("TEST_INT", "abc")
    with pytest.raises(ValueError, match="not a valid integer"):
        _parse_positive_int("TEST_INT", "10")


def test_parse_positive_int_zero(monkeypatch):
    monkeypatch.setenv("TEST_INT", "0")
    with pytest.raises(ValueError, match="must be a positive integer"):
        _parse_positive_int("TEST_INT", "10")


def test_parse_positive_int_negative(monkeypatch):
    monkeypatch.setenv("TEST_INT", "-5")
    with pytest.raises(ValueError, match="must be a positive integer"):
        _parse_positive_int("TEST_INT", "10")


def test_parse_choice_valid(monkeypatch):
    monkeypatch.setenv("TEST_CHOICE", "mute")
    assert _parse_choice("TEST_CHOICE", "bleep", {"mute", "bleep"}) == "mute"


def test_parse_choice_default(monkeypatch):
    monkeypatch.delenv("TEST_CHOICE", raising=False)
    assert _parse_choice("TEST_CHOICE", "bleep", {"mute", "bleep"}) == "bleep"


def test_parse_choice_invalid(monkeypatch):
    monkeypatch.setenv("TEST_CHOICE", "invalid")
    with pytest.raises(ValueError, match="must be one of"):
        _parse_choice("TEST_CHOICE", "bleep", {"mute", "bleep"})


def test_parse_choice_case_insensitive(monkeypatch):
    monkeypatch.setenv("TEST_CHOICE", "MUTE")
    assert _parse_choice("TEST_CHOICE", "bleep", {"mute", "bleep"}) == "mute"


def test_parse_language_valid(monkeypatch):
    monkeypatch.setenv("TEST_LANG", "es")
    assert _parse_language("TEST_LANG", "en") == "es"


def test_parse_language_default(monkeypatch):
    monkeypatch.delenv("TEST_LANG", raising=False)
    assert _parse_language("TEST_LANG", "en") == "en"


def test_parse_language_empty(monkeypatch):
    monkeypatch.setenv("TEST_LANG", "")
    with pytest.raises(ValueError, match="must not be empty"):
        _parse_language("TEST_LANG", "en")


def test_parse_language_whitespace_only(monkeypatch):
    monkeypatch.setenv("TEST_LANG", "   ")
    with pytest.raises(ValueError, match="must not be empty"):
        _parse_language("TEST_LANG", "en")
