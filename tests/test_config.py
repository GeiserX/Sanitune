"""Tests for the config module."""

import pytest

from sanitune.config import VALID_DEVICES, detect_device


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
