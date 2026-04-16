"""Configuration and hardware detection."""

from __future__ import annotations

import os
from dataclasses import dataclass

VALID_DEVICES = {"cpu", "cuda", "mps"}
VALID_DEVICE_OPTIONS = {"auto", *VALID_DEVICES}
VALID_MODES = {"mute", "bleep", "replace"}


def _parse_positive_int(name: str, default: str) -> int:
    raw = os.environ.get(name, default)
    try:
        value = int(raw)
    except ValueError as err:
        raise ValueError(f"Environment variable {name}='{raw}' is not a valid integer") from err
    if value <= 0:
        raise ValueError(f"Environment variable {name}={value} must be a positive integer")
    return value


def _parse_choice(name: str, default: str, valid: set[str]) -> str:
    raw = os.environ.get(name, default).strip().lower()
    if raw not in valid:
        valid_values = ", ".join(sorted(valid))
        raise ValueError(f"Environment variable {name}='{raw}' must be one of: {valid_values}")
    return raw


def _parse_language(name: str, default: str) -> str:
    raw = os.environ.get(name, default).strip().lower()
    if not raw:
        raise ValueError(f"Environment variable {name} must not be empty")
    return raw


@dataclass
class Settings:
    device: str = "auto"
    language: str = "en"
    default_mode: str = "mute"
    max_file_size_mb: int = 200
    bleep_freq: int = 1000

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            device=_parse_choice("SANITUNE_DEVICE", "auto", VALID_DEVICE_OPTIONS),
            language=_parse_language("SANITUNE_LANGUAGE", "en"),
            default_mode=_parse_choice("SANITUNE_DEFAULT_MODE", "mute", VALID_MODES),
            max_file_size_mb=_parse_positive_int("SANITUNE_MAX_FILE_SIZE", "200"),
            bleep_freq=_parse_positive_int("SANITUNE_BLEEP_FREQ", "1000"),
        )


def detect_device(requested: str = "auto") -> str:
    """Detect the best available compute device.

    Priority: user override > CUDA > MPS > CPU.
    """
    if requested != "auto":
        if requested not in VALID_DEVICES:
            raise ValueError(f"Unknown device '{requested}'. Valid: auto, {', '.join(sorted(VALID_DEVICES))}")
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"
