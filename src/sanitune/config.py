"""Configuration and hardware detection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


VALID_DEVICES = {"cpu", "cuda", "mps"}
VALID_MODES = {"mute", "bleep"}


def _parse_positive_int(name: str, default: str) -> int:
    raw = os.environ.get(name, default)
    try:
        value = int(raw)
    except ValueError as err:
        raise ValueError(f"Environment variable {name}='{raw}' is not a valid integer") from err
    if value <= 0:
        raise ValueError(f"Environment variable {name}={value} must be a positive integer")
    return value


@dataclass
class Settings:
    device: str = "auto"
    language: str = "en"
    default_mode: str = "mute"
    model_dir: Path = field(default_factory=lambda: Path(os.environ.get("SANITUNE_MODEL_DIR", "./models")))
    max_file_size_mb: int = 200
    bleep_freq: int = 1000
    llm_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            device=os.environ.get("SANITUNE_DEVICE", "auto"),
            language=os.environ.get("SANITUNE_LANGUAGE", "en"),
            default_mode=os.environ.get("SANITUNE_DEFAULT_MODE", "mute"),
            model_dir=Path(os.environ.get("SANITUNE_MODEL_DIR", "./models")),
            max_file_size_mb=_parse_positive_int("SANITUNE_MAX_FILE_SIZE", "200"),
            bleep_freq=_parse_positive_int("SANITUNE_BLEEP_FREQ", "1000"),
            llm_api_key=os.environ.get("SANITUNE_LLM_API_KEY"),
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
