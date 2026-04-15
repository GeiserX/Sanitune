"""Configuration and hardware detection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Settings:
    device: str = "auto"
    language: str = "en"
    default_mode: str = "mute"
    model_dir: Path = field(default_factory=lambda: Path(os.environ.get("SANITUNE_MODEL_DIR", "./models")))
    max_file_size_mb: int = 200
    bleep_freq: int = 1000
    llm_api_key: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            device=os.environ.get("SANITUNE_DEVICE", "auto"),
            language=os.environ.get("SANITUNE_LANGUAGE", "en"),
            default_mode=os.environ.get("SANITUNE_DEFAULT_MODE", "mute"),
            model_dir=Path(os.environ.get("SANITUNE_MODEL_DIR", "./models")),
            max_file_size_mb=int(os.environ.get("SANITUNE_MAX_FILE_SIZE", "200")),
            bleep_freq=int(os.environ.get("SANITUNE_BLEEP_FREQ", "1000")),
            llm_api_key=os.environ.get("SANITUNE_LLM_API_KEY"),
        )


def detect_device(requested: str = "auto") -> str:
    """Detect the best available compute device.

    Priority: user override > CUDA > MPS > CPU.
    """
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"
