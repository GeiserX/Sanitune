"""Word-level transcription using WhisperX."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class Word:
    text: str
    start: float
    end: float
    score: float = 1.0


@dataclass
class TranscriptionResult:
    words: list[Word]
    language: str
    full_text: str


def transcribe(
    vocals: np.ndarray,
    sample_rate: int,
    *,
    device: str = "cpu",
    language: str = "en",
) -> TranscriptionResult:
    """Transcribe vocals with word-level timestamps using WhisperX.

    Args:
        vocals: Vocal audio as numpy array (samples, channels) or (samples,).
        sample_rate: Sample rate of the audio.
        device: Compute device (cpu, cuda, mps).
        language: Language code for transcription.

    Returns:
        TranscriptionResult with word-level timestamps.
    """
    import whisperx

    compute_type = "float16" if device == "cuda" else "float32"
    if device == "mps":
        # WhisperX/CTranslate2 doesn't support MPS; fall back to CPU for transcription
        logger.info("MPS not supported by CTranslate2, falling back to CPU for transcription")
        device = "cpu"

    logger.info("Loading WhisperX model on %s (%s)...", device, compute_type)
    model = whisperx.load_model("large-v3", device, compute_type=compute_type, language=language)

    # WhisperX expects a file path or mono float32 numpy array
    if vocals.ndim == 2:
        audio_mono = vocals.mean(axis=1).astype(np.float32)
    else:
        audio_mono = vocals.astype(np.float32)

    # Write to temp wav for WhisperX (it handles resampling internally)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        sf.write(tmp_path, audio_mono, sample_rate)

    logger.info("Transcribing...")
    result = model.transcribe(str(tmp_path), language=language)

    # Align for word-level timestamps
    logger.info("Aligning word timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, str(tmp_path), device)

    tmp_path.unlink(missing_ok=True)

    # Extract words
    words = []
    full_text_parts = []
    for segment in result.get("word_segments", result.get("segments", [])):
        if "words" in segment:
            for w in segment["words"]:
                words.append(Word(
                    text=w.get("word", "").strip(),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    score=w.get("score", 1.0),
                ))
        elif "word" in segment:
            words.append(Word(
                text=segment["word"].strip(),
                start=segment.get("start", 0.0),
                end=segment.get("end", 0.0),
                score=segment.get("score", 1.0),
            ))
        if "text" in segment:
            full_text_parts.append(segment["text"])

    full_text = " ".join(full_text_parts) if full_text_parts else " ".join(w.text for w in words)
    logger.info("Transcription complete: %d words detected", len(words))
    return TranscriptionResult(words=words, language=language, full_text=full_text)
