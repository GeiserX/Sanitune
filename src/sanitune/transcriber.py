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
class Segment:
    text: str
    start: float
    end: float
    words: list[Word]


@dataclass
class TranscriptionResult:
    words: list[Word]
    segments: list[Segment]
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
    import torch
    import whisperx

    # PyTorch 2.6+ defaults weights_only=True for torch.load, but pyannote-audio's
    # VAD checkpoint uses omegaconf types not in the safe list. Temporarily patch
    # torch.load to default weights_only=False during model loading.
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load

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

    try:
        logger.info("Transcribing...")
        result = model.transcribe(str(tmp_path), language=language)

        # Align for word-level timestamps
        logger.info("Aligning word timestamps...")
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, str(tmp_path), device)
    finally:
        tmp_path.unlink(missing_ok=True)
        # Restore original torch.load after all models are loaded
        torch.load = _original_torch_load

    # Extract words and segments
    words: list[Word] = []
    segments: list[Segment] = []
    full_text_parts: list[str] = []

    # WhisperX aligned output has "segments" with nested "words"
    raw_segments = result.get("segments", [])
    # word_segments is a flat list; prefer segments for sentence grouping
    for seg in raw_segments:
        seg_words: list[Word] = []
        if "words" in seg:
            for w in seg["words"]:
                word = Word(
                    text=w.get("word", "").strip(),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    score=w.get("score", 1.0),
                )
                words.append(word)
                seg_words.append(word)
        elif "word" in seg:
            word = Word(
                text=seg["word"].strip(),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                score=seg.get("score", 1.0),
            )
            words.append(word)
            seg_words.append(word)

        seg_text = seg.get("text", " ".join(w.text for w in seg_words)).strip()
        if seg_text:
            full_text_parts.append(seg_text)
        if seg_words:
            segments.append(Segment(
                text=seg_text,
                start=seg_words[0].start,
                end=seg_words[-1].end,
                words=seg_words,
            ))

    full_text = " ".join(full_text_parts) if full_text_parts else " ".join(w.text for w in words)
    if not words:
        logger.warning("No words detected in transcription. The output will be identical to the input.")
    logger.info("Transcription complete: %d words, %d segments detected", len(words), len(segments))
    return TranscriptionResult(words=words, segments=segments, language=language, full_text=full_text)
