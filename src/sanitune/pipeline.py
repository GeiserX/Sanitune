"""Full processing pipeline — orchestrates separation, transcription, detection, editing, and remixing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from sanitune.config import detect_device
from sanitune.detector import FlaggedWord, detect
from sanitune.editor import edit
from sanitune.remixer import remix
from sanitune.separator import separate
from sanitune.transcriber import TranscriptionResult, transcribe

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    output_path: Path
    flagged_words: list[FlaggedWord]
    transcription: TranscriptionResult
    elapsed_seconds: float


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


def process(
    input_path: Path,
    output_path: Path | None = None,
    *,
    mode: str = "mute",
    language: str = "en",
    device: str = "auto",
    custom_words: list[str] | None = None,
    exclude_words: list[str] | None = None,
    bleep_freq: int = 1000,
    model_name: str = "htdemucs_ft",
    max_file_size_mb: int = 200,
    artist: str | None = None,
    title: str | None = None,
    genius_api_key: str | None = None,
) -> PipelineResult:
    """Run the full Sanitune pipeline on an audio file.

    Args:
        input_path: Path to the input audio file.
        output_path: Path for the output file. Auto-generated if None.
        mode: Editing mode — 'mute' or 'bleep'.
        language: Language code for transcription and word list.
        device: Compute device (auto, cpu, cuda, mps).
        custom_words: Additional words to flag.
        exclude_words: Words to skip even if in word list.
        bleep_freq: Frequency for bleep tone in Hz.
        model_name: Demucs model name.
        artist: Artist name for lyrics lookup (optional).
        title: Song title for lyrics lookup (optional).
        genius_api_key: Genius API key for lyrics (optional).

    Returns:
        PipelineResult with output path, flagged words, and timing.
    """
    start_time = time.monotonic()

    # Validate input file
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{input_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        raise ValueError(
            f"Input file is {file_size_mb:.1f} MB, exceeding the "
            f"{max_file_size_mb} MB limit. Set --max-file-size or SANITUNE_MAX_FILE_SIZE to override."
        )

    resolved_device = detect_device(device)
    logger.info("Starting pipeline: %s → mode=%s, lang=%s, device=%s", input_path.name, mode, language, resolved_device)

    if output_path is None:
        suffix = input_path.suffix or ".wav"
        output_path = input_path.with_name(f"{input_path.stem}_clean{suffix}")

    # Step 1: Separate vocals from instrumentals
    logger.info("[1/5] Separating vocals...")
    sep_result = separate(input_path, device=resolved_device, model_name=model_name)

    # Step 2: Transcribe vocals
    logger.info("[2/5] Transcribing vocals...")
    trans_result = transcribe(
        sep_result.vocals,
        sep_result.sample_rate,
        device=resolved_device,
        language=language,
    )

    # Step 2.5 (optional): Cross-reference with official lyrics
    lyrics_words: list[str] = []
    if artist and title:
        try:
            from sanitune.lyrics import fetch_lyrics

            logger.info("[2.5/5] Fetching lyrics for '%s - %s'...", artist, title)
            lyrics_result = fetch_lyrics(artist, title, genius_api_key=genius_api_key)
            if lyrics_result:
                lyrics_words = lyrics_result.words
                logger.info("Found %d words in official lyrics (provider: %s)",
                            len(lyrics_words), lyrics_result.provider)
            else:
                logger.info("No lyrics found, continuing with transcription only")
        except ImportError:
            logger.info("Lyrics providers not installed (pip install sanitune[lyrics]), skipping")

    # Merge lyrics-only profane words into custom_words for detection
    if lyrics_words:
        from sanitune.detector import load_wordlist

        profanity_set = load_wordlist(language)
        lyrics_profane = {w for w in lyrics_words if w in profanity_set}
        if lyrics_profane:
            logger.info("Lyrics confirm %d profane words: %s", len(lyrics_profane), lyrics_profane)

    # Step 3: Detect profanity
    logger.info("[3/5] Detecting profanity...")
    flagged = detect(
        trans_result.words,
        language=language,
        custom_words=custom_words,
        exclude_words=exclude_words,
    )

    # Step 4: Edit vocals
    logger.info("[4/5] Editing vocals (%d words flagged)...", len(flagged))
    edited_vocals = edit(
        sep_result.vocals,
        sep_result.sample_rate,
        flagged,
        mode=mode,
        bleep_freq=bleep_freq,
    )

    # Step 5: Remix
    logger.info("[5/5] Remixing...")
    final_path = remix(
        edited_vocals,
        sep_result.instrumentals,
        sep_result.sample_rate,
        output_path,
    )

    elapsed = time.monotonic() - start_time
    logger.info("Pipeline complete in %.1fs. Output: %s", elapsed, final_path)

    return PipelineResult(
        output_path=final_path,
        flagged_words=flagged,
        transcription=trans_result,
        elapsed_seconds=elapsed,
    )
