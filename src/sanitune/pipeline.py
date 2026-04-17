"""Full processing pipeline — orchestrates separation, transcription, detection, editing, and remixing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from sanitune.config import Settings, detect_device
from sanitune.detector import FlaggedWord, build_profanity_set, clean_word, detect, detect_sentences, match_word
from sanitune.editor import edit
from sanitune.remixer import SUPPORTED_OUTPUT_EXTENSIONS, detect_audio_format, remix
from sanitune.separator import separate
from sanitune.transcriber import TranscriptionResult, Word, transcribe

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    output_path: Path
    flagged_words: list[FlaggedWord]
    transcription: TranscriptionResult
    elapsed_seconds: float


SUPPORTED_INPUT_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


def _merge_lyrics_reference_flags(
    words: list[Word],
    lyrics_words: list[str],
    profanity_set: set[str],
    existing_indices: set[int],
) -> list[FlaggedWord]:
    """Use rough token alignment to recover profane words missed by transcription."""
    transcribed_tokens: list[str] = []
    transcribed_indices: list[int] = []
    for index, word in enumerate(words):
        cleaned = clean_word(word.text)
        if cleaned:
            transcribed_tokens.append(cleaned)
            transcribed_indices.append(index)

    lyric_tokens = [clean_word(word) for word in lyrics_words]
    lyric_tokens = [word for word in lyric_tokens if word]
    if not transcribed_tokens or not lyric_tokens:
        return []

    added_flags: list[FlaggedWord] = []
    matcher = SequenceMatcher(a=transcribed_tokens, b=lyric_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag not in {"equal", "replace"}:
            continue

        block_size = min(i2 - i1, j2 - j1)
        for offset in range(block_size):
            word_index = transcribed_indices[i1 + offset]
            if word_index in existing_indices:
                continue

            matched_term = match_word(lyric_tokens[j1 + offset], profanity_set)
            if matched_term:
                added_flags.append(FlaggedWord(word=words[word_index], matched_term=matched_term, index=word_index))
                existing_indices.add(word_index)

    return added_flags


def process(
    input_path: Path,
    output_path: Path | None = None,
    *,
    mode: str | None = None,
    language: str | None = None,
    device: str | None = None,
    custom_words: list[str] | None = None,
    exclude_words: list[str] | None = None,
    bleep_freq: int | None = None,
    model_name: str = "htdemucs_ft",
    max_file_size_mb: int | None = None,
    artist: str | None = None,
    title: str | None = None,
    genius_api_key: str | None = None,
    custom_mapping_path: Path | None = None,
    tts_voice: str | None = None,
    synth_engine: str = "edge-tts",
    kits_api_key: str | None = None,
    kits_voice_model_id: int | None = None,
    ai_provider: str | None = None,
    ai_api_key: str | None = None,
    delete_sentences: list[str] | None = None,
) -> PipelineResult:
    """Run the full Sanitune pipeline on an audio file."""
    settings = Settings.from_env()
    resolved_mode = settings.default_mode if mode is None else mode
    resolved_language = settings.language if language is None else language
    requested_device = settings.device if device is None else device
    resolved_bleep_freq = settings.bleep_freq if bleep_freq is None else bleep_freq
    resolved_max_file_size_mb = settings.max_file_size_mb if max_file_size_mb is None else max_file_size_mb

    start_time = time.monotonic()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported input file type '{input_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_INPUT_EXTENSIONS))}"
        )

    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    if file_size_mb > resolved_max_file_size_mb:
        raise ValueError(
            f"Input file is {file_size_mb:.1f} MB, exceeding the "
            f"{resolved_max_file_size_mb} MB limit. Set --max-file-size or SANITUNE_MAX_FILE_SIZE to override."
        )

    # Detect input format for preservation
    input_format = detect_audio_format(input_path)

    if output_path is None:
        # Default: match input format (e.g. MP3 in → MP3 out)
        # Prefer detected extension over file suffix (handles mislabeled files)
        detected_ext = str(input_format.get("extension") or input_path.suffix).lower()
        out_ext = detected_ext if detected_ext in SUPPORTED_OUTPUT_EXTENSIONS else ".wav"
        output_path = input_path.with_name(f"{input_path.stem}_clean{out_ext}")
    elif output_path.suffix.lower() not in SUPPORTED_OUTPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported output file type '{output_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_OUTPUT_EXTENSIONS))}"
        )

    resolved_device = detect_device(requested_device)
    logger.info(
        "Starting pipeline: %s → mode=%s, lang=%s, device=%s",
        input_path.name,
        resolved_mode,
        resolved_language,
        resolved_device,
    )

    logger.info("[1/5] Separating vocals...")
    sep_result = separate(input_path, device=resolved_device, model_name=model_name)

    logger.info("[2/5] Transcribing vocals...")
    trans_result = transcribe(
        sep_result.vocals,
        sep_result.sample_rate,
        device=resolved_device,
        language=resolved_language,
    )

    lyrics_words: list[str] = []
    if artist and title:
        try:
            from sanitune.lyrics import fetch_lyrics

            logger.info("[2.5/5] Fetching lyrics for '%s - %s'...", artist, title)
            lyrics_result = fetch_lyrics(artist, title, genius_api_key=genius_api_key)
            if lyrics_result:
                lyrics_words = lyrics_result.words
                logger.info(
                    "Found %d words in official lyrics (provider: %s)",
                    len(lyrics_words),
                    lyrics_result.provider,
                )
            else:
                logger.info("No lyrics found, continuing with transcription only")
        except ImportError:
            logger.info("Lyrics providers not installed (pip install sanitune[lyrics]), skipping")

    logger.info("[3/5] Detecting profanity...")
    flagged = detect(
        trans_result.words,
        language=resolved_language,
        custom_words=custom_words,
        exclude_words=exclude_words,
    )

    # Sentence-level deletion: match user-provided sentences against transcribed segments
    if delete_sentences:
        sentence_flags = detect_sentences(trans_result.segments, delete_sentences)
        if sentence_flags:
            flagged.extend(sentence_flags)
            logger.info("Added %d sentence-level deletions", len(sentence_flags))

    if lyrics_words:
        profanity_set = build_profanity_set(
            resolved_language,
            custom_words=custom_words,
            exclude_words=exclude_words,
        )
        lyrics_flags = _merge_lyrics_reference_flags(
            trans_result.words,
            lyrics_words,
            profanity_set,
            existing_indices={item.index for item in flagged},
        )
        if lyrics_flags:
            flagged.extend(lyrics_flags)
            flagged.sort(key=lambda item: item.index)
            logger.info("Lyrics alignment added %d extra flagged words", len(lyrics_flags))

    # Step 3.5: AI-powered contextual replacement suggestions
    ai_suggestions: dict[str, str] | None = None
    if ai_provider and ai_api_key and resolved_mode == "replace" and flagged:
        try:
            from sanitune.ai_suggest import suggest_replacements_batch

            logger.info("[3.5/5] Getting AI replacement suggestions...")
            ai_items = []
            for fw in flagged:
                idx = fw.index
                words = trans_result.words
                before = " ".join(w.text for w in words[max(0, idx - 10):idx])
                after = " ".join(w.text for w in words[idx + 1:idx + 11])
                ai_items.append({
                    "word": fw.matched_term,
                    "context_before": before,
                    "context_after": after,
                })

            ai_suggestions = suggest_replacements_batch(
                ai_items,
                language=resolved_language,
                provider=ai_provider,
                api_key=ai_api_key,
            )
            if ai_suggestions:
                logger.info("AI suggested %d replacements", len(ai_suggestions))
        except Exception as exc:
            logger.warning("AI suggestions failed: %s — continuing with built-in mappings", exc)

    logger.info("[4/5] Editing vocals (%d words flagged)...", len(flagged))
    edited_vocals = edit(
        sep_result.vocals,
        sep_result.sample_rate,
        flagged,
        mode=resolved_mode,
        bleep_freq=resolved_bleep_freq,
        language=resolved_language,
        custom_mapping_path=custom_mapping_path,
        tts_voice=tts_voice,
        device=resolved_device,
        synth_engine=synth_engine,
        kits_api_key=kits_api_key,
        kits_voice_model_id=kits_voice_model_id,
        ai_suggestions=ai_suggestions,
    )

    logger.info("[5/5] Remixing...")
    final_path = remix(
        edited_vocals,
        sep_result.instrumentals,
        sep_result.sample_rate,
        output_path,
        original=sep_result.original,
        flagged=flagged,
        input_format=input_format,
    )

    elapsed = time.monotonic() - start_time
    logger.info("Pipeline complete in %.1fs. Output: %s", elapsed, final_path)

    return PipelineResult(
        output_path=final_path,
        flagged_words=flagged,
        transcription=trans_result,
        elapsed_seconds=elapsed,
    )
