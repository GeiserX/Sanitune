"""Profanity detection engine with configurable word lists."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from sanitune.transcriber import Segment, Word

logger = logging.getLogger(__name__)
TOKEN_STRIP_PATTERN = re.compile(r"[^\w']", re.UNICODE)


@dataclass
class FlaggedWord:
    word: Word
    matched_term: str
    index: int


def _normalize(text: str) -> str:
    """Normalize text by removing diacritics for accent-insensitive matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def clean_word(text: str) -> str:
    """Normalize a single token for profanity matching."""
    return TOKEN_STRIP_PATTERN.sub("", text.lower()).strip("'")


def load_wordlist(language: str) -> set[str]:
    """Load the built-in profanity word list for a language."""
    if not re.match(r"^[a-z]{2,5}$", language):
        raise ValueError(f"Invalid language code: '{language}'. Expected 2-5 lowercase letters.")

    try:
        ref = resources.files("sanitune.wordlists").joinpath(f"{language}.txt")
        text = ref.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError) as err:
        wordlist_path = Path(__file__).parent / "wordlists" / f"{language}.txt"
        if not wordlist_path.exists():
            raise FileNotFoundError(f"No wordlist found for language '{language}'") from err
        text = wordlist_path.read_text(encoding="utf-8")

    words = set()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line.lower())
            # Also add accent-stripped form for matching
            normalized = _normalize(line)
            if normalized != line.lower():
                words.add(normalized)
    return words


def build_profanity_set(
    language: str,
    custom_words: list[str] | None = None,
    exclude_words: list[str] | None = None,
) -> set[str]:
    """Build the effective profanity set after custom additions/exclusions."""
    profanity_set = load_wordlist(language)

    if custom_words:
        for word in custom_words:
            lowered = word.lower()
            profanity_set.add(lowered)
            profanity_set.add(_normalize(lowered))

    if exclude_words:
        to_remove: set[str] = set()
        for word in exclude_words:
            lowered = word.lower()
            to_remove.add(lowered)
            to_remove.add(_normalize(lowered))
        profanity_set -= to_remove

    return profanity_set


def match_word(text: str, profanity_set: set[str]) -> str | None:
    """Return the matched profane term for a token, if any."""
    cleaned = clean_word(text)
    if not cleaned:
        return None

    cleaned_normalized = _normalize(cleaned)
    if cleaned in profanity_set or cleaned_normalized in profanity_set:
        return cleaned

    for term in profanity_set:
        if len(term) >= 4 and term in cleaned_normalized and cleaned_normalized != term:
            return term

    return None


def detect(
    words: list[Word],
    *,
    language: str = "en",
    custom_words: list[str] | None = None,
    exclude_words: list[str] | None = None,
) -> list[FlaggedWord]:
    """Detect profane words in a transcription.

    Uses exact match against the word list after normalizing punctuation.

    Args:
        words: List of transcribed words with timestamps.
        language: Language code for built-in word list.
        custom_words: Additional words to flag.
        exclude_words: Words to skip even if in the word list.

    Returns:
        List of flagged words with their indices.
    """
    profanity_set = build_profanity_set(language, custom_words=custom_words, exclude_words=exclude_words)

    flagged = []

    for i, word in enumerate(words):
        matched_term = match_word(word.text, profanity_set)
        if matched_term:
            flagged.append(FlaggedWord(word=word, matched_term=matched_term, index=i))

    logger.info("Detected %d flagged words out of %d total", len(flagged), len(words))
    return flagged


def _normalize_sentence(text: str) -> str:
    """Normalize a sentence for fuzzy matching: lowercase, strip punctuation, collapse whitespace."""
    text = _normalize(text)
    text = TOKEN_STRIP_PATTERN.sub(" ", text)
    return " ".join(text.split())


def detect_sentences(
    segments: list[Segment],
    target_sentences: list[str],
) -> list[FlaggedWord]:
    """Match user-provided sentences against transcribed segments and flag entire segments.

    Uses normalized substring matching: if the target text appears within a segment
    (or vice versa), the entire segment is flagged for deletion.

    Returns FlaggedWord objects where the Word spans the full segment time range.
    """
    if not segments or not target_sentences:
        return []

    targets = [_normalize_sentence(s) for s in target_sentences if s.strip()]
    if not targets:
        return []

    flagged: list[FlaggedWord] = []
    for seg in segments:
        seg_norm = _normalize_sentence(seg.text)
        if not seg_norm:
            continue

        for target in targets:
            # Match if target is contained in segment or segment is contained in target
            if target in seg_norm or seg_norm in target:
                sentence_word = Word(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    score=1.0,
                )
                flagged.append(FlaggedWord(
                    word=sentence_word,
                    matched_term=f"[sentence] {target}",
                    index=-1,
                ))
                logger.debug("Sentence match: '%s' in segment '%s' (%.2f-%.2f)", target, seg.text, seg.start, seg.end)
                break  # Don't double-flag the same segment

    logger.info("Detected %d sentence-level flags out of %d segments", len(flagged), len(segments))
    return flagged
