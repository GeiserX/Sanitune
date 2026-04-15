"""Profanity detection engine with configurable word lists."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from sanitune.transcriber import Word

logger = logging.getLogger(__name__)


@dataclass
class FlaggedWord:
    word: Word
    matched_term: str
    index: int


def _normalize(text: str) -> str:
    """Normalize text by removing diacritics for accent-insensitive matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


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
    profanity_set = load_wordlist(language)

    if custom_words:
        profanity_set.update(w.lower() for w in custom_words)

    if exclude_words:
        profanity_set -= {w.lower() for w in exclude_words}

    flagged = []
    strip_pattern = re.compile(r"[^\w']", re.UNICODE)

    for i, word in enumerate(words):
        cleaned = strip_pattern.sub("", word.text.lower()).strip("'")
        if not cleaned:
            continue

        # Try both raw and accent-stripped forms
        cleaned_normalized = _normalize(cleaned)

        if cleaned in profanity_set or cleaned_normalized in profanity_set:
            flagged.append(FlaggedWord(word=word, matched_term=cleaned, index=i))
            continue

        # Check if the word contains a profane substring (for compound words / suffixed forms)
        # Note: this can produce false positives (e.g. "cockpit" matching "cock").
        # The >=4 char threshold mitigates most cases. Consider word-boundary regex for v0.2.0.
        for term in profanity_set:
            if len(term) >= 4 and term in cleaned_normalized and cleaned_normalized != term:
                flagged.append(FlaggedWord(word=word, matched_term=term, index=i))
                break

    logger.info("Detected %d flagged words out of %d total", len(flagged), len(words))
    return flagged
