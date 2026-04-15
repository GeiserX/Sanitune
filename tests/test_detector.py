"""Tests for the profanity detection module."""

import pytest

from sanitune.detector import FlaggedWord, detect, load_wordlist
from sanitune.transcriber import Word


def test_load_wordlist_en():
    words = load_wordlist("en")
    assert isinstance(words, set)
    assert len(words) > 0
    assert "fuck" in words
    assert "damn" in words


def test_load_wordlist_es():
    words = load_wordlist("es")
    assert isinstance(words, set)
    assert len(words) > 0
    assert "joder" in words
    assert "mierda" in words


def test_detect_exact_match(sample_words):
    flagged = detect(sample_words)
    matched_terms = {fw.matched_term for fw in flagged}
    assert "damn" in matched_terms
    assert "fuck" in matched_terms or "fucking" in matched_terms


def test_detect_no_profanity():
    clean_words = [
        Word(text="This", start=0.0, end=0.3),
        Word(text="is", start=0.3, end=0.5),
        Word(text="clean", start=0.5, end=0.8),
    ]
    flagged = detect(clean_words)
    assert len(flagged) == 0


def test_detect_custom_words():
    words = [Word(text="dingus", start=0.0, end=0.5)]
    flagged = detect(words, custom_words=["dingus"])
    assert len(flagged) == 1
    assert flagged[0].matched_term == "dingus"


def test_detect_exclude_words(sample_words):
    flagged = detect(sample_words, exclude_words=["damn"])
    matched_terms = {fw.matched_term for fw in flagged}
    assert "damn" not in matched_terms


def test_detect_case_insensitive():
    words = [Word(text="FUCK", start=0.0, end=0.5)]
    flagged = detect(words)
    assert len(flagged) == 1


def test_detect_with_punctuation():
    words = [Word(text="fuck!", start=0.0, end=0.5)]
    flagged = detect(words)
    assert len(flagged) == 1
    assert flagged[0].matched_term == "fuck"


def test_detect_substring_compound_word():
    words = [Word(text="motherfucking", start=0.0, end=0.5)]
    flagged = detect(words)
    assert len(flagged) >= 1


def test_flagged_word_has_index(sample_words):
    flagged = detect(sample_words)
    for fw in flagged:
        assert isinstance(fw.index, int)
        assert fw.index >= 0


def test_load_wordlist_rejects_path_traversal():
    with pytest.raises(ValueError, match="Invalid language code"):
        load_wordlist("../../etc/passwd")

    with pytest.raises(ValueError, match="Invalid language code"):
        load_wordlist("EN")

    with pytest.raises(ValueError, match="Invalid language code"):
        load_wordlist("")


def test_detect_accent_insensitive():
    """Spanish words with/without accents should both match."""
    words = [
        Word(text="cabrón", start=0.0, end=0.5),
        Word(text="cabron", start=0.5, end=1.0),
    ]
    flagged = detect(words, language="es")
    assert len(flagged) == 2


def test_detect_pito_in_spanish():
    words = [Word(text="pito", start=0.0, end=0.5)]
    flagged = detect(words, language="es")
    assert len(flagged) == 1
    assert flagged[0].matched_term == "pito"


def test_detect_unsupported_language_raises():
    with pytest.raises(FileNotFoundError, match="No wordlist found"):
        load_wordlist("xx")


def test_detect_empty_words_list():
    flagged = detect([])
    assert len(flagged) == 0


def test_detect_word_all_punctuation():
    words = [Word(text="!!!", start=0.0, end=0.5)]
    flagged = detect(words)
    assert len(flagged) == 0


def test_detect_short_term_not_matched_as_substring():
    """Short profane terms (<4 chars) should not match as substrings."""
    words = [Word(text="class", start=0.0, end=0.5)]
    flagged = detect(words, custom_words=["ass"])
    # "ass" is 3 chars, so substring match threshold (>=4) should prevent matching "class"
    assert len(flagged) == 0


def test_detect_custom_words_accent_normalized():
    """Custom words should be normalized the same way as built-ins."""
    words = [Word(text="cojon", start=0.0, end=0.5)]
    flagged = detect(words, custom_words=["cojón"])
    assert len(flagged) == 1


def test_detect_exclude_words_accent_normalized():
    """Excluding an accented word should also remove its normalized form."""
    words = [
        Word(text="cabron", start=0.0, end=0.5),
        Word(text="cabrón", start=0.5, end=1.0),
    ]
    flagged = detect(words, language="es", exclude_words=["cabrón"])
    assert len(flagged) == 0
