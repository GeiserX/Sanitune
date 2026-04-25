"""Tests for the profanity detection module."""

import pytest

from sanitune.detector import (
    _normalize,
    _normalize_sentence,
    build_profanity_set,
    clean_word,
    detect,
    detect_sentences,
    load_wordlist,
    match_word,
)
from sanitune.transcriber import Segment, Word


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
        Word(text="cabron", start=0.0, end=0.5),
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
    flagged = detect(words, custom_words=["cojon"])
    assert len(flagged) == 1


def test_detect_exclude_words_accent_normalized():
    """Excluding an accented word should also remove its normalized form."""
    words = [
        Word(text="cabron", start=0.0, end=0.5),
        Word(text="cabron", start=0.5, end=1.0),
    ]
    flagged = detect(words, language="es", exclude_words=["cabron"])
    assert len(flagged) == 0


# --- Tests for functions not previously covered ---


def test_normalize():
    """_normalize should strip diacritics and lowercase."""
    assert _normalize("Cabron") == "cabron"
    assert _normalize("HELLO") == "hello"
    assert _normalize("cafe") == "cafe"


def test_clean_word_strips_punctuation():
    assert clean_word("hello!") == "hello"
    assert clean_word("'it's'") == "it's"
    assert clean_word("...test...") == "test"


def test_clean_word_empty():
    assert clean_word("!!!") == ""
    assert clean_word("") == ""


def test_match_word_exact_match():
    profanity = {"fuck", "shit"}
    assert match_word("fuck", profanity) == "fuck"
    assert match_word("FUCK", profanity) == "fuck"


def test_match_word_no_match():
    profanity = {"fuck", "shit"}
    assert match_word("hello", profanity) is None


def test_match_word_empty_token():
    profanity = {"fuck"}
    assert match_word("!!!", profanity) is None


def test_match_word_substring_long_term():
    """Terms >= 4 chars should match as substrings."""
    profanity = {"fuck"}  # 4 chars
    assert match_word("motherfucker", profanity) == "fuck"


def test_match_word_substring_short_term():
    """Terms < 4 chars should NOT match as substrings."""
    profanity = {"ass"}  # 3 chars
    assert match_word("class", profanity) is None


def test_build_profanity_set_basic():
    ps = build_profanity_set("en")
    assert isinstance(ps, set)
    assert "fuck" in ps


def test_build_profanity_set_custom_words():
    ps = build_profanity_set("en", custom_words=["myword"])
    assert "myword" in ps


def test_build_profanity_set_exclude_words():
    ps = build_profanity_set("en", exclude_words=["fuck"])
    assert "fuck" not in ps


def test_normalize_sentence():
    assert _normalize_sentence("Hello, World!") == "hello world"
    assert _normalize_sentence("  multiple   spaces  ") == "multiple spaces"
    assert _normalize_sentence("Cafe") == "cafe"


def test_normalize_sentence_empty():
    assert _normalize_sentence("") == ""
    assert _normalize_sentence("   ") == ""


def test_detect_sentences_basic():
    segments = [
        Segment(text="I will kill you", start=0.0, end=1.0, words=[]),
        Segment(text="This is fine", start=1.0, end=2.0, words=[]),
    ]
    targets = ["kill you"]
    flagged = detect_sentences(segments, targets)
    assert len(flagged) == 1
    assert flagged[0].index == -1
    assert flagged[0].word.start == 0.0
    assert flagged[0].word.end == 1.0


def test_detect_sentences_no_match():
    segments = [
        Segment(text="This is clean", start=0.0, end=1.0, words=[]),
    ]
    targets = ["kill you"]
    flagged = detect_sentences(segments, targets)
    assert len(flagged) == 0


def test_detect_sentences_empty_inputs():
    assert detect_sentences([], ["test"]) == []
    assert detect_sentences([Segment(text="hello", start=0.0, end=1.0, words=[])], []) == []


def test_detect_sentences_empty_target_after_normalize():
    """Targets that normalize to empty should be skipped."""
    segments = [Segment(text="hello", start=0.0, end=1.0, words=[])]
    assert detect_sentences(segments, ["!!!"]) == []


def test_detect_sentences_empty_segment():
    """Empty segments should be skipped."""
    segments = [Segment(text="", start=0.0, end=1.0, words=[])]
    assert detect_sentences(segments, ["test"]) == []


def test_detect_sentences_no_double_flag():
    """Same segment should not be flagged twice even if it matches multiple targets."""
    segments = [
        Segment(text="kill you now", start=0.0, end=1.0, words=[]),
    ]
    targets = ["kill you", "kill you now"]
    flagged = detect_sentences(segments, targets)
    assert len(flagged) == 1


def test_detect_sentences_reverse_containment():
    """When the target is longer than the segment, should still match (padded containment)."""
    segments = [
        Segment(text="kill you", start=0.0, end=1.0, words=[]),
    ]
    targets = ["i will kill you now"]
    flagged = detect_sentences(segments, targets)
    assert len(flagged) == 1
