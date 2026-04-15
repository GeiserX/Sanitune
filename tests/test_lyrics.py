"""Tests for the lyrics module."""

from types import SimpleNamespace

from sanitune.lyrics import (
    LyricsResult,
    SyncedLine,
    _parse_lrc,
    extract_profane_lines,
    fetch_lyrics,
    fetch_synced,
)


def test_parse_lrc_basic():
    lrc = "[00:12.50] Hello world\n[00:15.30] This is a test\n"
    lines = _parse_lrc(lrc)
    assert len(lines) == 2
    assert lines[0].text == "Hello world"
    assert lines[0].timestamp_ms == 12500
    assert lines[1].text == "This is a test"
    assert lines[1].timestamp_ms == 15300


def test_parse_lrc_skips_empty_lines():
    lrc = "[00:10.00]\n[00:12.00] Actual text\n[00:14.00]   \n"
    lines = _parse_lrc(lrc)
    assert len(lines) == 1
    assert lines[0].text == "Actual text"


def test_parse_lrc_handles_non_lrc_lines():
    lrc = "Not an LRC line\n[00:05.00] Real line\nAnother bad line\n"
    lines = _parse_lrc(lrc)
    assert len(lines) == 1
    assert lines[0].text == "Real line"


def test_parse_lrc_minute_rollover():
    lrc = "[02:30.00] Two minutes thirty seconds\n"
    lines = _parse_lrc(lrc)
    assert len(lines) == 1
    assert lines[0].timestamp_ms == 150000  # 2*60*1000 + 30*1000


def test_lyrics_result_is_synced():
    result = LyricsResult(text="hello", provider="test")
    assert not result.is_synced

    result_synced = LyricsResult(
        text="hello",
        provider="test",
        synced_lines=[SyncedLine(text="hello", timestamp_ms=0)],
    )
    assert result_synced.is_synced


def test_lyrics_result_words():
    result = LyricsResult(text="Hello world, this is a test!", provider="test")
    assert result.words == ["hello", "world", "this", "is", "a", "test"]


def test_lyrics_result_words_unicode():
    result = LyricsResult(text="Cancion con pito y mierda", provider="test")
    words = result.words
    assert "pito" in words
    assert "mierda" in words


def test_extract_profane_lines():
    lyrics = LyricsResult(
        text="This is clean\nThis has fuck in it\nAlso clean\nDamn this too\n",
        provider="test",
    )
    profanity_set = {"fuck", "damn"}
    profane = extract_profane_lines(lyrics, profanity_set)
    assert len(profane) == 2
    assert "This has fuck in it" in profane
    assert "Damn this too" in profane


def test_extract_profane_lines_empty():
    lyrics = LyricsResult(text="All clean here\nNothing bad\n", provider="test")
    profane = extract_profane_lines(lyrics, {"fuck", "shit"})
    assert len(profane) == 0


def test_extract_profane_lines_case_insensitive():
    lyrics = LyricsResult(text="FUCK this\n", provider="test")
    profane = extract_profane_lines(lyrics, {"fuck"})
    assert len(profane) == 1


def test_fetch_synced_returns_none_for_unsynced_text(monkeypatch):
    fake_module = SimpleNamespace(search=lambda _term: "plain text without any timestamps")
    monkeypatch.setitem(__import__("sys").modules, "syncedlyrics", fake_module)

    assert fetch_synced("Artist", "Title") is None


def test_fetch_lyrics_falls_back_when_synced_provider_has_no_timestamps(monkeypatch):
    monkeypatch.setattr("sanitune.lyrics.fetch_synced", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "sanitune.lyrics.fetch_genius",
        lambda *_args, **_kwargs: LyricsResult(
            text="Fallback lyrics",
            provider="genius",
        ),
    )

    result = fetch_lyrics("Artist", "Title")

    assert result is not None
    assert result.provider == "genius"
