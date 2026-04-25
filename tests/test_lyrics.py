"""Tests for the lyrics module."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sanitune.lyrics import (
    LyricsResult,
    SyncedLine,
    _parse_lrc,
    extract_profane_lines,
    fetch_genius,
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


# --- Additional tests for uncovered paths ---


def test_fetch_synced_import_error(monkeypatch):
    """Should return None when syncedlyrics is not installed."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "syncedlyrics":
            raise ImportError("No module named 'syncedlyrics'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    # Need to also remove it from sys.modules if present
    import sys

    monkeypatch.delitem(sys.modules, "syncedlyrics", raising=False)

    result = fetch_synced("Artist", "Title")
    assert result is None


def test_fetch_synced_search_exception(monkeypatch):
    """Should return None when syncedlyrics.search raises."""
    def raise_error(_term):
        raise RuntimeError("network failure")

    fake_module = SimpleNamespace(search=raise_error)
    monkeypatch.setitem(__import__("sys").modules, "syncedlyrics", fake_module)

    result = fetch_synced("Artist", "Title")
    assert result is None


def test_fetch_synced_returns_none_when_no_lrc(monkeypatch):
    """Should return None when syncedlyrics returns None/empty."""
    fake_module = SimpleNamespace(search=lambda _term: None)
    monkeypatch.setitem(__import__("sys").modules, "syncedlyrics", fake_module)

    result = fetch_synced("Artist", "Title")
    assert result is None


def test_fetch_synced_success(monkeypatch):
    """Should return LyricsResult with synced lines on success."""
    lrc_text = "[00:05.00] Hello world\n[00:10.00] Goodbye world\n"
    fake_module = SimpleNamespace(search=lambda _term: lrc_text)
    monkeypatch.setitem(__import__("sys").modules, "syncedlyrics", fake_module)

    result = fetch_synced("Artist", "Title")
    assert result is not None
    assert result.provider == "syncedlyrics"
    assert result.is_synced
    assert len(result.synced_lines) == 2
    assert result.artist == "Artist"
    assert result.title == "Title"


def test_fetch_genius_import_error(monkeypatch):
    """Should return None when lyricsgenius is not installed."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "lyricsgenius":
            raise ImportError("No module named 'lyricsgenius'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    import sys

    monkeypatch.delitem(sys.modules, "lyricsgenius", raising=False)

    result = fetch_genius("Artist", "Title")
    assert result is None


def test_fetch_genius_no_api_key(monkeypatch):
    """Should return None when no API key is available."""
    fake_module = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "lyricsgenius", fake_module)
    monkeypatch.delenv("GENIUS_API_KEY", raising=False)

    result = fetch_genius("Artist", "Title", api_key=None)
    assert result is None


def test_fetch_genius_search_exception(monkeypatch):
    """Should return None when Genius search raises."""
    fake_genius_instance = MagicMock()
    fake_genius_instance.search_song.side_effect = RuntimeError("API error")
    fake_module = MagicMock()
    fake_module.Genius.return_value = fake_genius_instance
    monkeypatch.setitem(__import__("sys").modules, "lyricsgenius", fake_module)

    result = fetch_genius("Artist", "Title", api_key="test-key")
    assert result is None


def test_fetch_genius_no_song_found(monkeypatch):
    """Should return None when Genius finds no song."""
    fake_genius_instance = MagicMock()
    fake_genius_instance.search_song.return_value = None
    fake_module = MagicMock()
    fake_module.Genius.return_value = fake_genius_instance
    monkeypatch.setitem(__import__("sys").modules, "lyricsgenius", fake_module)

    result = fetch_genius("Artist", "Title", api_key="test-key")
    assert result is None


def test_fetch_genius_success(monkeypatch):
    """Should return LyricsResult on successful Genius lookup."""
    fake_song = MagicMock()
    fake_song.lyrics = "Some lyrics text here"
    fake_genius_instance = MagicMock()
    fake_genius_instance.search_song.return_value = fake_song
    fake_module = MagicMock()
    fake_module.Genius.return_value = fake_genius_instance
    monkeypatch.setitem(__import__("sys").modules, "lyricsgenius", fake_module)

    result = fetch_genius("Artist", "Title", api_key="test-key")
    assert result is not None
    assert result.provider == "genius"
    assert result.text == "Some lyrics text here"


def test_fetch_genius_env_api_key(monkeypatch):
    """Should use GENIUS_API_KEY env var if api_key is None."""
    monkeypatch.setenv("GENIUS_API_KEY", "env-key")
    fake_song = MagicMock()
    fake_song.lyrics = "lyrics"
    fake_genius_instance = MagicMock()
    fake_genius_instance.search_song.return_value = fake_song
    fake_module = MagicMock()
    fake_module.Genius.return_value = fake_genius_instance
    monkeypatch.setitem(__import__("sys").modules, "lyricsgenius", fake_module)

    result = fetch_genius("Artist", "Title", api_key=None)
    assert result is not None


def test_fetch_lyrics_prefer_synced_true(monkeypatch):
    """With prefer_synced=True, syncedlyrics should be tried first."""
    synced_result = LyricsResult(
        text="synced text",
        provider="syncedlyrics",
        synced_lines=[SyncedLine(text="synced text", timestamp_ms=0)],
    )
    monkeypatch.setattr("sanitune.lyrics.fetch_synced", lambda *a, **kw: synced_result)
    genius_called = []
    monkeypatch.setattr("sanitune.lyrics.fetch_genius", lambda *a, **kw: genius_called.append(1) or None)

    result = fetch_lyrics("Artist", "Title", prefer_synced=True)
    assert result.provider == "syncedlyrics"
    assert len(genius_called) == 0  # genius should not be called since synced succeeded


def test_fetch_lyrics_prefer_synced_false(monkeypatch):
    """With prefer_synced=False, genius should be tried first."""
    genius_result = LyricsResult(text="genius text", provider="genius")
    monkeypatch.setattr("sanitune.lyrics.fetch_genius", lambda *a, **kw: genius_result)
    synced_called = []
    monkeypatch.setattr("sanitune.lyrics.fetch_synced", lambda *a, **kw: synced_called.append(1) or None)

    result = fetch_lyrics("Artist", "Title", prefer_synced=False)
    assert result.provider == "genius"
    assert len(synced_called) == 0


def test_fetch_lyrics_returns_none_all_fail(monkeypatch):
    """Should return None when all providers fail."""
    monkeypatch.setattr("sanitune.lyrics.fetch_synced", lambda *a, **kw: None)
    monkeypatch.setattr("sanitune.lyrics.fetch_genius", lambda *a, **kw: None)

    result = fetch_lyrics("Artist", "Title")
    assert result is None
