"""Lyrics fetching from multiple providers with fallback chain.

Supports synced (timed) and plain lyrics. Used to cross-reference
transcription results against official lyrics for improved detection accuracy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SyncedLine:
    """A single line of lyrics with timestamp."""

    text: str
    timestamp_ms: int


@dataclass
class LyricsResult:
    """Result from a lyrics fetch operation."""

    text: str
    provider: str
    synced_lines: list[SyncedLine] = field(default_factory=list)
    artist: str = ""
    title: str = ""

    @property
    def is_synced(self) -> bool:
        return len(self.synced_lines) > 0

    @property
    def words(self) -> list[str]:
        """Extract individual words from lyrics text."""
        return [w.lower() for w in re.findall(r"[\w']+", self.text, re.UNICODE)]


def _parse_lrc(lrc_text: str) -> list[SyncedLine]:
    """Parse LRC format into SyncedLine objects.

    LRC format: [mm:ss.xx] lyrics text
    """
    lines = []
    pattern = re.compile(r"\[(\d+):(\d+)\.(\d+)\]\s*(.*)")
    for line in lrc_text.splitlines():
        match = pattern.match(line.strip())
        if match:
            minutes, seconds, centiseconds, text = match.groups()
            timestamp_ms = (int(minutes) * 60 + int(seconds)) * 1000 + int(centiseconds) * 10
            if text.strip():
                lines.append(SyncedLine(text=text.strip(), timestamp_ms=timestamp_ms))
    return lines


def fetch_synced(artist: str, title: str) -> LyricsResult | None:
    """Fetch synced (timed) lyrics using syncedlyrics library.

    Uses LRCLIB and other free sources. No API key required.

    Args:
        artist: Artist name.
        title: Song title.

    Returns:
        LyricsResult with synced lines, or None if not found.
    """
    try:
        import syncedlyrics
    except ImportError:
        logger.debug("syncedlyrics not installed, skipping synced lyrics fetch")
        return None

    search_term = f"{artist} {title}"
    logger.info("Searching synced lyrics for '%s'...", search_term)

    try:
        lrc = syncedlyrics.search(search_term)
    except Exception:
        logger.warning("syncedlyrics search failed for '%s'", search_term, exc_info=True)
        return None

    if not lrc:
        logger.info("No synced lyrics found for '%s'", search_term)
        return None

    synced_lines = _parse_lrc(lrc)
    plain_text = "\n".join(line.text for line in synced_lines)

    logger.info("Found synced lyrics: %d lines", len(synced_lines))
    return LyricsResult(
        text=plain_text,
        provider="syncedlyrics",
        synced_lines=synced_lines,
        artist=artist,
        title=title,
    )


def fetch_genius(artist: str, title: str, api_key: str | None = None) -> LyricsResult | None:
    """Fetch plain lyrics from Genius.

    Args:
        artist: Artist name.
        title: Song title.
        api_key: Genius API token. If None, tries GENIUS_API_KEY env var.

    Returns:
        LyricsResult with plain text, or None if not found.
    """
    try:
        import lyricsgenius
    except ImportError:
        logger.debug("lyricsgenius not installed, skipping Genius fetch")
        return None

    import os

    token = api_key or os.environ.get("GENIUS_API_KEY")
    if not token:
        logger.debug("No Genius API key available, skipping")
        return None

    logger.info("Searching Genius for '%s - %s'...", artist, title)

    try:
        genius = lyricsgenius.Genius(token, verbose=False, timeout=10)
        song = genius.search_song(title, artist)
    except Exception:
        logger.warning("Genius search failed for '%s - %s'", artist, title, exc_info=True)
        return None

    if not song or not song.lyrics:
        logger.info("No lyrics found on Genius for '%s - %s'", artist, title)
        return None

    logger.info("Found lyrics on Genius (%d chars)", len(song.lyrics))
    return LyricsResult(
        text=song.lyrics,
        provider="genius",
        artist=artist,
        title=title,
    )


def fetch_lyrics(
    artist: str,
    title: str,
    *,
    genius_api_key: str | None = None,
    prefer_synced: bool = True,
) -> LyricsResult | None:
    """Fetch lyrics using a fallback chain of providers.

    Order: syncedlyrics (free, synced) -> Genius (API key, plain text).

    Args:
        artist: Artist name.
        title: Song title.
        genius_api_key: Optional Genius API token.
        prefer_synced: If True, try synced providers first.

    Returns:
        LyricsResult or None if no provider has the lyrics.
    """
    providers: list[tuple[str, callable]] = []

    if prefer_synced:
        providers.append(("syncedlyrics", lambda: fetch_synced(artist, title)))
        providers.append(("genius", lambda: fetch_genius(artist, title, genius_api_key)))
    else:
        providers.append(("genius", lambda: fetch_genius(artist, title, genius_api_key)))
        providers.append(("syncedlyrics", lambda: fetch_synced(artist, title)))

    for name, fetch_fn in providers:
        logger.debug("Trying provider: %s", name)
        result = fetch_fn()
        if result:
            return result

    logger.warning("No lyrics found from any provider for '%s - %s'", artist, title)
    return None


def extract_profane_lines(lyrics: LyricsResult, profanity_set: set[str]) -> list[str]:
    """Find lines in lyrics that contain known profane words.

    Useful for cross-referencing transcription against official lyrics
    to improve detection confidence.

    Args:
        lyrics: Fetched lyrics result.
        profanity_set: Set of profane words to check against.

    Returns:
        List of lines containing at least one profane word.
    """
    profane_lines = []
    for line in lyrics.text.splitlines():
        line_words = {w.lower() for w in re.findall(r"[\w']+", line, re.UNICODE)}
        if line_words & profanity_set:
            profane_lines.append(line)
    return profane_lines
