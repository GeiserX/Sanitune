"""Replacement word mappings — profanity-to-clean-word dictionaries."""

from __future__ import annotations

import json
import logging
from importlib.resources import files
from pathlib import Path

logger = logging.getLogger(__name__)


def load_mapping(language: str, custom_mapping_path: Path | None = None) -> dict[str, str]:
    """Load a profanity-to-clean-word mapping.

    Args:
        language: Language code (e.g. 'en', 'es').
        custom_mapping_path: Optional path to a JSON file that overrides/extends built-in mappings.

    Returns:
        Dictionary mapping profane words (lowercase) to clean replacements.
    """
    mapping: dict[str, str] = {}

    # Load built-in mapping
    try:
        builtin = files("sanitune.mappings").joinpath(f"{language}.json")
        mapping = json.loads(builtin.read_text(encoding="utf-8"))
        logger.debug("Loaded %d built-in mappings for '%s'", len(mapping), language)
    except FileNotFoundError:
        logger.warning("No built-in mapping for language '%s', trying 'en'", language)
        try:
            fallback = files("sanitune.mappings").joinpath("en.json")
            mapping = json.loads(fallback.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("No built-in mapping files found")

    # Overlay custom mapping
    if custom_mapping_path is not None:
        with open(custom_mapping_path, encoding="utf-8") as f:
            custom = json.load(f)
        mapping.update(custom)
        logger.debug("Loaded %d custom mappings from %s", len(custom), custom_mapping_path)

    return {k.lower(): v for k, v in mapping.items()}


def get_replacement(word: str, mapping: dict[str, str]) -> str | None:
    """Look up a clean replacement for a flagged word.

    Returns the replacement string, or None if no mapping exists.
    """
    return mapping.get(word.lower().strip())
