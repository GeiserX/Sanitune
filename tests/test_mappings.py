"""Tests for the replacement word mappings module."""

from __future__ import annotations

import json

from sanitune.mappings import get_replacement, load_mapping


def test_load_builtin_english():
    mapping = load_mapping("en")
    assert len(mapping) > 0
    assert "fuck" in mapping
    assert mapping["fuck"] == "fudge"


def test_load_builtin_spanish():
    mapping = load_mapping("es")
    assert len(mapping) > 0
    assert "mierda" in mapping
    assert mapping["mierda"] == "rayos"


def test_load_unknown_language_falls_back_to_english():
    mapping = load_mapping("xx")
    assert "fuck" in mapping


def test_load_custom_mapping_overrides(tmp_path):
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps({"fuck": "CUSTOM", "newword": "replacement"}))

    mapping = load_mapping("en", custom_mapping_path=custom)
    assert mapping["fuck"] == "CUSTOM"  # overridden
    assert mapping["newword"] == "replacement"  # added
    assert "damn" in mapping  # built-in still present


def test_load_custom_mapping_only(tmp_path):
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps({"badword": "goodword"}))

    mapping = load_mapping("xx", custom_mapping_path=custom)
    assert mapping["badword"] == "goodword"


def test_get_replacement_found():
    mapping = {"fuck": "fudge", "damn": "darn"}
    assert get_replacement("fuck", mapping) == "fudge"
    assert get_replacement("DAMN", mapping) == "darn"
    assert get_replacement("  fuck  ", mapping) == "fudge"


def test_get_replacement_not_found():
    mapping = {"fuck": "fudge"}
    assert get_replacement("hello", mapping) is None
    assert get_replacement("", mapping) is None


def test_mapping_keys_are_lowercase():
    mapping = load_mapping("en")
    for key in mapping:
        assert key == key.lower(), f"Key '{key}' is not lowercase"


def test_mapping_values_are_non_empty():
    for lang in ("en", "es"):
        mapping = load_mapping(lang)
        for key, value in mapping.items():
            assert value.strip(), f"Empty value for key '{key}' in {lang}"
