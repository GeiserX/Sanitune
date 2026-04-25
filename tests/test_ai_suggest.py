"""Tests for the AI suggestion module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from sanitune.ai_suggest import (
    _parse_response,
    suggest_replacement,
    suggest_replacements_batch,
)


class TestParseResponse:
    def test_valid_json(self):
        word, confidence = _parse_response('{"replacement": "fudge", "confidence": 0.9}')
        assert word == "fudge"
        assert confidence == 0.9

    def test_invalid_json(self):
        word, confidence = _parse_response("not json at all")
        assert word is None
        assert confidence == 0.0

    def test_empty_replacement(self):
        word, confidence = _parse_response('{"replacement": "", "confidence": 0.9}')
        assert word is None
        assert confidence == 0.0

    def test_confidence_out_of_range(self):
        word, confidence = _parse_response('{"replacement": "fudge", "confidence": 1.5}')
        assert word is None
        assert confidence == 0.0

    def test_missing_keys(self):
        word, confidence = _parse_response('{"other": "value"}')
        assert word is None
        assert confidence == 0.0

    def test_whitespace_around_json(self):
        word, confidence = _parse_response('  {"replacement": "fudge", "confidence": 0.8}  ')
        assert word == "fudge"
        assert confidence == 0.8


class TestSuggestReplacement:
    @patch("sanitune.ai_suggest._call_anthropic")
    def test_anthropic_provider(self, mock_call):
        mock_call.return_value = ("fudge", 0.9)
        word, confidence = suggest_replacement(
            "fuck", "this is", "song", provider="anthropic", api_key="test-key"
        )
        assert word == "fudge"
        assert confidence == 0.9
        mock_call.assert_called_once()

    @patch("sanitune.ai_suggest._call_openai")
    def test_openai_provider(self, mock_call):
        mock_call.return_value = ("darn", 0.85)
        word, confidence = suggest_replacement(
            "damn", "oh", "it", provider="openai", api_key="test-key"
        )
        assert word == "darn"
        mock_call.assert_called_once()

    def test_unknown_provider(self):
        word, confidence = suggest_replacement(
            "word", "before", "after", provider="unknown", api_key="key"
        )
        assert word is None
        assert confidence == 0.0

    @patch("sanitune.ai_suggest._call_anthropic")
    def test_exception_returns_none(self, mock_call):
        mock_call.side_effect = Exception("API error")
        word, confidence = suggest_replacement(
            "fuck", "before", "after", provider="anthropic", api_key="key"
        )
        assert word is None
        assert confidence == 0.0


class TestCallAnthropic:
    @patch("requests.post")
    def test_successful_call(self, mock_post):
        from sanitune.ai_suggest import _call_anthropic

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"text": '{"replacement": "fudge", "confidence": 0.9}'}]
        }
        mock_post.return_value = mock_resp

        word, confidence = _call_anthropic("prompt", "api-key", None)
        assert word == "fudge"
        assert confidence == 0.9
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_api_error_status(self, mock_post):
        from sanitune.ai_suggest import _call_anthropic

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal server error"
        mock_post.return_value = mock_resp

        word, confidence = _call_anthropic("prompt", "api-key", None)
        assert word is None
        assert confidence == 0.0

    @patch("requests.post")
    def test_unexpected_response_format(self, mock_post):
        from sanitune.ai_suggest import _call_anthropic

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"unexpected": "format"}
        mock_resp.text = '{"unexpected": "format"}'
        mock_post.return_value = mock_resp

        word, confidence = _call_anthropic("prompt", "api-key", None)
        assert word is None
        assert confidence == 0.0

    @patch("requests.post")
    def test_custom_model(self, mock_post):
        from sanitune.ai_suggest import _call_anthropic

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"text": '{"replacement": "x", "confidence": 0.7}'}]
        }
        mock_post.return_value = mock_resp

        _call_anthropic("prompt", "key", "custom-model")
        call_json = mock_post.call_args[1]["json"]
        assert call_json["model"] == "custom-model"


class TestCallOpenAI:
    @patch("requests.post")
    def test_successful_call(self, mock_post):
        from sanitune.ai_suggest import _call_openai

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"replacement": "darn", "confidence": 0.8}'}}]
        }
        mock_post.return_value = mock_resp

        word, confidence = _call_openai("prompt", "api-key", None)
        assert word == "darn"
        assert confidence == 0.8

    @patch("requests.post")
    def test_api_error_status(self, mock_post):
        from sanitune.ai_suggest import _call_openai

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Rate limited"
        mock_post.return_value = mock_resp

        word, confidence = _call_openai("prompt", "api-key", None)
        assert word is None

    @patch("requests.post")
    def test_unexpected_response(self, mock_post):
        from sanitune.ai_suggest import _call_openai

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.text = "{}"
        mock_post.return_value = mock_resp

        word, confidence = _call_openai("prompt", "api-key", None)
        assert word is None

    @patch("requests.post")
    def test_custom_model(self, mock_post):
        from sanitune.ai_suggest import _call_openai

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"replacement": "x", "confidence": 0.7}'}}]
        }
        mock_post.return_value = mock_resp

        _call_openai("prompt", "key", "gpt-4")
        call_json = mock_post.call_args[1]["json"]
        assert call_json["model"] == "gpt-4"


class TestSuggestReplacementsBatch:
    @patch("sanitune.ai_suggest.suggest_replacement")
    def test_collects_suggestions(self, mock_suggest):
        mock_suggest.side_effect = [("fudge", 0.9), ("darn", 0.8)]

        items = [
            {"word": "fuck", "context_before": "this", "context_after": "song"},
            {"word": "damn", "context_before": "oh", "context_after": "it"},
        ]
        result = suggest_replacements_batch(items, api_key="key")
        assert result == {"fuck": "fudge", "damn": "darn"}

    @patch("sanitune.ai_suggest.suggest_replacement")
    def test_skips_low_confidence(self, mock_suggest):
        mock_suggest.return_value = ("maybe", 0.3)

        items = [{"word": "heck", "context_before": "", "context_after": ""}]
        result = suggest_replacements_batch(items, api_key="key")
        assert result == {}

    @patch("sanitune.ai_suggest.suggest_replacement")
    def test_respects_max_calls(self, mock_suggest):
        mock_suggest.return_value = ("replacement", 0.9)

        items = [{"word": f"word{i}"} for i in range(10)]
        result = suggest_replacements_batch(items, api_key="key", max_calls=3)
        assert len(result) == 3
        assert mock_suggest.call_count == 3

    @patch("sanitune.ai_suggest.suggest_replacement")
    def test_deduplicates_words(self, mock_suggest):
        mock_suggest.return_value = ("fudge", 0.9)

        items = [
            {"word": "fuck", "context_before": "a", "context_after": "b"},
            {"word": "fuck", "context_before": "c", "context_after": "d"},
        ]
        result = suggest_replacements_batch(items, api_key="key")
        assert mock_suggest.call_count == 1  # Only called once for "fuck"

    @patch("sanitune.ai_suggest.suggest_replacement")
    def test_handles_none_replacement(self, mock_suggest):
        mock_suggest.return_value = (None, 0.0)

        items = [{"word": "word1"}]
        result = suggest_replacements_batch(items, api_key="key")
        assert result == {}
