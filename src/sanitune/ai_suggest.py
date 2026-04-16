"""AI-powered contextual replacement suggestions using LLM APIs."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a lyric editor for clean versions of songs. Given a flagged explicit word \
and its surrounding lyrics context, suggest a single clean replacement word that:

1. Preserves the emotional tone and intensity of the original
2. Matches the syllable count as closely as possible
3. Fits the rhyme scheme if one is apparent
4. Is appropriate for all audiences
5. Sounds natural in the context

Respond with ONLY a JSON object: {"replacement": "word", "confidence": 0.0-1.0}
Do not include any other text.\
"""


def suggest_replacement(
    flagged_word: str,
    context_before: str,
    context_after: str,
    language: str = "en",
    *,
    provider: str = "anthropic",
    api_key: str,
    model: str | None = None,
) -> tuple[str | None, float]:
    """Ask an LLM for a contextually appropriate clean replacement.

    Args:
        flagged_word: The explicit word to replace.
        context_before: ~10 words of lyrics before the flagged word.
        context_after: ~10 words of lyrics after the flagged word.
        language: Language code of the lyrics.
        provider: 'anthropic' or 'openai'.
        api_key: API key for the chosen provider.
        model: Override model name. Defaults to a small/fast model per provider.

    Returns:
        Tuple of (replacement_word, confidence). Returns (None, 0.0) on failure.
    """
    user_prompt = (
        f"Language: {language}\n"
        f"Context before: \"{context_before}\"\n"
        f"Flagged word: \"{flagged_word}\"\n"
        f"Context after: \"{context_after}\"\n\n"
        f"Suggest a clean replacement for \"{flagged_word}\" in this lyric context."
    )

    try:
        if provider == "anthropic":
            return _call_anthropic(user_prompt, api_key, model)
        elif provider == "openai":
            return _call_openai(user_prompt, api_key, model)
        else:
            logger.warning("Unknown AI provider '%s', skipping suggestion", provider)
            return None, 0.0
    except Exception as exc:
        logger.warning("AI suggestion failed for '%s': %s", flagged_word, exc)
        return None, 0.0


def _call_anthropic(user_prompt: str, api_key: str, model: str | None) -> tuple[str | None, float]:
    """Call Anthropic Messages API."""
    import requests

    resolved_model = model or "claude-haiku-4-5-20251001"

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": resolved_model,
            "max_tokens": 100,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=15,
    )

    if resp.status_code != 200:
        logger.warning("Anthropic API error %d: %s", resp.status_code, resp.text[:200])
        return None, 0.0

    content = resp.json()["content"][0]["text"]
    return _parse_response(content)


def _call_openai(user_prompt: str, api_key: str, model: str | None) -> tuple[str | None, float]:
    """Call OpenAI Chat Completions API."""
    import requests

    resolved_model = model or "gpt-4o-mini"

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": resolved_model,
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=15,
    )

    if resp.status_code != 200:
        logger.warning("OpenAI API error %d: %s", resp.status_code, resp.text[:200])
        return None, 0.0

    content = resp.json()["choices"][0]["message"]["content"]
    return _parse_response(content)


def _parse_response(content: str) -> tuple[str | None, float]:
    """Parse LLM JSON response into (replacement, confidence)."""
    try:
        data = json.loads(content.strip())
        word = data.get("replacement", "").strip()
        confidence = float(data.get("confidence", 0.0))
        if word and 0.0 <= confidence <= 1.0:
            return word, confidence
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.debug("Failed to parse AI response: %s", content[:100])
    return None, 0.0


def suggest_replacements_batch(
    flagged_words: list[dict],
    language: str = "en",
    *,
    provider: str = "anthropic",
    api_key: str,
    max_calls: int = 20,
) -> dict[str, str]:
    """Get AI suggestions for multiple flagged words.

    Args:
        flagged_words: List of dicts with 'word', 'context_before', 'context_after'.
        language: Language code.
        provider: LLM provider.
        api_key: API key.
        max_calls: Maximum number of API calls to prevent cost overruns.

    Returns:
        Dict mapping flagged word → suggested replacement.
    """
    suggestions: dict[str, str] = {}
    calls_made = 0

    for item in flagged_words:
        if calls_made >= max_calls:
            logger.info("AI suggestion limit reached (%d calls), stopping", max_calls)
            break

        word = item["word"]
        # Skip duplicates
        if word in suggestions:
            continue

        replacement, confidence = suggest_replacement(
            word,
            item.get("context_before", ""),
            item.get("context_after", ""),
            language=language,
            provider=provider,
            api_key=api_key,
        )
        calls_made += 1

        if replacement and confidence >= 0.5:
            suggestions[word] = replacement
            logger.info("AI suggestion: '%s' → '%s' (confidence=%.2f)", word, replacement, confidence)
        else:
            logger.debug("AI suggestion rejected for '%s': %s (%.2f)", word, replacement, confidence)

    return suggestions
