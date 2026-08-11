"""Provider-independent opener logic: the no-dash sanitizer (Corey-Wayne rules),
sentence-count enforcement, image-media-type sniffing, and the shared system prompt.

Gemini is the only opener client this project ships (the legacy Anthropic/Claude path
has been removed from operation_love/opener/opener.py entirely -- see
operation_love/opener/service.py and config.py for the corresponding provider-level
enforcement). Most tests below call the module-level helpers directly since they're
pure functions with no provider dependency; a few OpenerParseError edge cases
(missing "opener" key, a non-int referenced_index, an opener over the two-sentence
cap) are exercised through GeminiOpener with an injected fake transport, the same
technique tests/test_gemini_opener.py uses for its own (much larger) REST/quota/
thinking-config coverage. No SDK/network either way.
"""
import json

import pytest

from operation_love.opener.opener import (
    GeminiOpener,
    OpenerParseError,
    _ANCHOR_SYSTEM,
    _SYSTEM,
    _image_media_type,
    _sanitize,
    _sentence_count,
    _system_text,
)
from operation_love.perception.capture import Profile


def test_image_media_type_detects_png():
    assert _image_media_type(b"\x89PNG\r\n\x1a\n" + b"rest") == "image/png"


def test_image_media_type_detects_jpeg():
    assert _image_media_type(b"\xff\xd8\xff" + b"rest") == "image/jpeg"


def test_image_media_type_detects_gif():
    assert _image_media_type(b"GIF89a" + b"rest") == "image/gif"
    assert _image_media_type(b"GIF87a" + b"rest") == "image/gif"


def test_image_media_type_detects_webp():
    assert _image_media_type(b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"rest") == "image/webp"


def test_image_media_type_defaults_to_png_for_unknown_bytes():
    assert _image_media_type(b"not an image") == "image/png"
    assert _image_media_type(b"") == "image/png"
    assert _image_media_type(b"\x00\x01") == "image/png"


def test_sanitize_removes_em_dash_and_hyphen():
    out = _sanitize("Matcha and yoga — but cry-in-the-car energy")
    assert "—" not in out and "-" not in out and "–" not in out


def test_sanitize_credentials_lose_the_hyphen():
    assert _sanitize("PA-C energy") == "PA C energy"


def test_sanitize_no_dangling_comma_from_boundary_dash():
    assert _sanitize("your dog—") == "your dog"          # trailing dash -> no trailing comma
    assert _sanitize("—start") == "start"                # leading dash -> no leading comma
    assert _sanitize("Bold move—!") == "Bold move!"      # dash before terminal -> clean
    assert "," not in _sanitize("nice try—")             # no stray comma anywhere


# Every dash-like codepoint a model has been observed to emit (owner hard rule: NO em dashes,
# NO hyphens of any kind in a generated opener -- it's the single biggest AI-written tell).
# Property-style: loop over the codepoints so a newly-encountered dash is a one-line addition.
_ALL_DASH_CODEPOINTS = [
    "—",  # — em dash
    "–",  # – en dash
    "-",  # -  hyphen-minus
    "‐",  # ‐ hyphen
    "‑",  # ‑ non-breaking hyphen
    "‒",  # ‒ figure dash
    "―",  # ― horizontal bar
    "−",  # − minus sign
    "﹘",  # ﹘ small em dash
    "﹣",  # ﹣ small hyphen-minus
    "－",  # － fullwidth hyphen-minus
]


def test_sanitize_strips_every_dash_codepoint():
    for ch in _ALL_DASH_CODEPOINTS:
        out = _sanitize(f"left{ch}right")
        assert ch not in out, f"dash codepoint U+{ord(ch):04X} survived sanitize: {out!r}"


def test_sentence_count_basic_cases():
    assert _sentence_count("One profile-specific thought") == 1
    assert _sentence_count("One thought. One easy question?") == 2
    assert _sentence_count("Dr. Dolittle energy. What's the story?") == 2   # abbreviation guard
    assert _sentence_count("One. Two? Three!") == 3


def test_system_prompt_keeps_faithful_corey_opener_policy_and_two_sentence_cap():
    """_SYSTEM is sent verbatim by GeminiOpener (see test_gemini_opener.py's request-shape
    assertions); its actual wording is checked here directly, once, independent of any
    provider's request format."""
    lowered = _SYSTEM.lower()
    assert "90/10 framework" in _SYSTEM
    assert "genuinely curious" in lowered
    assert "do not force teasing into every opener" in lowered
    assert "one open, easy-to-answer question" in lowered
    assert "positive, fun conversation" in lowered
    assert "brief greeting is optional" in lowered
    assert "two sentences is the absolute maximum" in lowered
    assert "exactly one concrete detail" in lowered
    assert "never use an em dash or any hyphen" in lowered
    assert "low investment so she chases" not in lowered
    assert "tease her like a bratty little sister" not in lowered


def test_system_text_appends_anchor_system_only_when_anchored():
    """_system_text is the single place that decides whether a request's systemInstruction
    carries the anchor addendum (see GeminiOpener.generate(), which derives its own
    `anchored` flag once and calls this for every payload). Pinned directly, independent of
    any request-building machinery, the same way _SYSTEM's own wording is pinned above: the
    unanchored case must stay byte-identical to plain _SYSTEM (the compatibility guarantee
    every un-anchored request depends on -- see test_gemini_opener.py's own byte-identical
    request test), while the anchored case must be _SYSTEM with _ANCHOR_SYSTEM appended, and
    must say plainly that the extra image isn't part of her profile."""
    assert _system_text(False) == _SYSTEM
    assert _system_text(True) == _SYSTEM + _ANCHOR_SYSTEM
    assert _system_text(True) != _SYSTEM
    assert _system_text(True).startswith(_SYSTEM)
    assert "is NOT part of her profile" in _system_text(True)


# ---------------------------------------------------------------------------------------
# OpenerParseError edge cases, exercised through GeminiOpener + an injected fake transport
# (GeminiOpener is the only opener client shipped; these parsing rules live in
# GeminiOpener._parse but are provider-independent in spirit -- referenced_index coercion,
# a required "opener" key, and the two-sentence cap all trace back to the shared _SYSTEM
# contract and _sentence_count/_sanitize above).
# ---------------------------------------------------------------------------------------

class _Transport:
    def __init__(self, responses):
        self.responses = iter(responses)

    def __call__(self, url, payload, headers, timeout, *, method="POST"):
        return next(self.responses)


def _gemini_response(structured: dict) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps(structured)}]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
    }


def _opener(transport) -> GeminiOpener:
    return GeminiOpener(["gemini-test"], api_key="test-key", transport=transport)


def test_generate_defaults_bad_referenced_index_to_zero():
    payload = _gemini_response({"opener": "hi", "referenced": "x", "referenced_index": "notanint"})
    result = _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
    assert result.referenced_index == 0


def test_generate_raises_parse_error_on_missing_opener_key():
    payload = _gemini_response({"referenced": "x", "referenced_index": 0})   # no "opener" key
    with pytest.raises(OpenerParseError):
        _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")


def test_generate_raises_parse_error_when_opener_exceeds_two_sentence_maximum():
    payload = _gemini_response({"opener": "One. Two? Three!", "referenced": "x", "referenced_index": 0})
    with pytest.raises(OpenerParseError, match="two-sentence maximum"):
        _opener(_Transport([(200, payload)])).generate(Profile(photos=[b"a"]), style="s")
