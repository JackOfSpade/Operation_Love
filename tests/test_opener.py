"""AnthropicOpener: the no-dash sanitizer (Corey-Wayne rules) + referenced_index parsing.

No SDK/network: a fake Anthropic client returns a canned structured-output payload.
"""
import json

import pytest

from operation_love.opener.opener import (AnthropicOpener, OpenerError, OpenerParseError,
                                          _SYSTEM, _image_media_type, _sanitize,
                                          _sentence_count)
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


class _Usage:
    input_tokens = output_tokens = cache_read_input_tokens = cache_creation_input_tokens = 0


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Resp:
    model = "claude-test"

    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)] if text is not None else []
        self.usage = _Usage()
        self.stop_reason = stop_reason


class _FakeAnthropic:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload = payload
        self.stop_reason = stop_reason
        self.messages = self
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _Resp(self.payload, self.stop_reason)


def test_generate_parses_index_and_sanitizes_dashes():
    payload = json.dumps({"opener": "Matcha and yoga — noted", "referenced": "matcha", "referenced_index": 3})
    op = AnthropicOpener("claude-test", client=_FakeAnthropic(payload))
    res = op.generate(Profile(photos=[b"a", b"b"]), style="be cool")
    assert res.referenced_index == 3
    assert "—" not in res.opener and "-" not in res.opener


def test_generate_sends_faithful_corey_opener_policy_and_structured_schema():
    payload = json.dumps({"opener": "That pottery mug has a story. What happened?",
                          "referenced": "pottery", "referenced_index": 0})
    client = _FakeAnthropic(payload)
    op = AnthropicOpener("claude-test", client=client)
    op.generate(Profile(photos=[b"a"], bio="Weekend potter"), style="custom style")

    request = client.last_kwargs
    system = request["system"]
    lowered = system.lower()
    assert request["system"] == _SYSTEM
    assert "90/10 framework" in system
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
    schema = request["output_config"]["format"]["schema"]
    assert set(schema["required"]) == {"opener", "referenced", "referenced_index"}

    text_block = request["messages"][0]["content"][-1]["text"]
    assert "STYLE GUIDE:\ncustom style" in text_block
    assert "HER PROFILE TEXT:\nWeekend potter" in text_block
    assert "profile in scroll order" in text_block


def test_sentence_counter_and_generate_enforce_absolute_two_sentence_maximum():
    assert _sentence_count("One profile-specific thought") == 1
    assert _sentence_count("One thought. One easy question?") == 2
    assert _sentence_count("Dr. Dolittle energy. What's the story?") == 2
    assert _sentence_count("One. Two? Three!") == 3

    payload = json.dumps({"opener": "One. Two? Three!",
                          "referenced": "x", "referenced_index": 0})
    op = AnthropicOpener("claude-test", client=_FakeAnthropic(payload))
    with pytest.raises(OpenerParseError, match="two-sentence maximum"):
        op.generate(Profile(photos=[b"a"]), style="s")


def test_generate_defaults_bad_index_to_zero():
    payload = json.dumps({"opener": "hi", "referenced": "x", "referenced_index": "notanint"})
    op = AnthropicOpener("claude-test", client=_FakeAnthropic(payload))
    res = op.generate(Profile(photos=[b"a"]), style="s")
    assert res.referenced_index == 0


def test_content_declares_png_media_type_for_real_png_bytes():
    # regression guard: every real capture path (Playwright screenshot, adb screencap) is PNG,
    # but the code used to hardcode "image/jpeg" — the API 400s on a mismatched media_type.
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"rest of a fake png"
    op = AnthropicOpener("claude-test", client=_FakeAnthropic("{}"))
    blocks = op._content(Profile(photos=[png_bytes]), style="be cool")
    image_blocks = [b for b in blocks if b["type"] == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"]["media_type"] == "image/png"


# regression: a call that returns 200 with real billed usage but a body that doesn't parse
# into a usable opener must raise OpenerParseError carrying usage/model (not swallow the
# spend) -- the caller (OpenerService) uses that to still record what Anthropic billed.

def test_generate_raises_parse_error_on_invalid_json_and_preserves_usage():
    op = AnthropicOpener("claude-test", client=_FakeAnthropic("not json at all"))
    with pytest.raises(OpenerParseError) as exc_info:
        op.generate(Profile(photos=[b"a"]), style="s")
    assert exc_info.value.usage is not None
    assert exc_info.value.model == "claude-test"


def test_generate_raises_parse_error_on_missing_opener_key():
    payload = json.dumps({"referenced": "x", "referenced_index": 0})  # no "opener" key
    op = AnthropicOpener("claude-test", client=_FakeAnthropic(payload))
    with pytest.raises(OpenerParseError):
        op.generate(Profile(photos=[b"a"]), style="s")


class _EmptyContentResp:
    model = "claude-test"

    def __init__(self):
        self.content = []   # no text block at all
        self.usage = _Usage()


class _FakeAnthropicNoText:
    def __init__(self):
        self.messages = self

    def create(self, **_):
        return _EmptyContentResp()


def test_generate_raises_parse_error_when_no_text_block():
    op = AnthropicOpener("claude-test", client=_FakeAnthropicNoText())
    with pytest.raises(OpenerParseError):
        op.generate(Profile(photos=[b"a"]), style="s")


def test_generate_raises_opener_error_on_refusal():
    op = AnthropicOpener("claude-test", client=_FakeAnthropic("", stop_reason="refusal"))
    with pytest.raises(OpenerError):
        op.generate(Profile(photos=[b"a"]), style="s")


def test_generate_raises_opener_error_when_no_text_block():
    client = _FakeAnthropic(None)
    with pytest.raises(OpenerError):
        op = AnthropicOpener("claude-test", client=client)
        op.generate(Profile(photos=[b"a"]), style="s")


def test_generate_raises_opener_error_on_truncated_json():
    # max_tokens hit mid-JSON -> not parseable
    op = AnthropicOpener("claude-test", client=_FakeAnthropic('{"opener": "hi"', stop_reason="max_tokens"))
    with pytest.raises(OpenerError, match="truncated"):
        op.generate(Profile(photos=[b"a"]), style="s")
