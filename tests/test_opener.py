"""AnthropicOpener: the no-dash sanitizer (Corey-Wayne rules) + referenced_index parsing.

No SDK/network: a fake Anthropic client returns a canned structured-output payload.
"""
import json

from operation_love.opener.opener import AnthropicOpener, _sanitize
from operation_love.perception.capture import Profile


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


class _Usage:
    input_tokens = output_tokens = cache_read_input_tokens = cache_creation_input_tokens = 0


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Resp:
    model = "claude-test"

    def __init__(self, text):
        self.content = [_Block(text)]
        self.usage = _Usage()


class _FakeAnthropic:
    def __init__(self, payload):
        self.payload = payload
        self.messages = self

    def create(self, **_):
        return _Resp(self.payload)


def test_generate_parses_index_and_sanitizes_dashes():
    payload = json.dumps({"opener": "Matcha and yoga — noted", "referenced": "matcha", "referenced_index": 3})
    op = AnthropicOpener("claude-test", client=_FakeAnthropic(payload))
    res = op.generate(Profile(photos=[b"a", b"b"]), style="be cool")
    assert res.referenced_index == 3
    assert "—" not in res.opener and "-" not in res.opener


def test_generate_defaults_bad_index_to_zero():
    payload = json.dumps({"opener": "hi", "referenced": "x", "referenced_index": "notanint"})
    op = AnthropicOpener("claude-test", client=_FakeAnthropic(payload))
    res = op.generate(Profile(photos=[b"a"]), style="s")
    assert res.referenced_index == 0
