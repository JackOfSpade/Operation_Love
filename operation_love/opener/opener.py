"""Opener generation via Claude, with enforced JSON output.

Structured outputs guarantee the model returns exactly {opener, referenced} —
no "Sure! Here's a great opener:" preamble (the problem that killed the original
ChatGPT attempt). The provider is behind a small interface so it stays swappable.

Returns the parsed opener AND the token usage, so the caller can record spend
and enforce the per-run budget (see operation_love.costing).
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Protocol

from ..costing import Usage
from ..perception.capture import Profile
from ..typography import DASH_TRANSLATION, tidy_punctuation_spacing

_SCHEMA = {
    "type": "object",
    "properties": {
        "opener": {"type": "string", "description": "The message to send, bare text only. No em dash, no hyphen."},
        "referenced": {"type": "string", "description": "The specific profile detail it references."},
        "referenced_index": {
            "type": "integer",
            "description": "0-based index, in the order the images were given (profile scroll order), "
                           "of the image whose photo/prompt your opener is about.",
        },
    },
    "required": ["opener", "referenced", "referenced_index"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You write the opening message a man sends a woman on a dating app, in the VOICE of dating "
    "coach Corey Wayne ('How to Be a 3% Man'): short, playful, teasing, cocky and funny, relaxed "
    "and confident. ONE line, two at most. Low investment so SHE chases. Tease her like a bratty "
    "little sister (never mean). Reference ONE concrete detail from a specific photo or prompt. "
    "NO interview-style questions, NO long earnest paragraphs, NO compliments on her looks, NO "
    "greeting ('hey'/'hi'), NO pet names. HARD RULE: never use an em dash or any hyphen; use commas "
    "or periods instead (write 'physician assistant', not 'PA-C'). The images are her profile in "
    "scroll order; set referenced_index to the 0-based index of the image your opener is about. "
    "Follow the style guide. Output only the structured result."
)


def _image_media_type(data: bytes) -> str:
    """Sniff the real image format from magic bytes. The Anthropic API 400s if the declared
    media_type doesn't match the actual bytes, so we can't just hardcode one. Defaults to PNG
    since every capture path here (Playwright screenshot, adb screencap) produces PNG."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _sanitize(text: str) -> str:
    """Enforce the no-dash opener rule as a safety net (the prompt also instructs it): every
    dash-like codepoint (em/en dash, hyphen, and their lookalikes) becomes a comma or space
    via the canonical table in operation_love.typography, then collapse whitespace and tidy
    punctuation. OWNER HARD RULE: no em dashes, no hyphens of any kind -- it's the single
    biggest AI-written tell."""
    t = str(text).translate(DASH_TRANSLATION)
    return tidy_punctuation_spacing(t)


@dataclass
class OpenerResult:
    opener: str
    referenced: str
    usage: Usage
    model: str
    referenced_index: int = 0


class OpenerParseError(Exception):
    """The API call reached Anthropic and was billed (it returned token usage), but the
    response body didn't parse into a usable opener (bad JSON, missing keys, no text
    block). Carries usage/model so the caller can still record the spend -- Anthropic
    charges for the call whether or not we could parse a usable result out of it."""

    def __init__(self, message: str, usage: Usage, model: str):
        super().__init__(message)
        self.usage = usage
        self.model = model


class OpenerClient(Protocol):
    def generate(self, profile: Profile, style: str) -> OpenerResult: ...


class AnthropicOpener:
    """Claude-backed opener writer. Uses multimodal input (photos + text)."""

    def __init__(self, model: str, max_tokens: int = 400, request_timeout_s: float = 30,
                 client=None):
        self.model = model
        self.max_tokens = max_tokens
        if client is None:
            import anthropic  # imported lazily so tests don't need the SDK
            client = anthropic.Anthropic(timeout=request_timeout_s)
        self.client = client

    def _content(self, profile: Profile, style: str) -> list[dict]:
        blocks: list[dict] = []
        for img in profile.photos:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _image_media_type(img),
                    "data": base64.standard_b64encode(img).decode(),
                },
            })
        blocks.append({
            "type": "text",
            "text": (
                f"STYLE GUIDE:\n{style}\n\n"
                f"HER PROFILE TEXT:\n{profile.text_blob() or '(none)'}\n\n"
                f"The {len(profile.photos)} image(s) above are her profile in scroll order "
                "(index 0 first). Set referenced_index to the index of the one your opener is about. "
                "Write the opener now."
            ),
        })
        return blocks

    def generate(self, profile: Profile, style: str) -> OpenerResult:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": self._content(profile, style)}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
        # Capture usage/model before parsing: Anthropic bills for the call the moment it
        # returns, regardless of whether the body below parses into a usable opener, so a
        # parse failure must still surface what was billed (see OpenerParseError).
        usage = Usage.from_response(resp.usage)
        model = getattr(resp, "model", self.model)
        try:
            text = next(b.text for b in resp.content if getattr(b, "type", None) == "text")
            data = json.loads(text)
            opener = data["opener"]
        except (StopIteration, json.JSONDecodeError, KeyError, TypeError) as e:
            raise OpenerParseError(f"{type(e).__name__}: {e}", usage, model) from e
        try:
            idx = max(0, int(data.get("referenced_index", 0)))
        except (TypeError, ValueError):
            idx = 0
        return OpenerResult(
            opener=_sanitize(opener),
            referenced=data.get("referenced", "").strip(),
            usage=usage,
            model=model,
            referenced_index=idx,
        )
