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

_SCHEMA = {
    "type": "object",
    "properties": {
        "opener": {"type": "string", "description": "The message to send, bare text only."},
        "referenced": {"type": "string", "description": "The specific profile detail it references."},
    },
    "required": ["opener", "referenced"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You write the opening message a man sends a woman he matched with on a dating app. "
    "Read her photos and profile text, then write one opener that references a specific, "
    "concrete detail. Follow the style guide exactly. Output only the structured result."
)


@dataclass
class OpenerResult:
    opener: str
    referenced: str
    usage: Usage
    model: str


class OpenerClient(Protocol):
    def generate(self, profile: Profile, style: str) -> OpenerResult: ...


class AnthropicOpener:
    """Claude-backed opener writer. Uses multimodal input (photos + text)."""

    def __init__(self, model: str, max_tokens: int = 400, client=None):
        self.model = model
        self.max_tokens = max_tokens
        if client is None:
            import anthropic  # imported lazily so tests don't need the SDK
            client = anthropic.Anthropic()
        self.client = client

    def _content(self, profile: Profile, style: str) -> list[dict]:
        blocks: list[dict] = []
        for img in profile.photos:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(img).decode(),
                },
            })
        blocks.append({
            "type": "text",
            "text": (
                f"STYLE GUIDE:\n{style}\n\n"
                f"HER PROFILE TEXT:\n{profile.text_blob() or '(none)'}\n\n"
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
        text = next(b.text for b in resp.content if getattr(b, "type", None) == "text")
        data = json.loads(text)
        return OpenerResult(
            opener=data["opener"].strip(),
            referenced=data.get("referenced", "").strip(),
            usage=Usage.from_response(resp.usage),
            model=getattr(resp, "model", self.model),
        )
