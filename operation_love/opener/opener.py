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
import re
from dataclasses import dataclass
from typing import Protocol

from ..costing import Usage
from ..perception.capture import Profile
from ..typography import DASH_TRANSLATION, tidy_punctuation_spacing

_SCHEMA = {
    "type": "object",
    "properties": {
        "opener": {"type": "string", "description": "The message to send, bare text only. Maximum two sentences. No em dash, no hyphen."},
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
    "You write the opening message a man sends a woman on a dating app. Use the dating and "
    "conversational principles associated with Coach Corey Wayne's 'How to Be a 3% Man', without "
    "copying his wording. Be a charming gentleman first: relaxed, confident, playful, genuinely "
    "curious, and direct without pressure. Carry the spirit of his 90/10 framework across the "
    "interaction: default to sincere interest and easy confidence, and reserve light teasing or "
    "cheeky humor for the occasional profile where it arises naturally. Do not force teasing into "
    "every opener. Ground this opener in exactly ONE concrete detail from a specific photo or "
    "prompt. Make one clear, positive, profile-specific bid, then leave room for her reply. Favor a "
    "sincere observation, a direct low-pressure invitation, or one open, easy-to-answer question "
    "about that detail. Questions should invite positive, fun conversation, not form an interview. "
    "Any teasing must be clearly good-natured and never belittling, arrogant, condescending, or "
    "mean. Mild innuendo is eligible only when her own profile clearly invites that playful tone; "
    "never force it. A brief greeting is optional but cannot substitute for profile-specific "
    "substance. At most one authentic, specific compliment is allowed; never pile on flattery or "
    "seek approval. Do not act as if intimacy or romantic interest already exists. Keep the tone "
    "non-needy and do not demand that she chase. APPLICATION RULE: ONE short sentence is preferred "
    "and TWO sentences is the absolute maximum. A second sentence may be one easy positive question "
    "or a direct low-pressure invitation. Do not try to build a text relationship in the opener. "
    "HARD RULE: never use an em dash or any hyphen; use commas or periods instead (write 'physician "
    "assistant', not 'PA-C'). The "
    "images are her profile in scroll order; set referenced_index to the 0-based index of the image "
    "your opener is about. Follow the style guide. Output only the structured result."
)

_COMMON_ABBREVIATION_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc)\.", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"(?:[!?]+|\.+)(?=(?:[\"'”’)]*)?(?:\s+|$))")


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


def _sentence_count(text: str) -> int:
    """Count ordinary message sentences for the hard two-sentence send guard.

    This is intentionally narrower than a prose tokenizer: openers are short plain
    messages, while protecting common abbreviations avoids rejecting harmless text
    such as ``Dr. Dolittle energy. What's the story?``.
    """
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return 0
    protected = _COMMON_ABBREVIATION_RE.sub(
        lambda match: match.group(0).replace(".", "\u2024"), cleaned)
    protected = re.sub(r"\b(?:e\.g|i\.e)\.",
                       lambda match: match.group(0).replace(".", "\u2024"),
                       protected, flags=re.IGNORECASE)
    endings = _SENTENCE_END_RE.findall(protected)
    return max(1, len(endings))


class OpenerError(RuntimeError):
    """Claude's response couldn't be turned into an opener (refusal, or truncated/
    malformed structured output) — distinct from a transport/billing failure."""


@dataclass
class OpenerResult:
    opener: str
    referenced: str
    usage: Usage
    model: str
    referenced_index: int = 0


class OpenerParseError(OpenerError):
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
        # Capture usage/model BEFORE parsing: Anthropic bills for the call the moment it
        # returns, whether or not the body parses into a usable opener — a refusal and a
        # truncated response are both billed. Every failure below therefore raises
        # OpenerParseError, which carries usage/model so the caller can still record spend.
        usage = Usage.from_response(resp.usage)
        model = getattr(resp, "model", self.model)
        if getattr(resp, "stop_reason", None) == "refusal":
            raise OpenerParseError(
                f"Claude refused to generate an opener (model={self.model!r})", usage, model)
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise OpenerParseError(
                f"Claude returned no text content (stop_reason={getattr(resp, 'stop_reason', None)!r})",
                usage, model)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            truncated = " (output was likely truncated — raise opener.max_tokens)" \
                if getattr(resp, "stop_reason", None) == "max_tokens" else ""
            raise OpenerParseError(
                f"Claude's opener output wasn't valid JSON{truncated}: {exc}", usage, model) from exc
        try:
            opener = data["opener"]
        except (KeyError, TypeError) as e:
            raise OpenerParseError(f"{type(e).__name__}: {e}", usage, model) from e
        try:
            idx = max(0, int(data.get("referenced_index", 0)))
        except (TypeError, ValueError):
            idx = 0
        sanitized = _sanitize(opener)
        if _sentence_count(sanitized) > 2:
            raise OpenerParseError(
                "Claude returned an opener longer than the two-sentence maximum",
                usage, model)
        return OpenerResult(
            opener=sanitized,
            referenced=data.get("referenced", "").strip(),
            usage=usage,
            model=model,
            referenced_index=idx,
        )
