"""Repeatable live audit of which Gemini models THIS account can actually use.

config.yaml's opener.models comment states the hard rule this tool exists to enforce: a model
may only enter opener.models after it has been EMPIRICALLY VERIFIED to return a valid opener
using this project's exact request shape (inline image + responseJsonSchema + thinkingConfig).
ListModels is NOT sufficient on its own -- MEASURED fact, recorded in that same comment and in
GeminiOpener.preflight()'s docstring: gemini-2.5-flash, gemini-2.5-flash-lite and
gemini-2.5-pro are all still listed by ListModels with "generateContent" in
supportedGenerationMethods, yet every one of them 404s NOT_FOUND on every real generateContent
call for this account. This tool used to be a one-off manual audit; it is now a command anyone
can re-run the next time the cascade needs revisiting.

FOUR STEPS, always in this order:

  1. ENUMERATE  -- GET ListModels, following nextPageToken pagination. Free: it does not spend
     any model's generateContent per-day quota (see enumerate_models()).
  2. CLASSIFY    -- split the enumerated, generateContent-capable ids into three buckets:
     already configured (opener.models), plausible general-purpose text+image candidates, and
     excluded kinds (image-generation, tts, live/omni, robotics, music/lyria, embedding,
     computer-use, deep-research/agent -- see _EXCLUSION_RULES). Exclusion is by explicit
     substring rule, each with its own comment, not a denylist of exact ids that rots the day
     Google renames a model.
  3. PROBE       -- for each candidate, issue a REAL generateContent call built by
     GeminiOpener._payload against a SYNTHETIC, non-personal profile (a generated solid-colour
     PNG and a fixed neutral bio -- never real captured data). See probe_model() for the
     thinking-variant escalation rule and the exact verdict vocabulary.
  4. REPORT      -- print, per model: its verdict, the thinkingConfig that actually worked (if
     any), any QuotaFailure detail, and a truncated preview of the returned opener. For every
     USABLE model, also emit a ready-to-paste config.yaml snippet covering all three places
     config.py's validate() requires an entry (opener.models, opener.thinking, budget.pricing),
     so wiring in a verified model is copy/paste rather than hand-editing three blocks and
     forgetting one.

COST. Every PROBE call is a REAL, billed generateContent request against that model's own
per-day free-tier quota (measured: as low as 20 RPD for the flash tier -- see config.yaml's
opener.models comment). By default this tool never re-probes a model already in
opener.models, because that would spend PRODUCTION capacity rather than audit capacity; pass
--include-configured to opt in. Before issuing a single billed request this tool prints exactly
which models it is about to probe and the worst-case request count, and requires either --yes
or an interactive "yes" (see main()'s COST TRANSPARENCY section).

The API key is never printed, logged, or written anywhere by this tool, including on an error
path (see _redact()). It is loaded from the environment, falling back to this project's own
.env exactly the way operation_love.__main__.main() and tools/hinge_calibrate.py's main() do
(see _load_dotenv()) -- never by walking parent directories.

TESTABILITY. Every network call in this module goes through an injected GeminiTransport
(operation_love.opener.opener.GeminiTransport: a ``(url, payload, headers, timeout, *,
method="POST") -> (code, body)`` callable), exactly like GeminiOpener(transport=...). main()
accepts one directly so tests never touch the real network; only __main__'s own invocation of
main() uses the real stdlib transport.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote

import yaml

from operation_love import config as cfg_mod
from operation_love.opener.opener import (
    GeminiOpener,
    GeminiTransport,
    INDEX_SPACE_MODEL_ITEMS,
    ItemRequest,
    OpenerError,
    _GEMINI_MODELS_LIST_URL,
    _MAX_PREFLIGHT_PAGES,
    _gemini_error,
    _stdlib_gemini_transport,
)
from operation_love.perception.capture import Profile
from operation_love.private_files import load_private_dotenv

# ---------------------------------------------------------------------------------------
# Verdict vocabulary -- exactly these seven, nothing else. USABLE is the only one that means
# "safe to add to opener.models"; every other verdict is a reason NOT to, spelled out precisely
# enough that an operator does not have to go re-read a traceback to know why.
# ---------------------------------------------------------------------------------------
VERDICT_USABLE = "USABLE"
VERDICT_RETIRED_404 = "RETIRED_404"
VERDICT_QUOTA_429 = "QUOTA_429"
VERDICT_BUSY_5XX = "BUSY_5XX"
VERDICT_TRANSPORT = "TRANSPORT"
VERDICT_BAD_REQUEST_400 = "BAD_REQUEST_400"
VERDICT_UNUSABLE_RESPONSE = "UNUSABLE_RESPONSE"


# ---------------------------------------------------------------------------------------
# CLASSIFY -- explicit substring exclusion rules, not a denylist of exact ids.
#
# Each rule is (substring, human-readable reason), matched case-insensitively against the bare
# model id (the part after "models/"). Google renames and re-versions model ids constantly
# (e.g. an image-capable chat variant might ship today as "gemini-2.0-flash-preview-image-
# generation" and next month under a different version number) -- a denylist of exact ids goes
# stale the day that happens and silently stops excluding. A substring keyed to the FAMILY name
# keeps matching regardless of the version number attached to it.
#
# Every rule below excludes a kind of model that supports generateContent (so it would
# otherwise reach the candidate list) but cannot answer this project's opener request the way a
# general-purpose text+image chat model can -- it returns image bytes, audio, robot actions, or
# is built for a different call shape entirely (a streaming session, a multi-step agent loop).
# ---------------------------------------------------------------------------------------
_EXCLUSION_RULES: tuple[tuple[str, str], ...] = (
    # Image-generation/editing models (Imagen, and "chat" variants that emit image bytes, e.g.
    # "gemini-2.0-flash-preview-image-generation", "gemini-2.5-flash-image"): they return image
    # data, not the structured JSON opener this project's responseJsonSchema requires.
    ("imagen", "image-generation model (Imagen family); returns image bytes, not JSON"),
    ("-image", "image-generation/editing model; returns image bytes, not JSON"),
    # Text-to-speech models return audio, never JSON.
    ("tts", "text-to-speech model; returns audio, not JSON"),
    # Live/omni streaming voice-and-video session models (Gemini Live, native-audio dialog):
    # built for a bidirectional streaming session, not a single generateContent call.
    ("live", "live/omni streaming voice-and-video model; not a single-call generateContent model"),
    ("native-audio", "live/omni native-audio dialog model; not a single-call generateContent model"),
    # Embodied/robotics action models emit robot control tokens, not conversational prose.
    ("robotics", "embodied/robotics action model; not a conversational text model"),
    # Music generation (Lyria and friends).
    ("lyria", "music-generation model; not a text+image chat model"),
    ("music", "music-generation model; not a text+image chat model"),
    # Embedding-only models have no chat/generation behaviour at all. Listed defensively --
    # ListModels should already omit "generateContent" for these, but the rule stays explicit
    # rather than relying on that.
    ("embedding", "embedding-only model; cannot generate an opener"),
    # Computer-use / browser-control agent models emit UI actions, not prose.
    ("computer-use", "computer-use agent model; not a conversational text model"),
    # Deep-research / autonomous multi-step agent ids are tuned for long tool-use loops, not a
    # single structured-JSON opener call.
    ("deep-research", "deep-research agent model; not a single-call opener model"),
    ("-agent", "autonomous agent model; not a single-call opener model"),
    # MANAGED-AGENT ids served by the Interactions API (POST /v1beta/interactions), NOT by
    # generateContent in this project's request shape. antigravity-preview-05-2026 is listed by
    # ListModels WITH "generateContent" in supportedGenerationMethods, so nothing above catches
    # it -- and it was measured (2026-08-13) to be unusable here for a second, independent
    # reason: the Antigravity agent does not support structured output at all, and every opener
    # this project sends carries responseJsonSchema. Probing it can only ever burn quota.
    ("antigravity", "managed-agent model (Interactions API); no structured-output support, so "
                    "it can never serve a responseJsonSchema opener"),
    # Marketing aliases for image-generation models whose ID contains no "-image" substring, so
    # the "-image" rule above cannot see them. "Nano Banana" is Google's own product name for
    # the gemini-*-image family (nano-banana-pro-preview is the gemini-3-pro-image alias), and
    # an alias is exactly the case a substring rule on the technical id misses. Without this
    # rule the tool spends up to six billed requests discovering a model returns image bytes --
    # something already knowable from its published identity.
    ("nano-banana", "image-generation model (Nano Banana alias for the gemini-*-image family); "
                    "returns image bytes, not JSON"),
)


def excluded_reason(model_id: str) -> str | None:
    """The exclusion reason for ``model_id``, or None if no rule matches."""
    lowered = model_id.lower()
    for substring, reason in _EXCLUSION_RULES:
        if substring in lowered:
            return reason
    return None


@dataclass(frozen=True)
class Classification:
    configured: tuple[str, ...]
    candidates: tuple[str, ...]
    excluded: Mapping[str, str]


def classify_models(model_ids: Iterable[str], configured: Iterable[str]) -> Classification:
    """Split generateContent-capable ``model_ids`` into configured / candidate / excluded.

    ``configured`` wins over an exclusion match (a model the owner already wired in is reported
    as configured, not excluded, even if it happens to match a substring rule) -- this function
    only decides what to PRINT and what the default probe set is; it is never asked to second-
    guess an owner's existing config.yaml.
    """
    configured_set = set(configured)
    configured_out: list[str] = []
    candidates: list[str] = []
    excluded: dict[str, str] = {}
    for model_id in sorted(set(model_ids)):
        if model_id in configured_set:
            configured_out.append(model_id)
            continue
        reason = excluded_reason(model_id)
        if reason:
            excluded[model_id] = reason
            continue
        candidates.append(model_id)
    return Classification(tuple(configured_out), tuple(candidates), excluded)


# ---------------------------------------------------------------------------------------
# ENUMERATE
# ---------------------------------------------------------------------------------------

def enumerate_models(*, api_key: str, transport: GeminiTransport, timeout: float) -> dict[str, list[str]]:
    """GET ListModels, following nextPageToken pagination. Returns {model_id: methods}.

    FREE: ListModels is not a generateContent call, so it does not consume any model's
    per-day free-tier quota -- only the PROBE step below does that. Mirrors
    GeminiOpener.preflight()'s own pagination loop exactly, because that loop is already the
    project's proven-correct implementation of this exact API; this function only differs in
    keeping the full per-id methods list instead of collapsing straight to a pass/fail check.
    """
    seen: dict[str, list[str]] = {}
    page_token: str | None = None
    seen_page_tokens: set[str] = set()
    for _page_number in range(1, _MAX_PREFLIGHT_PAGES + 1):
        # Percent-encoded, bounded and repeat-refusing, exactly as preflight is. A page token is
        # an opaque server string: a raw '+' in one is decoded back as a space and a raw '&' or
        # '#' truncates the query, so an unencoded token turns page 2 onward into a 400 that
        # aborts the whole audit -- in a tool whose entire job is to see the FULL catalog before
        # any billed probe is spent.
        url = (_GEMINI_MODELS_LIST_URL if page_token is None else
               f"{_GEMINI_MODELS_LIST_URL}?pageToken={quote(page_token, safe='')}")
        code, response = transport(url, None, {"X-goog-api-key": api_key}, timeout, method="GET")
        code = int(code)
        if not 200 <= code < 300:
            error = _gemini_error(code, response)
            raise RuntimeError(
                f"ListModels failed: HTTP {error.http_code} {error.status}: {error.message}")
        if not isinstance(response, Mapping):
            raise RuntimeError("ListModels returned a malformed response (not a JSON object)")
        for entry in response.get("models") or []:
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                continue
            model_id = name.split("/", 1)[1] if "/" in name else name
            methods = entry.get("supportedGenerationMethods")
            seen[model_id] = [str(m) for m in methods] if isinstance(methods, list) else []
        next_token = response.get("nextPageToken")
        if next_token in (None, ""):
            break
        if not isinstance(next_token, str) or not next_token.strip():
            raise RuntimeError("ListModels returned a malformed nextPageToken")
        if next_token in seen_page_tokens:
            raise RuntimeError("ListModels repeated a pagination token")
        seen_page_tokens.add(next_token)
        page_token = next_token
    else:
        raise RuntimeError(f"ListModels exceeded {_MAX_PREFLIGHT_PAGES} pages")
    return seen


def generate_content_ids(catalog: Mapping[str, list[str]]) -> list[str]:
    """Ids in ``catalog`` (as returned by enumerate_models) that list generateContent support."""
    return sorted(model_id for model_id, methods in catalog.items() if "generateContent" in methods)


# ---------------------------------------------------------------------------------------
# PROBE -- the synthetic request, the thinking-variant escalation, and per-model classification.
# ---------------------------------------------------------------------------------------

_SYNTHETIC_BIO = (
    "Synthetic Operation Love model-verification profile. This bio and photo are generated "
    "for the sole purpose of confirming a Gemini model can serve this project's opener "
    "generateContent request shape. No personal data of any kind."
)


def _solid_color_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """A minimal, valid, solid-colour RGB PNG, built with only zlib/struct (no PIL dependency
    needed just to make a synthetic probe image). Used instead of any real captured photo --
    see this module's docstring and the owner's rule that a live probe must never touch real
    captured data."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit depth, RGB color type
    row = bytes([0]) + bytes(rgb) * width  # filter-type byte 0 (none) + width solid RGB pixels
    raw = row * height
    idat = zlib.compress(raw, 9)
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def build_synthetic_profile_and_items() -> tuple[Profile, ItemRequest]:
    """The one synthetic, non-personal (Profile, ItemRequest) pair every probe call reuses."""
    png = _solid_color_png(64, 64, (128, 128, 128))
    profile = Profile(bio=_SYNTHETIC_BIO, prompts=[])
    item_request = ItemRequest(items=(png,), name="Model Probe", context=(), truncated=False)
    return profile, item_request


# Discovery order for generationConfig.thinkingConfig, per this module's docstring and the
# owner's task: four thinkingLevel values (the 3.x model family), then the older thinkingBudget
# field (0 disables thinking; the 2.5 family), then omitted entirely (let the model's own
# server-side default apply). A label paired with None means "send no thinkingConfig at all"
# -- see probe_model()'s use of it below.
_THINKING_VARIANTS: tuple[tuple[str, dict[str, Any] | None], ...] = (
    ("thinkingLevel=minimal", {"thinkingLevel": "minimal"}),
    ("thinkingLevel=low", {"thinkingLevel": "low"}),
    ("thinkingLevel=medium", {"thinkingLevel": "medium"}),
    ("thinkingLevel=high", {"thinkingLevel": "high"}),
    ("thinkingBudget=0", {"thinkingBudget": 0}),
    ("omitted", None),
)


def is_thinking_related_400(message: str) -> bool:
    """True when a 400's error message specifically names the thinking config.

    Real Gemini 400s for a bad thinkingConfig name the offending field verbatim (e.g. an
    "Unknown name \\"thinking_level\\": Cannot find field" for a model that doesn't support
    thinkingLevel, or a message naming thinkingBudget as unsupported/out of range) -- so a
    case-insensitive substring check on the word "thinking" is precise enough to tell "wrong
    variant, try the next one" apart from every other 400 (bad schema, oversized request,
    permission, ...) without having to parse Gemini's free-text error prose more closely.
    """
    return "thinking" in (message or "").lower()


def quota_detail(body: Any) -> dict[str, str] | None:
    """quotaId/quotaMetric/quotaValue out of a 429's structured QuotaFailure detail, if the
    body carries one -- not every 429 does (see GeminiAPIError's own docstring)."""
    if not isinstance(body, Mapping):
        return None
    error = body.get("error")
    if not isinstance(error, Mapping):
        return None
    for entry in error.get("details") or []:
        if not isinstance(entry, Mapping):
            continue
        if "QuotaFailure" not in str(entry.get("@type", "")):
            continue
        for violation in entry.get("violations") or []:
            if not isinstance(violation, Mapping):
                continue
            quota_id = violation.get("quotaId")
            quota_metric = violation.get("quotaMetric")
            quota_value = violation.get("quotaValue")
            if quota_id or quota_metric or quota_value:
                return {
                    "quotaId": str(quota_id) if quota_id is not None else "",
                    "quotaMetric": str(quota_metric) if quota_metric is not None else "",
                    "quotaValue": str(quota_value) if quota_value is not None else "",
                }
    return None


@dataclass
class ProbeAttempt:
    thinking_label: str
    thinking_config: dict[str, Any] | None
    http_code: int | None
    outcome: str
    message: str = ""


@dataclass
class ProbeResult:
    model: str
    verdict: str
    thinking_config: dict[str, Any] | None = None   # only set when verdict == USABLE
    thinking_label: str = ""
    quota: dict[str, str] | None = None             # only set when verdict == QUOTA_429
    opener_text: str = ""                           # only set when verdict == USABLE, truncated
    message: str = ""
    attempts: list[ProbeAttempt] = field(default_factory=list)


_OPENER_PREVIEW_LEN = 160


def probe_model(model: str, *, api_key: str, transport: GeminiTransport, timeout: float,
                 max_tokens: int, style: str, profile: Profile,
                 item_request: ItemRequest) -> ProbeResult:
    """Probe one model, escalating through _THINKING_VARIANTS in order.

    ONLY a 400 whose message specifically names the thinking config advances to the next
    variant -- every other outcome (200, 404, 429, 5xx, a transport failure, or a 400 that is
    NOT about thinking) stops this model's probe immediately. Retrying a different thinking
    level against, say, a 404 or a per-day 429 would just burn more of that model's quota for
    no new information: the model is not going to become available, or stop being over quota,
    because a different thinkingConfig was sent.

    Never raises: every branch (including an unexpected exception from _payload/_parse) is
    caught and turned into a ProbeResult, because this is an operator tool and a traceback
    here would abort the whole audit over one model.
    """
    attempts: list[ProbeAttempt] = []
    for label, thinking_cfg in _THINKING_VARIANTS:
        thinking_map = {model: dict(thinking_cfg)} if thinking_cfg is not None else {}
        try:
            opener = GeminiOpener([model], max_tokens=max_tokens, request_timeout_s=timeout,
                                  api_key=api_key, transport=transport, thinking=thinking_map)
            payload = opener._payload(profile, style, model, items=item_request)
        except Exception as exc:  # noqa: BLE001 -- request-building must never crash the audit
            attempts.append(ProbeAttempt(label, thinking_cfg, None, "build-error",
                                         f"{type(exc).__name__}: {exc}"))
            return ProbeResult(model, VERDICT_BAD_REQUEST_400,
                               message=f"could not build the request: {type(exc).__name__}: {exc}",
                               attempts=attempts)

        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{_url_quote(model)}:generateContent")
        headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
        try:
            code, response = transport(url, payload, headers, timeout)
        except OSError as exc:
            message = f"{type(exc).__name__}: {exc}"
            attempts.append(ProbeAttempt(label, thinking_cfg, None, "transport", message))
            return ProbeResult(model, VERDICT_TRANSPORT, message=message, attempts=attempts)

        code = int(code)
        if 200 <= code < 300:
            if not isinstance(response, Mapping):
                attempts.append(ProbeAttempt(label, thinking_cfg, code, "malformed-2xx"))
                return ProbeResult(model, VERDICT_UNUSABLE_RESPONSE,
                                   message="malformed success response (not a JSON object)",
                                   attempts=attempts)
            try:
                result = opener._parse(response, model, index_space=INDEX_SPACE_MODEL_ITEMS,
                                       numbered_item_count=item_request.item_count)
            except OpenerError as exc:
                attempts.append(ProbeAttempt(label, thinking_cfg, code, "unusable", str(exc)))
                return ProbeResult(model, VERDICT_UNUSABLE_RESPONSE, message=str(exc),
                                   attempts=attempts)
            attempts.append(ProbeAttempt(label, thinking_cfg, code, "usable"))
            opener_text = result.opener
            preview = (opener_text if len(opener_text) <= _OPENER_PREVIEW_LEN
                      else opener_text[:_OPENER_PREVIEW_LEN - 3] + "...")
            return ProbeResult(model, VERDICT_USABLE, thinking_config=thinking_cfg,
                               thinking_label=label, opener_text=preview,
                               message="returned a valid opener", attempts=attempts)

        error = _gemini_error(code, response)
        if code == 429:
            attempts.append(ProbeAttempt(label, thinking_cfg, code, "quota", error.message))
            return ProbeResult(model, VERDICT_QUOTA_429, quota=quota_detail(response),
                               message=error.message, attempts=attempts)
        if code == 404:
            attempts.append(ProbeAttempt(label, thinking_cfg, code, "retired", error.message))
            return ProbeResult(model, VERDICT_RETIRED_404, message=error.message, attempts=attempts)
        if code >= 500:
            attempts.append(ProbeAttempt(label, thinking_cfg, code, "busy", error.message))
            return ProbeResult(model, VERDICT_BUSY_5XX, message=error.message, attempts=attempts)
        if code == 400 and is_thinking_related_400(error.message):
            attempts.append(ProbeAttempt(label, thinking_cfg, code, "thinking-400", error.message))
            continue  # advance to the next thinking variant
        # Any other 4xx (a 400 unrelated to thinking, 401, 403, ...): a property of the
        # request/credentials, not of which thinkingConfig was sent -- stop immediately.
        attempts.append(ProbeAttempt(label, thinking_cfg, code, "bad-request", error.message))
        return ProbeResult(model, VERDICT_BAD_REQUEST_400, message=error.message, attempts=attempts)

    last = attempts[-1] if attempts else None
    return ProbeResult(
        model, VERDICT_BAD_REQUEST_400,
        message=("every thinking variant, including omitted, was rejected as a thinking-"
                 f"related 400; last: {last.message if last else '(no attempts)'}"),
        attempts=attempts)


def _url_quote(model: str) -> str:
    from urllib.parse import quote
    return quote(model, safe="-_.")


# ---------------------------------------------------------------------------------------
# YAML snippet
# ---------------------------------------------------------------------------------------

_SNIPPET_HEADER = (
    "# Paste `models` into opener.models, `thinking` into opener.thinking, and `pricing` into\n"
    "# budget.pricing -- config.py's validate() requires an entry in all three for every\n"
    "# configured model. Update pricing off the Google free tier before enabling billing (see\n"
    "# config.yaml's budget.pricing comment).\n"
)


def build_yaml_snippet(usable: list[tuple[str, dict[str, Any] | None]]) -> str:
    """A ready-to-paste config.yaml fragment for every USABLE model.

    ``usable`` is a list of (model_id, working_thinking_config) pairs -- working_thinking_config
    is None for the "omitted" variant, which config.py documents as the explicit, sanctioned
    way to say "use this model's server default" ({}). Returns "" when ``usable`` is empty (no
    model to paste).
    """
    if not usable:
        return ""
    models = [model for model, _ in usable]
    thinking = {model: (dict(cfg) if cfg else {}) for model, cfg in usable}
    pricing = {model: {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0}
              for model, _ in usable}
    doc = {"models": models, "thinking": thinking, "pricing": pricing}
    return _SNIPPET_HEADER + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TIMEOUT_S = 30.0
_FALLBACK_STYLE = (
    "Be a charming, curious, low-pressure gentleman. Make one clear, positive, specific "
    "observation or guess about the item shown, then optionally one easy question. Two "
    "sentences maximum. No em dash or hyphen. Write in the spoken register a person texts "
    "in, with natural contractions, and deliver any compliment as an offhand remark about "
    "the thing rather than a verdict on her. (Placeholder style guide -- config.yaml's real "
    "opener.style could not be loaded; this has no bearing on whether the account can serve "
    "this request shape.)"
)


def _redact(text: str, secret: str | None) -> str:
    """Never let the API key reach stdout/stderr, even on an error path."""
    if not text or not secret:
        return text
    return text.replace(secret, "<redacted-api-key>")


def _load_dotenv() -> None:
    """Load this project's own .env, exactly like operation_love.__main__.main() and
    tools/hinge_calibrate.py's main() -- never walking parent directories, rejecting unsafe
    link leaves, and tightening a real file to owner-only mode before python-dotenv reads it."""
    load_private_dotenv(Path.cwd() / ".env")


def _interactive_confirm(prompt: str) -> bool:
    try:
        reply = input(prompt)
    except EOFError:
        return False
    return reply.strip().lower() in ("y", "yes")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/gemini_model_probe.py",
        description="Empirically verify which Gemini models this account can actually use "
                     "with this project's exact opener request shape (inline image + "
                     "responseJsonSchema + thinkingConfig). ListModels alone is NOT sufficient "
                     "-- see config.yaml's opener.models comment: gemini-2.5-flash, "
                     "gemini-2.5-flash-lite, and gemini-2.5-pro are all still listed by "
                     "ListModels yet 404 NOT_FOUND on every real generateContent call for "
                     "this account.")
    parser.add_argument("--models", default=None,
                        help="comma-separated model ids to probe directly, bypassing "
                             "auto-classification (an id already in opener.models is still "
                             "skipped by default -- see --include-configured)")
    parser.add_argument("--include-configured", action="store_true",
                        help="also (re-)probe models already listed in opener.models. OFF by "
                             "default: probing a model already in production spends that "
                             "model's own per-day free-tier quota, which is production "
                             "capacity, not audit capacity.")
    parser.add_argument("--json", action="store_true",
                        help="emit one JSON report to stdout instead of the human-readable one")
    parser.add_argument("--yes", action="store_true",
                        help="proceed with billed probing without an interactive prompt")
    parser.add_argument("--config", default="config.yaml",
                        help="config.yaml to read opener.models/opener.style/opener.thinking "
                             "from (default config.yaml; never modified)")
    return parser


def _json_report(classification: Classification, results: list[ProbeResult], snippet: str) -> dict:
    return {
        "configured": list(classification.configured),
        "excluded": dict(classification.excluded),
        "candidates": list(classification.candidates),
        "results": [
            {
                "model": r.model,
                "verdict": r.verdict,
                "thinking_config": r.thinking_config,
                "thinking_label": r.thinking_label,
                "quota": r.quota,
                "opener_text": r.opener_text,
                "message": r.message,
            }
            for r in results
        ],
        "yaml_snippet": snippet or None,
    }


def _print_result(result: ProbeResult, api_key: str, *, log: Callable[..., None]) -> None:
    log(f"\n{result.model}: {result.verdict}")
    if result.thinking_label:
        shown = result.thinking_config if result.thinking_config is not None else "(server default)"
        log(f"  working thinkingConfig: {result.thinking_label} -> {shown}")
    if result.quota:
        log(f"  quota detail: {result.quota}")
    if result.opener_text:
        log(f"  opener (truncated): {result.opener_text!r}")
    if result.message:
        log(f"  detail: {_redact(result.message, api_key)}")


def main(argv: list[str] | None = None, *, transport: GeminiTransport | None = None,
         env: Mapping[str, str] | None = None,
         confirm: Callable[[str], bool] | None = None) -> int:
    """CLI entry point. ``transport``/``env``/``confirm`` are the injection seams tests use to
    keep this hermetic (no network, no real .env, no interactive stdin) -- see this module's
    docstring's TESTABILITY paragraph. Real usage (the __main__ block below) passes none of
    them, so it loads .env, reads the real environment, and prompts on the real stdin/stdout
    exactly like every other CLI tool in this project.
    """
    args = _build_arg_parser().parse_args(argv)
    # In --json mode stdout carries ONLY the final JSON document (so a caller can pipe it
    # straight into `jq`/`json.load` without stripping narrative text first); every other
    # message this function prints goes through `log`, which redirects to stderr in that mode
    # and behaves like plain `print` (to stdout) otherwise.
    log: Callable[..., None] = (
        (lambda *a, **k: print(*a, file=sys.stderr, **k)) if args.json else print)

    if env is None:
        _load_dotenv()
        environment: Mapping[str, str] = os.environ
    else:
        environment = env
    api_key = environment.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set (checked the environment and .env).",
              file=sys.stderr)
        return 1

    live_transport = transport or _stdlib_gemini_transport

    configured_models: list[str] = []
    style = _FALLBACK_STYLE
    max_tokens = _DEFAULT_MAX_TOKENS
    request_timeout_s = _DEFAULT_TIMEOUT_S
    try:
        cfg = cfg_mod.load(args.config)
        configured_models = list(cfg.opener.effective_models)
        style = cfg.opener.style or _FALLBACK_STYLE
        max_tokens = cfg.opener.max_tokens or _DEFAULT_MAX_TOKENS
        request_timeout_s = cfg.opener.request_timeout_s or _DEFAULT_TIMEOUT_S
    except Exception as exc:  # noqa: BLE001 -- config is informational here, never fatal
        print(f"WARNING: could not read {args.config!r} ({type(exc).__name__}: {exc}); "
              "proceeding with no known already-configured models and a placeholder style "
              "guide.", file=sys.stderr)

    log("=== ENUMERATE (ListModels; free, consumes no generateContent quota) ===")
    try:
        catalog = enumerate_models(api_key=api_key, transport=live_transport,
                                   timeout=request_timeout_s)
    except RuntimeError as exc:
        print(f"ERROR: {_redact(str(exc), api_key)}", file=sys.stderr)
        return 1
    ids = generate_content_ids(catalog)
    log(f"{len(catalog)} model id(s) listed; {len(ids)} support generateContent.")

    log("\n=== CLASSIFY ===")
    classification = classify_models(ids, configured_models)
    log(f"Already configured (opener.models): {list(classification.configured) or '(none)'}")
    log(f"Excluded by kind ({len(classification.excluded)}):")
    for model_id in sorted(classification.excluded):
        log(f"  - {model_id}: {classification.excluded[model_id]}")
    log(f"Plausible general-purpose candidates: {list(classification.candidates) or '(none)'}")

    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        unseen = [m for m in requested if m not in ids]
        if unseen:
            log(f"WARNING: {unseen} were not seen in ListModels' generateContent list "
                "(the catalog can lag reality -- that lag is this tool's whole reason for "
                "existing); probing them anyway.")
    else:
        requested = list(classification.candidates)
        if args.include_configured:
            requested += list(configured_models)

    configured_set = set(configured_models)
    to_probe: list[str] = []
    skipped_configured: list[str] = []
    seen_probe: set[str] = set()
    for model_id in requested:
        if model_id in configured_set and not args.include_configured:
            skipped_configured.append(model_id)
            continue
        if model_id in seen_probe:
            continue
        seen_probe.add(model_id)
        to_probe.append(model_id)

    if skipped_configured:
        log(f"\nSkipping already-configured model(s) (default; probing them spends "
            f"production quota): {skipped_configured}. Pass --include-configured to probe "
            "them too.")

    if not to_probe:
        log("\nNothing to probe.")
        if args.json:
            print(json.dumps(_json_report(classification, [], ""), indent=2))
        return 0

    log("\n=== COST TRANSPARENCY ===")
    log(f"Up to {len(_THINKING_VARIANTS)} billed generateContent request(s) will be issued "
        f"against EACH of the following {len(to_probe)} model(s) (a model's probe stops "
        "earlier the moment a non-thinking-config outcome is reached):")
    for model_id in to_probe:
        log(f"  - {model_id}")
    log(f"Worst case total: {len(to_probe) * len(_THINKING_VARIANTS)} billed request(s). "
        "Each request spends one call against that model's own per-day free-tier quota "
        "(measured: as low as 20 RPD for the flash tier).")

    if not args.yes:
        ask = confirm or _interactive_confirm
        if not ask("Proceed with these billed generateContent calls? [y/N] "):
            log("Aborted -- no requests were issued.")
            return 1

    profile, item_request = build_synthetic_profile_and_items()
    log("\n=== PROBE ===")
    results: list[ProbeResult] = []
    for model_id in to_probe:
        try:
            result = probe_model(model_id, api_key=api_key, transport=live_transport,
                                 timeout=request_timeout_s, max_tokens=max_tokens, style=style,
                                 profile=profile, item_request=item_request)
        except Exception as exc:  # noqa: BLE001 -- an operator tool must never traceback
            result = ProbeResult(model_id, VERDICT_UNUSABLE_RESPONSE,
                                 message=f"unexpected {type(exc).__name__}: {exc}")
        results.append(result)
        _print_result(result, api_key, log=log)

    usable = [(r.model, r.thinking_config) for r in results if r.verdict == VERDICT_USABLE]
    snippet = build_yaml_snippet(usable)

    if args.json:
        print(json.dumps(_json_report(classification, results, snippet), indent=2))
        return 0

    log("\n=== SUMMARY ===")
    for result in results:
        log(f"  {result.model}: {result.verdict}")
    if snippet:
        log("\n=== READY-TO-PASTE config.yaml SNIPPET ===")
        log(snippet.rstrip())
    else:
        log("\nNo model was verified USABLE this run -- nothing to paste.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
