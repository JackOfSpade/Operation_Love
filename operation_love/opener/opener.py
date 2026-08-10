"""Provider-backed opener generation, with enforced JSON output.

Structured outputs guarantee the model returns exactly {opener, referenced} —
no "Sure! Here's a great opener:" preamble (the problem that killed the original
ChatGPT attempt). The provider is behind a small interface so it stays swappable.

Returns the parsed opener AND the token usage, so the caller can record spend
and enforce the per-run budget (see operation_love.costing).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

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

# Cap on how much of a malformed model-output value gets echoed into an error message --
# long enough to be diagnostic, short enough that a huge/garbage payload can't blow up a
# log line or the hub's stop-reason display.
_MAX_ERROR_REPR_LEN = 200


def _truncated_repr(value: Any) -> str:
    """repr() a model-output value for an operator-facing error message, truncated so a
    pathological payload (e.g. a multi-KB string where a short opener was expected)
    doesn't dominate the message."""
    text = repr(value)
    if len(text) > _MAX_ERROR_REPR_LEN:
        text = text[:_MAX_ERROR_REPR_LEN] + "...(truncated)"
    return text


def _image_media_type(data: bytes) -> str:
    """Sniff the real image format from magic bytes. Gemini 400s if the declared mimeType
    doesn't match the actual bytes, so we can't just hardcode one. Defaults to PNG since
    every capture path here (Playwright screenshot, adb screencap) produces PNG."""
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
    """A provider response could not be turned into an opener (refusal, or truncated/
    malformed structured output) — distinct from a transport/billing failure."""


@dataclass
class OpenerResult:
    opener: str
    referenced: str
    usage: Usage
    model: str
    referenced_index: int = 0


class OpenerParseError(OpenerError):
    """A billed API response did not parse into a usable opener.

    Carries normalized usage/model so the caller can still record a provider's spend
    after bad JSON, missing keys, or a response without usable text.
    """

    def __init__(self, message: str, usage: Usage, model: str):
        super().__init__(message)
        self.usage = usage
        self.model = model


class OpenerAborted(OpenerError):
    """The run is stopping: ``should_stop()`` reported true mid-cascade, before the next
    model's request was issued (see GeminiOpener.generate()'s per-iteration check).

    Deliberately NOT the same thing as an ordinary OpenerError: a corrupt photo or a
    provider refusal says something is wrong with THIS profile's request, while this says
    nothing about the request at all -- the operator clicked Stop (or the supervisor is
    shutting down) and the cascade is unwinding on purpose. Callers (OpenerService.
    maybe_opener) must treat the two very differently: an ordinary OpenerError is a real
    per-profile failure that counts toward the transient-failure latch, but a deliberate
    shutdown must not retry, must not call _exhaust(), and must not poison the service's
    health state (the latch counters, exhausted_reason) with what is not a provider
    failure at all -- see BUG 1's fix. Message text says plainly that the run is stopping,
    not that a request failed, so an operator reading a log line (or last_skip_reason on
    the hub) is never left thinking the provider did something wrong.
    """


class OpenerClient(Protocol):
    def generate(self, profile: Profile, style: str, retry_hint: str = "", *,
                 should_stop: Callable[[], bool] | None = None,
                 skip_models: frozenset[str] = frozenset()) -> OpenerResult: ...


class GeminiAPIError(RuntimeError):
    """A non-success Gemini GenerateContent response, without request credentials.

    ``status`` is Gemini's machine-readable error status (for example,
    ``RESOURCE_EXHAUSTED``); ``http_code`` is the HTTP status.  Keeping those fields
    separate lets the fallback policy distinguish actual capacity exhaustion from a
    malformed request that also happens to use a 4xx status.

    ``quota_id``/``quota_metric`` are populated only for a 429 whose body carries a
    structured ``google.rpc.QuotaFailure`` detail; both are ``None`` for every other
    error and for a 429 whose body doesn't include that detail (some do not). They
    exist so the fallback policy can tell a transient per-minute cap apart from an
    actual per-day exhaustion instead of treating every 429 identically.
    """

    def __init__(self, http_code: int, status: str | None, message: str, *,
                 quota_id: str | None = None, quota_metric: str | None = None):
        self.http_code = int(http_code)
        self.status = status or ""
        self.message = message
        self.quota_id = quota_id
        self.quota_metric = quota_metric
        detail = f"Gemini API HTTP {self.http_code}"
        if self.status:
            detail += f" {self.status}"
        super().__init__(f"{detail}: {message}")


class GeminiCapacityExhausted(RuntimeError):
    """Every configured Gemini model returned 429 RESOURCE_EXHAUSTED this run."""


# Callable[...] rather than a precise positional signature: preflight() calls this with an
# extra keyword-only `method="GET"` that generate()'s POST calls never pass, and Callable
# can't express "these args, plus this optional kwarg" precisely.
GeminiTransport = Callable[..., tuple[int, Any]]

# ListModels (used by preflight()) has no request body of its own.
_GEMINI_MODELS_LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Gemini hard-caps a request's TOTAL encoded size (text + system instruction + inline image
# bytes) at 20MB. We budget under that with real headroom: base64 already inflates raw photo
# bytes by ~33%, our size estimate doesn't count the generationConfig/schema JSON scaffolding,
# and it's cheaper to compress a bit more than necessary than to 400 a nearly-fitting request.
_MAX_INLINE_REQUEST_BYTES = 18 * 1024 * 1024


def _first_quota_violation(details: Any) -> tuple[str | None, str | None]:
    """Pull ``quotaId``/``quotaMetric`` out of a 429's ``error.details[]``, if present.

    Defensive by design: Google does not guarantee every 429 body includes a structured
    QuotaFailure (and third-party proxies/mocks in tests may omit it entirely), so any
    shape mismatch here must degrade to "no violation found" rather than raise -- the
    caller's classifier already treats that as "unknown" and handles it safely.
    """
    if not isinstance(details, list):
        return None, None
    for entry in details:
        if not isinstance(entry, dict):
            continue
        entry_type = entry.get("@type")
        if not isinstance(entry_type, str) or "QuotaFailure" not in entry_type:
            continue
        violations = entry.get("violations")
        if not isinstance(violations, list):
            continue
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            quota_id = violation.get("quotaId")
            quota_metric = violation.get("quotaMetric")
            if quota_id or quota_metric:
                return (
                    str(quota_id) if quota_id is not None else None,
                    str(quota_metric) if quota_metric is not None else None,
                )
    return None, None


def _classify_quota_exhaustion(error: GeminiAPIError) -> str:
    """Classify a 429 as ``"day"``, ``"minute"``, or ``"unknown"`` from its quota details.

    Real free-tier quota ids/metrics spell the period both as "PerDay"/"PerMinute" (id
    casing) and "per_day"/"per_minute" (metric, underscored), so both spellings are
    checked case-insensitively against whichever field the response populated.
    """
    haystack = " ".join(filter(None, [error.quota_id, error.quota_metric])).lower()
    if "perday" in haystack or "per_day" in haystack:
        return "day"
    if "perminute" in haystack or "per_minute" in haystack:
        return "minute"
    return "unknown"


_QUOTA_SCOPE_LABELS = {"day": "per-day quota", "minute": "per-minute throttle",
                       "unknown": "unclassified 429", "gone": "model unavailable",
                       "busy": "provider 5xx", "transport": "network or timeout"}

# Scopes that clear on their own without any operator action. "busy" is a provider-side 5xx
# (503 UNAVAILABLE "this model is currently experiencing high demand" is the common one, and
# it is genuinely per-MODEL -- observed live on gemini-3.6-flash while the rest of the
# cascade was healthy), so it belongs with the per-minute caps rather than with the dead
# ends: retrying shortly, or on another model right now, is the correct response to all three.
_TRANSIENT_SCOPES = frozenset({"minute", "unknown", "busy", "transport"})


def _exhaustion_reason(scopes: Mapping[str, str]) -> str:
    """Explain why the whole cascade fell through, precisely enough to act on.

    This string becomes the run's stop reason in the hub, so it must not conflate the very
    different situations that can end the cascade. Each model's scope is one of: "day"
    (per-day 429, resets at midnight Pacific), "minute"/"unknown" (a transient per-minute or
    unclassifiable 429 that clears on its own within roughly a minute), or "gone" (HTTP 404
    NOT_FOUND -- the model id is retired / not available to this account and will NEVER come
    back mid-run; see generate()'s 404 handling and preflight()'s docstring for why a
    startup ListModels pass cannot catch this in advance).

    Every model hitting its per-DAY quota means there is genuinely no opener capacity left
    until the midnight Pacific reset -- that case keeps its own message below. Every model
    coming back "gone" is the opposite kind of dead end: no amount of waiting fixes a
    retired model id, so telling the operator to wait for a reset would be actively wrong;
    that case gets its own message too, pointing at opener.models instead of the clock.
    Anything else (a mix of scopes, or a transient per-minute/unknown 429 in the mix) falls
    through to the generic listing, which already assumes at least one transient cause may
    clear on its own shortly. We still stop the run in every case (see OpenerService),
    because sending a bare like with no opener is a worse outcome than halting; only the
    guidance differs.
    """
    if not scopes:                      # unreachable today (__init__ requires >=1 model)
        return "no configured Gemini model was available to serve the request"
    listed = ", ".join(f"{model} ({_QUOTA_SCOPE_LABELS.get(scope, scope)})"
                       for model, scope in scopes.items())
    if all(scope == "day" for scope in scopes.values()):
        return (f"every configured Gemini model has exhausted its per-day free-tier quota "
                f"({', '.join(scopes)}); free-tier daily quota resets at midnight Pacific")
    if all(scope == "gone" for scope in scopes.values()):
        # Distinct from the per-day case on purpose: mentioning a reset time here would
        # imply waiting helps, and it never does for a retired model id.
        return (f"every configured Gemini model is unavailable to this account (retired or "
                f"not found: {', '.join(scopes)}); this will not resolve on its own -- fix "
                "opener.models to name model ids this account can actually use")
    if all(scope in _TRANSIENT_SCOPES for scope in scopes.values()):
        # Nothing here is a real dead end: every model was either momentarily throttled or
        # reported a provider-side 5xx. Naming a quota reset would send the operator away
        # for hours over something that typically clears in seconds.
        return (f"no configured Gemini model could serve the request right now: {listed}. "
                "Every one of these is a transient failure (a per-minute cap or a provider "
                "side outage), not an exhausted daily quota, so simply restarting should "
                "succeed")
    return (f"no configured Gemini model could serve the request: {listed}. At least one of "
            "these is a transient per-minute cap rather than a per-day exhaustion, so "
            "restarting in a minute may well succeed instead of waiting for the midnight "
            "Pacific daily reset")


def _gemini_error(status_code: int, body: Any) -> GeminiAPIError:
    """Normalize a Gemini REST error body without echoing request credentials."""
    if not isinstance(body, dict):
        return GeminiAPIError(status_code, None, str(body) or "empty error response")
    error = body.get("error")
    if not isinstance(error, dict):
        return GeminiAPIError(status_code, None, "malformed error response")
    code = error.get("code", status_code)
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = status_code
    quota_id, quota_metric = _first_quota_violation(error.get("details"))
    return GeminiAPIError(code, error.get("status"), str(error.get("message") or "unknown error"),
                          quota_id=quota_id, quota_metric=quota_metric)


def _stdlib_gemini_transport(url: str, payload: dict[str, Any] | None, headers: dict[str, str],
                             timeout: float, *, method: str = "POST") -> tuple[int, Any]:
    """POST (the default, used by generate()) or GET (used by preflight()'s ListModels
    call) with the standard library; tests replace this whole transport.

    A GET must never carry a JSON body -- ListModels takes none, and some HTTP stacks
    reject a body on GET outright -- so ``data`` stays None whenever method isn't POST.
    """
    data = json.dumps(payload).encode("utf-8") if method == "POST" and payload is not None else None
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is fixed Gemini endpoint
            raw = response.read().decode("utf-8")
            return int(response.status), json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body: Any = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = raw
        return int(exc.code), body


class GeminiOpener:
    """Gemini GenerateContent opener writer with run-scoped ordered model fallback.

    Any HTTP 429 is treated as a capacity signal for that model (see generate()'s 429
    handling for why the status code alone, not Gemini's own ``RESOURCE_EXHAUSTED`` status
    string, is the reliable trigger). A 429 that DOES classify as a per-day quota means that
    model cannot serve the rest of the current run, so it is skipped thereafter and the same
    profile is attempted with the next configured model; a per-minute/unknown 429 cascades
    without being blacklisted. An HTTP 404 ``NOT_FOUND`` gets the permanent treatment: it
    means only THAT model id is retired / not available to this account, not the request
    itself, so it is permanently dropped from the cascade and the next configured model is
    tried instead -- see generate()'s 404 handling for why a 404 must not be allowed to take
    the rest of the cascade down with it. An HTTP 5xx and a TRANSPORT-level failure (a
    ``socket.timeout``, a ``urllib.error.URLError`` from a connection reset or DNS failure,
    or any other ``OSError`` the transport raises instead of returning -- see generate()'s
    ``OSError`` handling) both get the same non-blacklisting cascade as a per-minute 429:
    neither says anything about whether the model would answer the NEXT request, only that
    it failed to answer this one, so it cascades to the next configured model for this
    profile only and is retried first on the next profile. Authentication, permission, and
    malformed-request errors, and any other 4xx, are deliberately raised to the caller
    unchanged rather than silently cascading: those are properties of the request or
    credentials, not of one model id or one flaky connection, so they would fail identically
    on every other configured model too.

    THREAD SAFETY: instances are safe to share across worker threads. generate() acquires
    this instance's own internal lock for its full duration, so the model cascade and the
    ``_unavailable_models`` retirement it performs are atomic per call -- this class does not
    rely on an external caller (OpenerService) to serialize access on its behalf. Lock
    ordering: OpenerService.maybe_opener acquires ITS OWN lock first and calls into
    generate() while holding it, so the effective order is always "OpenerService's lock,
    then this one" -- and this lock is never held while calling back out into OpenerService
    or anything else that could re-enter it, so the two locks cannot deadlock against each
    other.
    """

    def __init__(self, models: list[str] | tuple[str, ...], max_tokens: int = 400,
                 request_timeout_s: float = 30, *, api_key: str | None = None,
                 env: Mapping[str, str] | None = None,
                 transport: GeminiTransport | None = None,
                 thinking: Mapping[str, Mapping[str, Any]] | None = None):
        self.models = tuple(str(model) for model in models if str(model).strip())
        if not self.models:
            raise ValueError("GeminiOpener requires at least one configured model")
        environment = os.environ if env is None else env
        self.api_key = api_key if api_key is not None else environment.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.max_tokens = max_tokens
        self.request_timeout_s = request_timeout_s
        self.transport = transport or _stdlib_gemini_transport
        # model id -> the exact generationConfig.thinkingConfig dict to send for that model,
        # e.g. {"thinkingLevel": "minimal"} (3.x family) or {"thinkingBudget": 0} (2.5 family).
        # A model with no entry gets no thinkingConfig at all, so its own default applies.
        self.thinking: Mapping[str, Mapping[str, Any]] = thinking or {}
        # model id -> the scope ("day" or "gone") it was permanently retired under earlier
        # THIS run. A dict rather than a set because a later all-exhausted stop must report
        # each retired model under the scope it actually failed with, not a hardcoded one --
        # a model retired for being out of per-day quota needs the "wait for midnight
        # Pacific" guidance, while one retired as 404-gone needs "fix opener.models" instead
        # (see _exhaustion_reason). Populated only by generate(); see its loop below.
        self._unavailable_models: dict[str, str] = {}
        # Self-contained lock (see the class docstring's THREAD SAFETY note): generate()
        # holds this for its entire body, so mutating self._unavailable_models and running
        # the model cascade are atomic per call regardless of what (if anything) an external
        # caller does. Previously this class relied entirely on OpenerService.maybe_opener's
        # own RLock for that safety -- correct today, but only by convention, and an
        # adversarial audit demonstrated a real double-spend race (two threads both burning a
        # billed API call on an already-retired model) the moment a call site bypasses that
        # external lock. RLock (not Lock): generate() calls other methods on self, and a
        # reentrant lock means a future refactor that has one of those helpers also take the
        # lock can't deadlock this instance against itself.
        self._lock = threading.RLock()

    def _image_parts(self, photos: list[bytes]) -> list[dict[str, Any]]:
        """Base64-encode every photo. This is the expensive step in building a request --
        full-resolution phone screenshots, then inflated ~33% by base64 -- so generate()
        computes it once per profile and reuses the result across every model tried during
        a capacity cascade instead of re-encoding the same screenshots per model."""
        return [{
            "inlineData": {
                "mimeType": _image_media_type(image),
                "data": base64.standard_b64encode(image).decode("ascii"),
            },
        } for image in photos]

    def _text_part(self, profile: Profile, style: str, retry_hint: str = "") -> dict[str, Any]:
        """Build the one part of the request that varies per attempt (see generate()'s "encode
        once, reuse across the cascade" note -- the image parts never depend on this).

        ``retry_hint`` defaults to "" (falsy) for an ordinary first attempt, in which case the
        text is byte-identical to what this produced before retries existed -- no stray
        delimiter, no trailing blank section, nothing for a diff to catch on the common path.
        When OpenerService is retrying a rejected attempt, it passes the specific reason that
        attempt was rejected for; that gets appended as its own clearly delimited block AFTER
        the profile content, so it is the most recent instruction the model reads before
        writing the corrected opener -- a corrected re-ask rather than an identical dice roll.
        """
        text = (
            f"STYLE GUIDE:\n{style}\n\n"
            f"HER PROFILE TEXT:\n{profile.text_blob() or '(none)'}\n\n"
            f"The {len(profile.photos)} image(s) above are her profile in scroll order "
            "(index 0 first). Set referenced_index to the index of the one your opener is about. "
            "Write the opener now."
        )
        if retry_hint:
            text += (
                # The two lists below are deliberately kept apart. Everything under HARD
                # REJECTION is a rule _parse() actually enforces by raising, so it is what a
                # retry must fix to succeed at all. The style rules are real owner rules but
                # are NOT rejection causes (dashes, for instance, are silently laundered by
                # _sanitize rather than rejected). Presenting the two as one undifferentiated
                # "mandatory" checklist -- as this block first did -- spends the model's
                # attention on cosmetics when the reason it failed was structural.
                "\n\n=== RETRY: YOUR PREVIOUS ATTEMPT WAS REJECTED AND NOT SENT ===\n"
                f"Reason: {retry_hint}\n"
                "Fix exactly that. HARD REJECTION RULES, checked in code, which will reject "
                "you again if broken: the 'opener' field must be a non-empty STRING (never "
                "null, a number, or an empty/whitespace value), and the opener must be at "
                "most TWO sentences. Also keep following the style guide above, especially "
                "the hard rule against em dashes and hyphens, and ground the opener in one "
                "concrete detail from her profile text or photos. Write the corrected opener "
                "now."
            )
        return {"text": text}

    def _payload(self, profile: Profile, style: str, model: str, *,
                 image_parts: list[dict[str, Any]] | None = None,
                 retry_hint: str = "") -> dict[str, Any]:
        """Build one model's GenerateContent request. ``image_parts`` lets generate() pass
        in already-encoded photos so a cascade across N models doesn't re-encode the same
        screenshots N times; when omitted it's computed fresh from ``profile``. ``retry_hint``
        is forwarded to _text_part unchanged -- it must reach EVERY model tried in this
        attempt's cascade, because it describes what the previous attempt got wrong, which
        stays true no matter which model ends up serving the retry."""
        parts = list(image_parts) if image_parts is not None else self._image_parts(profile.photos)
        parts.append(self._text_part(profile, style, retry_hint))
        generation_config: dict[str, Any] = {
            "maxOutputTokens": self.max_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": _SCHEMA,
        }
        # The thinkingConfig dict is passed through verbatim rather than derived from the
        # model id: the field name differs by model family (thinkingLevel on the 3.x line,
        # thinkingBudget on 2.5), sending the wrong one is a 400, and guessing the family
        # from a version-number prefix is exactly the kind of thing that silently breaks
        # the day a new naming scheme ships. So it's declared explicitly per model id in
        # config rather than inferred here.
        thinking_config = self.thinking.get(model)
        if thinking_config is not None:
            generation_config["thinkingConfig"] = dict(thinking_config)
        return {
            "systemInstruction": {"parts": [{"text": _SYSTEM}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }

    @staticmethod
    def _request_size_bytes(image_parts: list[dict[str, Any]], text_part: dict[str, Any]) -> int:
        """Approximate the wire size of one request against Gemini's 20MB inline-data cap:
        the base64 image payloads dominate, plus the system instruction and the per-request
        text block (style guide + profile text). generationConfig/schema JSON is a few
        hundred fixed bytes that don't scale with photo count, so it's left out of the
        estimate -- see _MAX_INLINE_REQUEST_BYTES for the headroom that covers it."""
        total = len(_SYSTEM.encode("utf-8")) + len(text_part["text"].encode("utf-8"))
        for part in image_parts:
            total += len(part["inlineData"]["data"])
        return total

    def _fit_images_to_budget(self, profile: Profile, image_parts: list[dict[str, Any]],
                              text_part: dict[str, Any]) -> list[dict[str, Any]]:
        """Guarantee the request fits Gemini's 20MB inline-image cap, compressing only if
        it doesn't.

        Happy path (a handful of already-reasonable screenshots) does zero extra work and
        returns the original encoded parts untouched. Only a profile with many
        full-resolution phone screenshots pays the recompression cost, and it's never
        silent -- we print exactly what was done so a systematically oversized capture
        pipeline is visible rather than a mysteriously smaller/blurrier opener input.
        """
        original_size = self._request_size_bytes(image_parts, text_part)
        if original_size <= _MAX_INLINE_REQUEST_BYTES:
            return image_parts

        from PIL import Image  # lazy: PIL is a project dependency, imported this same lazy
                                # way in operation_love/vision/quality.py so modules that
                                # never hit this path don't pay the import cost.

        # Step 1 is quality-85 JPEG recompression with no resize -- often enough on its own
        # for PNG phone screenshots, which carry a lot of lossless overhead. If that's still
        # over budget, progressively shrink the longest side until the request fits.
        new_size = original_size
        for max_side in (None, 1568, 1280, 1024, 768):
            recompressed: list[bytes] = []
            for index, photo in enumerate(profile.photos):
                try:
                    img = Image.open(io.BytesIO(photo)).convert("RGB")
                    if max_side is not None and max(img.size) > max_side:
                        scale = max_side / max(img.size)
                        img = img.resize(
                            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                            Image.LANCZOS,
                        )
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                except Exception as exc:  # noqa: BLE001 -- PIL raises several distinct types for
                    # a truncated/corrupt image (UnidentifiedImageError, plain OSError, even
                    # struct.error deep in the decoder for a stream that dies mid-read), and this
                    # path only runs on an oversized profile, so it's easy for a flaky `adb
                    # screencap` capture to reach it. Left as a bare PIL exception, this escaped
                    # generate() uncaught: OpenerService can't classify it as GeminiAPIError, so
                    # it fell into the generic transient branch and printed "swiping without" --
                    # identically, forever, with no escalation (adversarial audit item B). Convert
                    # it to OpenerError instead: the established "this profile's own content is
                    # bad, skip just this profile" signal, naming which photo and how big it was
                    # so the operator can tell a systemic capture bug from one bad frame.
                    raise OpenerError(
                        f"Gemini opener: photo index {index} ({len(photo)} bytes) could not be "
                        f"decoded/recompressed while fitting the request to the inline size "
                        f"budget: {type(exc).__name__}: {exc}") from exc
                recompressed.append(buf.getvalue())
            fitted_parts = [{
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": base64.standard_b64encode(photo).decode("ascii"),
                },
            } for photo in recompressed]
            new_size = self._request_size_bytes(fitted_parts, text_part)
            if new_size <= _MAX_INLINE_REQUEST_BYTES:
                print(f"Gemini opener: compressed {len(profile.photos)} photo(s) to fit the "
                      f"inline request budget ({original_size} -> {new_size} bytes, cap "
                      f"{_MAX_INLINE_REQUEST_BYTES}).")
                return fitted_parts
        # BUG 4 (adversarial audit): this used to always say "reduce photo count or
        # resolution", blaming the photos unconditionally -- but the text part (style guide +
        # profile text + a retry_hint, which can itself be a sizable corrective block; see
        # _text_part) counts against the same budget and does NOT shrink here, only the
        # images do. A large retry_hint can push an otherwise-fine profile over budget with
        # the photos barely contributing, and telling the operator to trim photos in that case
        # is actively misleading. Report the actual composition instead of assuming.
        image_bytes = sum(len(part["inlineData"]["data"]) for part in fitted_parts)
        text_bytes = len(_SYSTEM.encode("utf-8")) + len(text_part["text"].encode("utf-8"))
        raise OpenerError(
            f"Gemini opener: profile has {len(profile.photos)} photo(s); the request still "
            f"totals {new_size} bytes encoded even at the smallest compression step "
            f"({image_bytes} bytes of photos, {text_bytes} bytes of text -- style guide, "
            f"profile content, and system instruction, including any retry hint), over the "
            f"{_MAX_INLINE_REQUEST_BYTES} byte budget; refusing to silently drop photos. "
            "Reduce photo count or resolution upstream if photos dominate the total, or "
            "shorten the profile text/retry hint if text does.")

    @staticmethod
    def _usage(response: Mapping[str, Any]) -> Usage:
        metadata = response.get("usageMetadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}

        def token_count(name: str) -> int:
            try:
                return int(metadata.get(name, 0) or 0)
            except (TypeError, ValueError):
                return 0

        # Convert REST's camelCase field names to the provider-neutral normalizer.
        # In particular, cached content is a subset of prompt tokens, so the normalizer
        # prevents charging that same token count at both input and cache-read rates.
        return Usage.from_gemini(SimpleNamespace(
            prompt_token_count=token_count("promptTokenCount"),
            candidates_token_count=token_count("candidatesTokenCount"),
            thoughts_token_count=token_count("thoughtsTokenCount"),
            cached_content_token_count=token_count("cachedContentTokenCount"),
        ))

    @staticmethod
    def _text(response: Mapping[str, Any]) -> str | None:
        candidates = response.get("candidates") or []
        if not isinstance(candidates, list):
            return None
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            content = candidate.get("content") or {}
            if not isinstance(content, Mapping):
                continue
            parts = content.get("parts") or []
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    return part["text"]
        return None

    @staticmethod
    def _finish_reason(response: Mapping[str, Any]) -> str | None:
        candidates = response.get("candidates") or []
        if not isinstance(candidates, list):
            return None
        for candidate in candidates:
            if isinstance(candidate, Mapping) and isinstance(candidate.get("finishReason"), str):
                return candidate["finishReason"]
        return None

    @staticmethod
    def _thoughts_token_count(response: Mapping[str, Any]) -> int:
        metadata = response.get("usageMetadata")
        if not isinstance(metadata, Mapping):
            return 0
        try:
            return int(metadata.get("thoughtsTokenCount", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _parse(self, response: Mapping[str, Any], requested_model: str) -> OpenerResult:
        usage = self._usage(response)
        # Price against the configured model id. Gemini's optional modelVersion can be an
        # opaque serving revision rather than a key in budget.pricing.
        model = requested_model
        text = self._text(response)
        finish_reason = self._finish_reason(response)
        if text is None:
            if finish_reason == "MAX_TOKENS":
                # Thinking counts against maxOutputTokens for every model we use (all but
                # gemini-2.5-flash-lite default it on). When thoughts + candidates exceed
                # the budget, the candidate comes back with NO text part at all -- this is
                # the single most likely cause of "no text content", so name it explicitly
                # instead of leaving the operator to guess.
                thoughts = self._thoughts_token_count(response)
                raise OpenerParseError(
                    f"Gemini truncated the response before producing any text "
                    f"(finishReason=MAX_TOKENS): thinking alone used {thoughts} of "
                    f"{self.max_tokens} configured max_tokens, leaving no room for the opener "
                    f"JSON. Raise opener.max_tokens, or lower {model!r}'s opener.thinking level.",
                    usage, model)
            raise OpenerParseError(f"Gemini returned no text content (finishReason={finish_reason!r})",
                                   usage, model)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OpenerParseError(
                f"Gemini's opener output wasn't valid JSON (finishReason={finish_reason!r}): {exc}",
                usage, model) from exc
        try:
            opener = data["opener"]
        except (KeyError, TypeError) as exc:
            raise OpenerParseError(f"{type(exc).__name__}: {exc}", usage, model) from exc
        # _SCHEMA's "required": ["opener", ...] is a generation HINT sent to the model, not a
        # runtime guarantee the API enforces on the response -- Gemini can (and, per an
        # adversarial audit of this project, DOES in practice) still return {"opener": null}
        # or a non-string value despite "opener" being listed as required. Left unchecked,
        # _sanitize()'s str(text) would silently turn None into the literal string "None"
        # (truthy) or an int like 42 into "42" -- both pass the sentence-count guard below
        # and parse SUCCESSFULLY, and worker.py's `if opener: self.adb.text(opener)` then
        # types that literal text into the Hinge comment box and sends it to a real person.
        # So the type is re-verified here, at the boundary, rather than trusted.
        if not isinstance(opener, str):
            raise OpenerParseError(
                f"Gemini's opener field was not a string: received {type(opener).__name__} "
                f"{_truncated_repr(opener)}",
                usage, model)
        try:
            index = max(0, int(data.get("referenced_index", 0)))
        except (AttributeError, TypeError, ValueError):
            index = 0
        sanitized = _sanitize(opener)
        # A string that is empty, or becomes empty/whitespace-only once the dash-fold and
        # punctuation cleanup in _sanitize run, is just as unusable as a missing field --
        # sending nothing (or degrading silently to a bare like) is the same failure mode
        # this whole check exists to catch. An empty string happened to be falsy and degrade
        # safely downstream by luck alone; this makes it an explicit, named failure instead.
        if not sanitized.strip():
            raise OpenerParseError(
                f"Gemini's opener field was empty or whitespace only after sanitizing "
                f"(received {_truncated_repr(opener)})",
                usage, model)
        if _sentence_count(sanitized) > 2:
            raise OpenerParseError("Gemini returned an opener longer than the two-sentence maximum",
                                   usage, model)
        return OpenerResult(
            opener=sanitized,
            referenced=str(data.get("referenced", "")).strip(),
            usage=usage,
            model=model,
            referenced_index=index,
        )

    def generate(self, profile: Profile, style: str, retry_hint: str = "", *,
                 should_stop: Callable[[], bool] | None = None,
                 skip_models: frozenset[str] = frozenset()) -> OpenerResult:
        # retry_hint defaults to "" (falsy): an ordinary first attempt, no correction to make.
        # When OpenerService is re-asking after a rejected attempt, it passes the specific
        # reason here; _text_part appends it as a corrective instruction, and it must reach
        # EVERY model tried below, not just the first -- whichever model ends up serving the
        # retry, what the previous attempt got wrong is still what it got wrong.
        #
        # skip_models (item D of a later audit pass): generate() returns as soon as ANY model
        # answers HTTP 2xx, even when the response then fails to parse (see _parse). So a
        # parse-failure retry ordinarily restarts the cascade from the top and re-hits the
        # SAME (often scarcest-quota) model that just produced the bad response -- if that
        # model is systematically malformed, every retry attempt burns a request against it
        # while healthier models later in the cascade never even get tried. OpenerService
        # passes the model ids that already failed to parse FOR THIS PROFILE here so this
        # call's cascade steps over them, landing on the next configured model instead. A
        # skipped model is NOT recorded in self._unavailable_models: unlike a per-day 429 or a
        # 404, nothing here says the model is actually broken -- it produced a billed, well-
        # formed HTTP response, just not a usable opener -- so it stays fully eligible again on
        # the very next profile (or the next fresh call with no skip set).
        #
        # SAFETY VALVE: if every configured model is in skip_models, the set is ignored
        # entirely rather than leaving this call with nothing to try. This can only happen if
        # every model already failed to parse earlier in the SAME retry sequence -- at that
        # point a stochastic re-ask of an already-failed model still beats returning with no
        # opener at all (the owner's rule is to keep trying, not to give up early), and
        # OpenerService's own max_attempts ceiling is what eventually stops the retries, not
        # this method refusing to pick a model.
        #
        # should_stop is the caller's cheap, non-blocking "should I keep going?" check (in
        # practice, worker.py's threading.Event.is_set for the run's stop flag). BUG 1 (an
        # adversarial audit): pre-fix, nothing on this path ever consulted the stop signal, so
        # a Stop click during a live cascade (up to max_attempts x len(models) x
        # request_timeout_s -- 5 x 7 x 90s = 3150s, ~52 minutes, against the shipped config)
        # was silently ignored: the worker thread ran the full retry/cascade to completion
        # anyway, kept holding OpenerService's shared lock (blocking every other worker), could
        # write to the store AFTER the supervisor's shutdown had already closed it, and blew
        # straight past supervisor.py's own _WORKER_JOIN_TIMEOUT_S, misreporting a perfectly
        # healthy retry loop as a wedged worker. Checked at the TOP of every model iteration,
        # BEFORE that model's request is issued, so the worst case is now bounded by ONE
        # already-in-flight HTTP request (request_timeout_s), not the rest of the cascade.
        #
        # Held for the ENTIRE call, not just the self._unavailable_models mutations, so a
        # second thread calling generate() concurrently is fully serialized behind this one
        # rather than able to interleave and both hit the same about-to-be-retired model with
        # a real, billed request (see the class docstring's THREAD SAFETY note).
        with self._lock:
            # Encode once, reuse across every model tried in this call's cascade -- base64 and
            # (when a profile needs it) recompression are the expensive parts of building a
            # request, and neither depends on which model ends up serving it (nor on
            # retry_hint, which only ever varies the text part).
            image_parts = self._image_parts(profile.photos)
            text_part = self._text_part(profile, style, retry_hint)
            image_parts = self._fit_images_to_budget(profile, image_parts, text_part)
            # SAFETY VALVE (see this method's skip_models docstring paragraph above): if the
            # caller's skip set would leave literally nothing eligible, ignore it entirely
            # rather than raising GeminiCapacityExhausted without ever trying a single model.
            effective_skip_models = (
                skip_models if skip_models and any(m not in skip_models for m in self.models)
                else frozenset()
            )
            # Why each model declined to serve THIS call, so that if the whole cascade falls
            # through we can report an accurate stop reason instead of a generic one. The hub
            # shows this verbatim, and "wait until midnight Pacific" vs "retry in a minute" are
            # very different instructions to give the operator.
            scopes: dict[str, str] = {}
            for model in self.models:
                if should_stop is not None and should_stop():
                    # Checked before even the "already retired" skip below, so a stop signaled
                    # right after this model's slot comes up never issues a request for it --
                    # see this method's should_stop docstring paragraph for the full rationale.
                    raise OpenerAborted(
                        f"Opener cascade aborted before requesting {model!r}: the run is "
                        "stopping (should_stop signaled), not a provider failure")
                if model in effective_skip_models:
                    # Already produced an unusable (but well-formed, billed) response for THIS
                    # profile earlier in the same retry sequence -- not retired (see the
                    # skip_models docstring paragraph above), just deprioritized for this one
                    # call, so it is not added to `scopes` either: it was never actually tried
                    # this call, so it has nothing to report if the cascade falls through.
                    print(f"Gemini opener: skipping {model} for this retry -- it already "
                          "produced an unusable response for this profile; trying the next "
                          "configured model instead.")
                    continue
                if model in self._unavailable_models:
                    # Retired earlier THIS run -- either a per-day 429 or a 404 NOT_FOUND (see
                    # self._unavailable_models). Report the scope it actually failed under; a
                    # later all-exhausted stop needs the right guidance for each model, not a
                    # hardcoded "day" for one that was really 404-gone.
                    scopes[model] = self._unavailable_models[model]
                    continue
                payload = self._payload(profile, style, model, image_parts=image_parts,
                                        retry_hint=retry_hint)
                url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                       f"{quote(model, safe='-_.')}:generateContent")
                try:
                    code, response = self.transport(
                        url, payload,
                        {"Content-Type": "application/json", "X-goog-api-key": self.api_key},
                        self.request_timeout_s,
                    )
                except OSError as exc:
                    # TRANSPORT-level failure -- this is a layer BELOW the HTTP status-code
                    # cascade above: _stdlib_gemini_transport only catches urllib.error.
                    # HTTPError (a successful-at-the-socket-layer response that merely carries
                    # a non-2xx status), converting it into a (code, body) return value. A
                    # socket.timeout, a urllib.error.URLError (connection reset, DNS failure,
                    # refused connection, ...), or any other OSError never reaches that
                    # handling at all -- it propagates straight out of self.transport(...) to
                    # here. EMPIRICALLY OBSERVED: a live run against the real API timed out
                    # mid-request on the FIRST configured model and, pre-fix, that abandoned
                    # the entire cascade -- six other healthy, configured models were never
                    # tried and the profile got no opener. Treat it exactly like a provider
                    # 5xx ("busy"): a dropped connection says something about THIS request,
                    # not about whether this model (or any other) would answer the next one,
                    # so cascade to the next configured model for this profile only and do
                    # NOT retire it -- it stays first in line on the next profile.
                    #
                    # Caught as OSError specifically, NOT bare Exception. socket.timeout is a
                    # TimeoutError alias and urllib.error.URLError is a direct subclass, so
                    # OSError covers every real transport failure (timeout, reset, DNS, refused
                    # connection) without needing to enumerate each one. A bare `except
                    # Exception` here would be wrong: it would also swallow a TypeError or
                    # AttributeError raised by a broken transport implementation -- a
                    # programming bug in this code or an injected transport, not a flaky
                    # network -- and silently retry that bug across all 7 configured models
                    # instead of letting it surface immediately as the real error it is.
                    # Its own scope rather than reusing "busy": both are transient and both
                    # cascade identically, but the stop reason is operator-facing guidance,
                    # and reporting a connection timeout as a "provider 5xx" would point at
                    # Google when the problem may well be the local network.
                    scopes[model] = "transport"
                    print(f"Gemini opener: {model} failed at the transport level "
                          f"({type(exc).__name__}: {exc}); NOT blacklisting -- trying the "
                          "next configured model for this profile only (this model will be "
                          "retried first on the next profile).")
                    continue
                if not 200 <= int(code) < 300:
                    error = _gemini_error(int(code), response)
                    # The HTTP 429 status CODE is the reliable capacity signal -- Gemini's own
                    # machine-readable `error.status` ("RESOURCE_EXHAUSTED") is best-effort
                    # enrichment on top of it, not a precondition for treating the response as
                    # capacity. An infra-level rate limiter or proxy sitting in front of the API
                    # can (and, per an adversarial audit of this project, DOES) return a bare 429
                    # with an empty or differently-spelled status. Gating this branch on an exact
                    # status match used to let that response fall through every branch below and
                    # hit `raise error`, abandoning the whole cascade -- including every other
                    # healthy, configured model -- over what is still, unambiguously, a capacity
                    # response. So any 429 enters this branch; _classify_quota_exhaustion below
                    # still does the day/minute/unknown split from whatever quota detail (if any)
                    # the body carries, which decides HOW to react (blacklist vs. not), not
                    # whether to.
                    if error.http_code == 429:
                        scope = _classify_quota_exhaustion(error)
                        scopes[model] = scope
                        if scope == "day":
                            # RPD (requests-per-day) resets only at midnight Pacific, so this
                            # model genuinely cannot serve the rest of THIS run.
                            self._unavailable_models[model] = "day"
                            print(f"Gemini opener: {model} hit its per-day quota (resets at "
                                  "midnight Pacific); blacklisting it for the rest of this run "
                                  "and trying the next configured model.")
                        else:
                            # Per-minute caps are transient (free-tier RPM can be as low as 5)
                            # and clear within a minute, so do NOT blacklist -- just cascade to
                            # the next model for this profile; the preferred model is retried
                            # first on the next profile. "unknown" (no parseable quota details)
                            # gets the same non-blacklisting treatment: wrongly retiring the
                            # best model for a whole run on one ambiguous 429 is far more
                            # costly than one wasted retry per profile, and every model 429ing
                            # within a single call still raises GeminiCapacityExhausted below
                            # regardless of classification, so the "stop automation when
                            # everything is used up" guarantee holds either way.
                            kind = "per-minute" if scope == "minute" else "unclassified"
                            print(f"Gemini opener: {model} hit a {kind} 429; NOT blacklisting -- "
                                  "trying the next configured model for this profile only (this "
                                  "model will be retried first on the next profile).")
                        continue
                    if error.http_code == 404:
                        # NOT_FOUND: EMPIRICALLY MEASURED against the real API -- ListModels can
                        # list a model id with generateContent in its supportedGenerationMethods
                        # that then 404s the instant generateContent is actually called for this
                        # account (this happened for gemini-2.5-flash and gemini-2.5-flash-lite,
                        # both retired from opener.models entirely as a result; see
                        # preflight()'s docstring). A 404 is a property of THAT ONE model id, not
                        # of the request or the other configured models, so unlike an
                        # auth/permission/malformed-request error it must not be raised straight
                        # to the caller -- doing so would silently kill every other model in the
                        # cascade over one retired id. Drop just this model, permanently (a
                        # retired model does not come back mid-run), and try the next configured
                        # model, exactly like a per-day 429.
                        scopes[model] = "gone"
                        self._unavailable_models[model] = "gone"
                        print(f"Gemini opener: {model} returned 404 NOT_FOUND ({error.message}); "
                              "this model id is retired or unavailable to this account and will "
                              "not come back mid-run -- dropping it from the cascade for the "
                              "rest of this run and trying the next configured model.")
                        continue
                    if error.http_code >= 500:
                        # Provider-side failure (503 UNAVAILABLE "this model is currently
                        # experiencing high demand" is by far the common one; 500/502/504 behave
                        # the same). OBSERVED LIVE: gemini-3.6-flash returned 503 while every
                        # other configured model was serving normally, which proves this is a
                        # per-MODEL condition, not a provider-wide one. Raising it here would
                        # hand the caller a "transient error" for the whole service and the
                        # worker would send a bare like with no opener -- while six healthy
                        # models sat unused. So cascade to the next model immediately, and do
                        # NOT retire this one: high demand clears on its own, so it stays first
                        # in line for the next profile, exactly like a per-minute 429.
                        scopes[model] = "busy"
                        print(f"Gemini opener: {model} returned HTTP {error.http_code} "
                              f"{error.status or 'server error'}; NOT blacklisting -- trying the "
                              "next configured model for this profile only (this model will be "
                              "retried first on the next profile).")
                        continue
                    raise error
                if not isinstance(response, Mapping):
                    raise GeminiAPIError(int(code), None, "malformed success response")
                return self._parse(response, model)
            raise GeminiCapacityExhausted(_exhaustion_reason(scopes))

    def preflight(self) -> None:
        """Verify every configured model exists and supports generateContent before a run
        starts, so a typo'd model id or an invalid key fails fast at startup instead of
        mid-run after photos have already been captured for a live profile.

        GETs the ListModels endpoint through the injected transport (never real network in
        tests) and follows nextPageToken pagination to see the full catalog.

        NECESSARY BUT NOT SUFFICIENT: passing this check does not guarantee a model will
        actually serve a request. EMPIRICALLY MEASURED against the real API: ListModels
        listed both gemini-2.5-flash and gemini-2.5-flash-lite with "generateContent" in
        their supportedGenerationMethods, and generateContent still 404d NOT_FOUND for both
        on every call ("This model models/<id> is no longer available to new users").
        ListModels' catalog can lag behind which models an account can actually call. That
        is exactly why generate()'s runtime cascade must survive a 404 by retiring just that
        one model rather than treating it as fatal (see its 404 handling below) -- this
        preflight check narrows typos and dead keys, it is not a guarantee every listed
        model will work once a run is live.
        """
        seen: dict[str, list[str]] = {}
        url: str | None = _GEMINI_MODELS_LIST_URL
        while url:
            code, response = self.transport(
                url, None, {"X-goog-api-key": self.api_key}, self.request_timeout_s,
                method="GET",
            )
            if not 200 <= int(code) < 300:
                error = _gemini_error(int(code), response)
                # An invalid key surfaces here as HTTP 400 INVALID_ARGUMENT with message
                # "API key not valid...". Translate that into plain language instead of
                # leaking Gemini's raw wording, and never echo the key itself.
                if error.http_code == 400 and "api key not valid" in error.message.lower():
                    raise RuntimeError("GEMINI_API_KEY is not valid (rejected by Gemini's "
                                       "ListModels endpoint)")
                raise RuntimeError(
                    f"Gemini preflight failed: HTTP {error.http_code} {error.status}: {error.message}")
            if not isinstance(response, Mapping):
                raise RuntimeError("Gemini preflight failed: malformed ListModels response")
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
            url = f"{_GEMINI_MODELS_LIST_URL}?pageToken={quote(str(next_token), safe='')}" \
                if next_token else None

        missing = [m for m in self.models if m not in seen]
        unusable = [m for m in self.models if m in seen and "generateContent" not in seen[m]]
        if missing or unusable:
            problems = []
            if missing:
                problems.append(f"missing: {', '.join(missing)}")
            if unusable:
                problems.append(f"no generateContent support: {', '.join(unusable)}")
            available = ", ".join(sorted(seen)) or "(none returned)"
            raise RuntimeError(
                "Gemini preflight failed -- " + "; ".join(problems) +
                f". Available Gemini model ids: {available}")
