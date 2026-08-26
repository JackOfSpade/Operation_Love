"""Shared, budget-aware opener generation across all app workers.

One OpenerService is shared by every worker (Bumble, Hinge, ...) so the per-run
spend cap is GLOBAL, not per-app. Thread-safe. Handles the ways opener generation
ends for a run: the configured run budget, max_attempts consecutive bad AI responses
for one profile (see maybe_opener's retry loop below), the whole Gemini model cascade
falling through (GeminiCapacityExhausted -- every configured model out of per-day
quota, retired/404-gone, or some mix of those plus transient per-minute throttling;
see opener.py's GeminiOpener.generate()), and a permanent provider failure (bad/missing
API key, no access to the Gemini API at all, or -- only after several back-to-back
malformed-request errors -- a broken request schema/params) -- each flips the
service to disabled and asks the supervisor to stop all workers.

THE OWNER'S RULE (this is why there is no "swipe without an opener" mode anymore): a
commentless like is normally not acceptable. If the AI's response is bad, redo the prompt --
spending more usage is fine. The narrow exception is a provider safety-policy block: AUTO has
already decided to like that profile, so it may proceed without a comment rather than retrying
or stopping. Only if a profile's ordinary response is STILL bad after
max_attempts tries does that mean something is actually wrong, and only then does the
whole run stop. The old budget.on_exhausted="swipe_without_opener" configuration was
removed outright rather than left dormant -- the same reasoning that removed the
Anthropic provider path applies here: a setting that can silently reintroduce a banned
behavior is worse than no setting at all. Every AUTO exhaustion path below sets
stop_requested; advisory Observe failures disable further suggestions but deliberately leave
the human labelling session running.

Note what is NOT here: a single model id being retired (HTTP 404) is no longer a
service-wide, permanent failure. GeminiOpener.generate() retires just that one model
and cascades to the next configured model on its own -- see _permanent_reason's
docstring below for why this function deliberately does not classify a 404.

A LONE HTTP 400 is treated as transient, not permanent: an opener client embeds
that specific profile's own captured photos in every request, so a single 400
can be caused by a profile-specific payload
problem (a truncated/corrupt PNG from a flaky `adb screencap`, an oversized
image, too many photos) rather than by the request itself being malformed. Only
a run of consecutive 400s (see _BAD_REQUEST_LATCH_THRESHOLD) -- which a genuine
schema/param bug produces deterministically, since it re-fires on literally the
next call regardless of that call's content -- latches the service permanently
disabled.

A per-call failure is handled differently depending on whether re-asking can plausibly
help. Most OpenerParseError responses (a billed response that didn't parse into a usable
opener) are RETRIED, up to self.max_attempts times, with the model told what was wrong on each
retry -- see maybe_opener's docstring for the full loop and why: a parse failure is usually
stochastic (the model drew a bad sample), so a re-ask genuinely has a decent chance of
producing something usable. A prompt or response blocked by Gemini's safety policy is the
narrow exception: it is recorded and skipped for that profile without retrying or disabling the
run, because resending the same captured content cannot safely improve it. AUTO may still carry
out its already-made like decision without a comment. A non-parse
OpenerError (for example, a corrupt photo that can't even be encoded) is NOT retried -- it is
almost always specific to THAT
profile's content in a way a re-ask cannot fix (the same corrupt bytes go in again), so
THIS SERVICE just returns None and stays enabled for the next profile. Either way, every
billed attempt has its spend recorded before degrading. What a None return means to the
caller is the caller's decision, not this service's: OBSERVE mode swipes without an
opener (a human already made the call), but AUTO mode on an opener-capable app (Hinge)
does not -- see worker.py's _auto_loop, which refuses to send ANY like with no opener on
such an app, whether the None came from a narrow per-call path or from full exhaustion
below. last_skip_reason exists to give that AUTO guard an accurate, per-call reason for
exactly the narrow (non-exhausting) case (see its docstring).

A run of consecutive TRANSIENT failures (see _TRANSIENT_LATCH_THRESHOLD) is its own latch,
alongside the 400 streak above: an unclassified exception (a bare timeout, a connection
blip, a provider 5xx that GeminiAPIError didn't already classify) or a per-profile
non-parse OpenerError (see opener.py's item B corrupt-capture fix) is,
individually, exactly the kind of thing that must NOT kill a run over one unlucky profile.
But left completely unbounded, EVERY such failure used to fall through to the same silent
"swiping without an opener" branch forever, with the service left enabled and nothing on the
hub to show it: a systemic bug in our own request building, a dead network, or a
provider-wide outage would turn every remaining profile this run into a bare like with no
opener and no visible sign anything was wrong -- directly against this project's "fail
loudly, never silently degrade" rule. A short streak (not a lone failure, since one-off
blips are normal and must not halt the run) now latches the service exactly like the 400
streak does.
"""
from __future__ import annotations

import inspect
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from ..costing import CostTracker
from ..perception.capture import Profile
from .opener import (
    FIRST_ITEM_INDEX,
    INDEX_SPACE_MODEL_ITEMS,
    INDEX_SPACE_PROFILE_PHOTOS,
    ITEM_INDEX_ABSENT,
    GeminiAPIError,
    GeminiCapacityExhausted,
    ItemRequest,
    OpenerAborted,
    OpenerClient,
    OpenerDeadlineExceeded,
    OpenerError,
    OpenerParseError,
    REASON_PROMPT_BLOCKED,
    REASON_RESPONSE_BLOCKED,
    _leading_ngram,
)


@dataclass
class OpenerPick:
    """An opener plus which profile item it is about, so the driver can attach the comment to
    the RIGHT photo/prompt instead of always the first.

    `index` CHANGED MEANING on 2026-08-12 and was NOT renamed, so read this before using it
    (ops/OPENER-REDESIGN.md 5.1/5.3/5.7):

      before: 0-based index into the profile CAPTURE ORDER -- i.e. into the raw scroll frames
              the request was built from, which is not an item space at all (one card appears
              in several frames, one frame can hold two cards).
      now:    the MODEL ITEM INDEX, 1-based, straight off OpenerResult.item_index. It counts
              the numbered items the model was shown, and it is one answer to two questions:
              the item the opener is about AND the item to like.

    Both are ints, both are small, and neither carries its base in its type, which is exactly
    the class of bug doc 5.3 exists to prevent -- so no consumer may assume the old meaning
    still holds, and none may quietly add or subtract one to make an old consumer fit.

    `index_space` is what stops that from being a comment nobody reads: it says WHICH LIST
    `index` counts, is set from the request the opener client actually built, and
    `capture_order_index` below is the ONLY sanctioned way to turn this pick into something a
    driver can act on. It returns None whenever no sound conversion exists, which a driver must
    treat as "I was not told which item" rather than as any particular item.

    ITEM_INDEX_ABSENT (0) means the model gave no usable item number. Under the 1-based
    contract that is out of band by construction and can never be confused with a real pick --
    but note that it IS a legal value in the driver's own 0-based capture-order space, which is
    precisely why `capture_order_index` maps it to None instead of passing the zero along.

    `referenced` is the model's own one-line statement of WHICH profile detail the opener is
    about (OpenerResult.referenced, echoed here verbatim). It is not used for targeting --
    that's what `index` is for -- it exists so the hub can show the operator what the
    suggestion is supposed to be about, which is precisely how a mismatch between the opener
    and the item the comment actually hangs under becomes visible instead of silent -- the
    failure this whole anchor mechanism exists to prevent."""
    text: str
    index: int = ITEM_INDEX_ABSENT
    referenced: str = ""
    # The model's own free-text words for what this opener is DOING ("guessing where the ridge
    # is", "teasing her about the cold"), echoed verbatim from OpenerResult.angle. PURE
    # TELEMETRY: nothing in this service, the worker, or any driver branches on it, and it is
    # deliberately not an enum anywhere in the pipeline (ops/OPENER-REDESIGN.md 3.5) -- a closed
    # set would force the model to pick a move from a menu and shoehorn the opener into it,
    # which is exactly the awkwardness the move list is written to avoid. It exists so the
    # persisted `openers` rows can eventually answer which SHAPES correlate with matches, a
    # question this project currently cannot ask at all. Trailing, with a "" default, so every
    # existing OpenerPick(...) construction (worker.py, the tests' fakes) keeps working
    # untouched and a client that never populates the field degrades to "" rather than raising.
    angle: str = ""
    # The model's own short description of the ITEM it picked, echoed verbatim from
    # OpenerResult.item_description. Carried in BOTH modes, always (doc 5.7): auto logs it,
    # observe displays it, and it must never become conditional on `advisory` -- a
    # mode-dependent schema would make auto and observe issue different requests and quietly
    # destroy the canary property that is observe's entire reason to exist.
    #
    # Distinct from `referenced`: that is the DETAIL the opener reacts to, this is the ITEM it
    # was picked from. Doc 5.8 uses this one, coarsely (photo vs written prompt), to check our
    # own crop at `index` against what the model thought it chose BEFORE anything is tapped.
    # That check is a later workflow; nothing branches on this today.
    item_description: str = ""
    # WHICH LIST `index` COUNTS -- one of opener.INDEX_SPACE_*, echoed from
    # OpenerResult.index_space, which generate() sets from the request shape it actually built.
    #
    # Trailing with a default so every existing OpenerPick(...) construction keeps working, and
    # the default is the UNTRANSLATABLE space on purpose (same reasoning as
    # OpenerResult.index_space): a pick built by a fake or by some future call site that never
    # set this must not have its number converted into a tap on somebody's phone. Fail-safe
    # means "no target", not "target 1".
    index_space: str = INDEX_SPACE_MODEL_ITEMS
    # Private, one-run staging envelope.  AUTO and Observe both commit it only after the
    # driver's like boundary returns: a generated opener is not evidence of a landed action.
    _staged_record: "_StagedOpenerRecord | None" = field(
        default=None, repr=False, compare=False)

    @property
    def capture_order_index(self) -> int | None:
        """This pick as a 0-based index into the DRIVER's capture order, or None.

        This is the one sanctioned crossing between the model's index space and the driver's,
        and it exists because on 2026-08-12 the two stopped agreeing while nothing in the types
        said so. `Driver.like(item_index=...)` counts the frames the driver captured for this
        profile; `index` counts the numbered images the model was SENT. Handing one to the
        other unconverted is not a near miss -- an in-range 1-based item number resolves to a
        real, adjacent, WRONG frame and the driver reports it as on-target, so the comment
        lands one card down with full confidence and nothing detects it.

        None means "no sound conversion exists", and a driver must read it as "I was not told
        which item", never as an index. It is NOT the same as 0, which is a perfectly legal
        first frame in the driver's space -- collapsing the two is how ITEM_INDEX_ABSENT ("the
        model could not pick") turned into a confident like of item 1.

        The three cases:

        * ITEM_INDEX_ABSENT -> None. Doc 5.3: "treat a missing table as a hard stop, never as a
          reason to fall back to a fixed coordinate." And a hard stop is now literally what it
          is: worker._auto_loop refuses to call the driver at all with an untranslatable pick,
          and the driver itself refuses an opener carrying no item index
          (hinge._like_comment_sheet). Until 2026-08-12 this state instead tapped the first heart
          and repaired the message against whatever the sheet turned out to show; that repair
          hatch is removed, because rewriting the TEXT does not undo spending the LIKE on an item
          the model never chose (doc 5.6, never substitute).

        * INDEX_SPACE_PROFILE_PHOTOS -> index - FIRST_ITEM_INDEX. The numbered images WERE the
          profile's scroll frames, sent in capture order, so model item k is frame k-1. This is
          a derivation, not a fudge: the request was built from `profile.photos`, the driver
          captured `profile.photos` and keeps `_current_sigs` index-aligned with it (see
          hinge._capture_current, pinned by
          tests/test_hinge_observe.py::test_capture_keeps_sigs_index_aligned_with_photos), and
          the same Profile object is what the worker holds across both calls. The frames are
          poor "items" -- that is what doc 5.2 replaces with crops -- but the correspondence
          itself is exact.

        * INDEX_SPACE_MODEL_ITEMS -> None, and it STAYS None now that doc 5.5's counting
          navigation exists. Item k is the k-th SELECTABLE item, which the driver reaches by
          walking to its HEART ORDINAL -- not to a captured frame -- so there is no capture-order
          value that names it and inventing one is the bug this property was written to prevent.
          What changed on 2026-08-12 is what a CALLER does with the None: `worker._auto_loop` no
          longer reads it as "this pick cannot be acted on", it passes `pick.index` to
          `driver.like(model_item_index=...)` instead and lets the driver navigate. Also the
          default, so an unknown or unrecognised space refuses too -- and there a None really
          does mean "cannot be acted on", which is why the worker branches on `index_space`
          rather than on this being None.
        """
        if self.index == ITEM_INDEX_ABSENT:
            return None
        if self.index_space == INDEX_SPACE_PROFILE_PHOTOS:
            return self.index - FIRST_ITEM_INDEX
        return None


@dataclass(frozen=True)
class _StagedOpenerRecord:
    """An uncommitted, profile-attributable opener row for one pending like."""
    run_id: str
    app: str
    model: str
    opener: str
    referenced: str
    angle: str
    item_description: str
    recent_entry: dict


# How many consecutive provider HTTP 400s to require before treating the failure
# as permanent and latching the service disabled. A genuinely broken request (bad schema,
# bad param) is deterministic: it re-fires on the very next call no matter which profile's
# content is sent, so it still latches within a couple of profiles. A profile-specific
# payload problem (corrupt screenshot, oversized image) is very unlikely to repeat back to
# back across unrelated profiles' photos, so requiring a streak protects it from a single
# unlucky capture while still catching a systemic problem quickly.
_BAD_REQUEST_LATCH_THRESHOLD = 3

# How many consecutive TRANSIENT opener failures (an unclassified exception, or a per-profile
# OpenerError -- see _consecutive_transient_failures below for exactly what counts) to allow
# before latching the service disabled, mirroring _BAD_REQUEST_LATCH_THRESHOLD's tradeoff.
# Deliberately small, and deliberately not 1: a single timeout, one dropped connection, or
# one bad photo is ordinary and must not halt an otherwise-healthy run (that's the whole
# reason these failures degrade instead of raising in the first place). But an audit of this
# project demonstrated 50 consecutive bare RuntimeErrors sailing through with the service
# still enabled and nothing on the hub showing anything was wrong -- every one of those was a
# live profile that got a bare like with no opener. A short streak is enough to tell "one
# unlucky call" apart from "the pipeline itself is broken" (a systemic problem re-fires on
# essentially every subsequent call, exactly like the 400 case) while still absorbing an
# occasional real blip.
_TRANSIENT_LATCH_THRESHOLD = 3

# How many of the most recent SUCCESSFUL opener generations OpenerService.recent_openers keeps
# (see its docstring in __init__) for a bug report to inspect after the fact. Small and fixed:
# this is a live debugging aid, not the permanent record (that's self.store.record_opener), so
# it only needs to cover roughly the last screenful of activity, not the whole run.
_RECENT_OPENERS = 12

# Same idea as _RECENT_OPENERS, but for REJECTED attempts (OpenerParseError) -- see
# self.recent_rejections' docstring in __init__. Kept as its own ring buffer/constant rather
# than folded into recent_openers: a rejection is not a shorter-lived variant of a success,
# it is the other outcome entirely, and a bug report needs to show both without one crowding
# the other out of a shared, fixed-size buffer.
_RECENT_REJECTIONS = 12
_MAX_ATTEMPTS = 15

# How many LEADING words the entropy guard compares between openers -- see
# OpenerService._leading_ngram_collision / _apply_entropy_guard below and
# ops/OPENER-REDESIGN.md 3.6. Four is wide enough to catch a recurring OPENING FORMULA
# ("based on that ridgeline", "i bet you were") and narrow enough not to fire on two openers
# that merely share a word or two of ordinary English ("that view ...", "that husky ..."),
# which is not a fingerprint. Compared against self.recent_openers, whose own cap
# (_RECENT_OPENERS) therefore also decides how far back the guard can see.
_ENTROPY_NGRAM_WORDS = 4


def _safe_repr(value: object) -> str:
    """Represent rejected direct-call values without integer-string-limit failures."""
    try:
        return repr(value)
    except (ValueError, OverflowError):
        return f"<{type(value).__name__}>"


def _accepts_generate_deadline(client: object) -> bool:
    """Whether a duck-typed opener client accepts the advisory deadline keyword.

    GeminiOpener does, and that is the production path this deadline protects. Small legacy
    fakes and third-party client seams may predate the optional keyword; preserving their
    existing call shape keeps them usable. An opaque signature does not opt in: passing a new
    keyword to a legacy opaque callable is more likely to break its call than to enforce a
    deadline it never declared. The service's first-class OpenerClient protocol declares
    ``deadline``.
    """
    try:
        parameters = inspect.signature(client.generate).parameters.values()
    except (AttributeError, TypeError, ValueError):
        return False
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD
               or parameter.name == "deadline" for parameter in parameters)


def _record_staged_opener(store, record: "_StagedOpenerRecord", pick: OpenerPick, *,
                          profile_id: str, decision: str, decision_source: str,
                          decision_created_at: object | None,
                          pre_send_evidence: dict[str, object] | None = None) -> None:
    """Persist a committed draft with modern lineage when the store declares support for it.

    OpenerService has long accepted small duck-typed stores.  The action-lineage columns are a
    backward-compatible schema extension, not a reason for such a store to turn a real Like into
    a failed worker run.  Signature inspection (rather than a TypeError fallback) keeps a genuine
    store implementation error observable.
    """
    sink = store.record_opener
    try:
        parameters = inspect.signature(sink).parameters.values()
    except (TypeError, ValueError):
        accepts_lineage = True
    else:
        names = {parameter.name for parameter in parameters}
        accepts_lineage = (any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                               for parameter in parameters)
                           or {"profile_id", "decision", "decision_source",
                               "decision_created_at", "model_item_index"}.issubset(names))
    if accepts_lineage:
        sink(record.run_id, record.app, record.model, record.opener,
             record.referenced, record.angle, record.item_description,
             profile_id=profile_id, decision=decision, decision_source=decision_source,
             decision_created_at=decision_created_at,
             model_item_index=(pick.index if pick.index != ITEM_INDEX_ABSENT else None))
    else:
        sink(record.run_id, record.app, record.model, record.opener,
             record.referenced, record.angle, record.item_description)

    evidence_sink = getattr(store, "record_opener_send_evidence", None)
    if pre_send_evidence is not None and callable(evidence_sink):
        evidence_sink(
            record.run_id, record.app, record.opener, profile_id=profile_id,
            decision_source=decision_source, decision_created_at=decision_created_at,
            model_item_index=(pick.index if pick.index != ITEM_INDEX_ABSENT else None),
            evidence=pre_send_evidence,
        )


def _is_invalid_gemini_api_key(exc: Exception) -> bool:
    """True if exc is Gemini's actual invalid/revoked API key signal: HTTP 400
    INVALID_ARGUMENT with a message like "API key not valid. Please pass a valid API key."
    (some responses instead carry the machine-readable reason API_KEY_INVALID). Detected on
    the message/body, deliberately NOT on the bare 400 status: an ordinary malformed-payload
    400 (bad schema, a profile's own corrupt screenshot) must keep its existing 3-strikes
    streak behaviour (see _BAD_REQUEST_LATCH_THRESHOLD) rather than being misdiagnosed as a
    dead key, and a real dead key must not instead take 3 profiles to latch and get reported
    as a generic "malformed request" streak."""
    if not isinstance(exc, GeminiAPIError) or exc.http_code != 400:
        return False
    haystack = f"{exc.message} {exc.status}".lower()
    return "api key not valid" in haystack or "api_key_invalid" in haystack


def _permanent_reason(exc: Exception) -> str | None:
    """Classify a provider error as PERMANENT (will fail identically on every remaining
    profile this run: bad/missing key, no API access at all) vs. TRANSIENT (rate limit,
    5xx, connection blip -- worth retrying on the next profile). Returns an operator-facing
    explanation of what's wrong, or None if exc isn't a recognized permanent failure.
    Deliberately does NOT classify an ordinary HTTP 400 -- see _is_bad_request and
    OpenerService._consecutive_bad_requests, since a single 400 can be caused by that
    profile's own captured photos rather than by a systemic request problem. The one 400
    that IS classified here is an invalid/revoked Gemini API key (see
    _is_invalid_gemini_api_key), which Google reports as 400 INVALID_ARGUMENT rather than
    401 -- that specific failure is exactly as permanent as a 401 would be, so it must
    latch immediately rather than wait out the bad-request streak.

    Deliberately does NOT classify HTTP 404 either, even though a bad/typo'd model id looks
    just as permanent at a glance. It isn't the same kind of permanent: a 404 means ONE
    configured model id is retired or unavailable to this account, and says nothing about
    the rest of the configured cascade -- EMPIRICALLY MEASURED against the real API,
    gemini-2.5-flash and gemini-2.5-flash-lite both still appear in ListModels with
    generateContent support, yet 404 the instant generateContent is actually called (see
    GeminiOpener.preflight's docstring). Latching the WHOLE service disabled over one
    retired model id would have silently killed every other configured model too.
    GeminiOpener.generate() owns this instead: it retires just the offending model and
    cascades to the next configured one on its own (see its 404 handling), so a bare 404
    GeminiAPIError never reaches this function anymore. Only once the ENTIRE cascade has
    fallen through (whatever the mix of per-day/404/transient reasons) does generate() raise
    GeminiCapacityExhausted -- a different exception, handled by maybe_opener()'s own
    isinstance check before it ever gets here, not by this function.

    Gemini is the only opener provider (the Anthropic/Claude path was removed entirely), so
    this only ever classifies GeminiAPIError; anything else isn't a recognized permanent
    failure."""
    if isinstance(exc, GeminiAPIError):
        if _is_invalid_gemini_api_key(exc):
            return "GEMINI_API_KEY is not valid or has been revoked"
        if exc.http_code == 401:
            return "GEMINI_API_KEY is missing or invalid"
        if exc.http_code == 403:
            return "GEMINI_API_KEY does not have permission to use the Gemini API"
        return None
    return None


def _is_bad_request(exc: Exception) -> bool:
    """True if exc is Gemini's HTTP 400 error. Split out from _permanent_reason because a
    400 is only sometimes permanent -- see _BAD_REQUEST_LATCH_THRESHOLD. Callers must check
    _permanent_reason() FIRST: an invalid-key 400 (_is_invalid_gemini_api_key) is also a 400
    here, and must latch on the first occurrence, not join this streak."""
    return isinstance(exc, GeminiAPIError) and exc.http_code == 400


def _display_cost(cost: float | None) -> str:
    """Format known spend without pretending an unpriceable billed call cost $0."""
    return f"${cost:.4f}" if cost is not None else "an unknown amount"


class OpenerService:
    @property
    def last_skip_reason(self) -> str | None:
        """The current caller thread's most recent non-exhausting failure reason.

        One service is shared by concurrent app workers.  Keeping this per thread preserves the
        public attribute used by callers while preventing worker B from overwriting worker A's
        just-returned reason in the small window before A publishes its status.
        """
        return getattr(self._call_outcome, "last_skip_reason", None)

    @last_skip_reason.setter
    def last_skip_reason(self, value: str | None) -> None:
        self._call_outcome.last_skip_reason = value

    @property
    def last_skip_allows_commentless_like(self) -> bool:
        """Whether this thread's latest call permits AUTO to use its existing like decision.

        True only for Gemini content-policy blocks in a non-advisory call.  This structured,
        thread-local outcome keeps worker.py from parsing operator-facing message text and from
        confusing every other per-profile opener failure with the one exception the owner allows.
        """
        return bool(getattr(self._call_outcome, "allows_commentless_like", False))

    @last_skip_allows_commentless_like.setter
    def last_skip_allows_commentless_like(self, value: bool) -> None:
        self._call_outcome.allows_commentless_like = bool(value)

    def __init__(self, client: OpenerClient | None, tracker: CostTracker, store,
                 style: str, max_attempts: int = 5, advisory_max_attempts: int | None = None,
                 advisory_deadline_s: float = 60.0):
        # BUG 2 (adversarial audit): max_attempts=0 (or negative) made range(1, max_attempts+1)
        # empty, so maybe_opener()'s retry loop body never ran at all -- 0 API calls, disabled
        # stayed False, stop_requested stayed False, no reason was ever recorded. That is
        # exactly the silent-forever degradation this whole file exists to prevent: a service
        # that looks perfectly healthy on the hub while producing zero openers, forever.
        # config.validate() already keeps this out of the shipped app, but OpenerService is a
        # public class constructed directly all over the tests (and by any future direct call
        # site), so the invariant belongs here too, not only at the config layer. `bool` is
        # rejected explicitly even though it IS an int subclass in Python (True == 1) --
        # accepting it would silently treat OpenerService(..., max_attempts=True) as
        # max_attempts=1, a confusing coincidence rather than a real configuration.
        if (type(max_attempts) is not int
                or not 1 <= max_attempts <= _MAX_ATTEMPTS):
            raise ValueError(
                f"OpenerService max_attempts must be an int in [1, {_MAX_ATTEMPTS}], "
                f"got {_safe_repr(max_attempts)}")
        # Direct callers historically supplied only max_attempts. Keep that API usable for
        # small budgets while production passes the independently validated config value.
        if advisory_max_attempts is None:
            advisory_max_attempts = min(3, max_attempts)
        if type(advisory_max_attempts) is not int or advisory_max_attempts < 1:
            raise ValueError(
                "OpenerService advisory_max_attempts must be an int >= 1, got "
                f"{_safe_repr(advisory_max_attempts)}")
        if advisory_max_attempts > max_attempts:
            raise ValueError(
                "OpenerService advisory_max_attempts must be <= max_attempts, got "
                f"{_safe_repr(advisory_max_attempts)} > {_safe_repr(max_attempts)}")
        try:
            advisory_deadline = (
                float(advisory_deadline_s)
                if not isinstance(advisory_deadline_s, bool)
                and isinstance(advisory_deadline_s, (int, float))
                else None
            )
        except (TypeError, ValueError, OverflowError):
            advisory_deadline = None
        if (advisory_deadline is None or not math.isfinite(advisory_deadline)
                or not 0 < advisory_deadline <= 300):
            raise ValueError(
                "OpenerService advisory_deadline_s must be a number > 0 and <= 300, got "
                f"{_safe_repr(advisory_deadline_s)}")
        if not isinstance(style, str):
            raise ValueError(
                f"OpenerService style must be a string, got {_safe_repr(style)}")
        self.client = client
        self.tracker = tracker
        self.store = store
        self.style = style
        # How many times maybe_opener() will re-ask for ONE profile after an unusable
        # (but billed) response before giving up on that profile and stopping the whole
        # run -- see maybe_opener's docstring for the full retry loop. THE OWNER'S RULE:
        # a bad response is retried, never sent and never silently swapped for a bare
        # like; this many consecutive failures in a row is what turns "one-off bad luck"
        # into "something is actually wrong".
        self.max_attempts = max_attempts
        self.advisory_max_attempts = advisory_max_attempts
        self.advisory_deadline_s = advisory_deadline
        self.disabled = client is None
        self.stop_requested = False     # set by AUTO exhaustion; advisory exhaustion disables
                                         # suggestions without stopping human observation.
        # Human-readable cause of the FIRST exhaustion (see _exhaust) -- e.g. "run budget
        # reached" or "all configured Gemini models exhausted their free-tier quota; no
        # opener capacity remains". Every worker sharing this service reads the same value,
        # so whichever app happened to trigger it, every worker's hub status shows the one
        # true original cause rather than each other's follow-on symptoms (e.g. a second
        # worker's own maybe_opener() call failing merely because the service is already
        # disabled). None until the service has actually exhausted once.
        self.exhausted_reason: str | None = None
        # Per-thread outcome storage is required even though maybe_opener itself is serialized:
        # after it releases the service lock, another worker can complete and overwrite a shared
        # scalar before the first worker reads its reason. The property above keeps the existing
        # caller API but binds the value to the thread that made the call.
        self._call_outcome = threading.local()
        # Human-readable cause of THIS THREAD'S last maybe_opener() call that returned None WITHOUT
        # exhausting the service -- an unparseable response (OpenerParseError), a per-profile
        # OpenerError, a single sub-latch HTTP 400, or a single sub-latch transient failure
        # (see maybe_opener). Unlike exhausted_reason this deliberately keeps the LAST outcome
        # in each caller thread, not one first-writer-wins run-lifetime value: exhausted_reason answers "why
        # did the service ever stop serving openers at all", which every worker must agree on
        # forever after; this answers "why did THIS specific call just fail", which is allowed
        # (expected, even) to change from one profile to the next while the service otherwise
        # stays healthy. It exists so an AUTO-mode worker that refuses to send a like with no
        # opener (see worker.py's _auto_loop) can report the actual cause instead of a generic
        # "no opener" line. Cleared on every SUCCESSFUL call (see maybe_opener's success
        # branch) so a stale reason from three profiles ago never lingers and gets misread as
        # describing the current one. Deliberately left untouched by the disabled-at-entry
        # short-circuit and by every path that calls _exhaust(): those are global-exhaustion
        # causes, already carried by exhausted_reason, and must not be shadowed by whatever
        # per-call symptom happened to trigger that exhaustion.
        self.last_skip_reason = None
        self.last_skip_allows_commentless_like = False
        self._consecutive_bad_requests = 0   # streak of back-to-back 400s; see
                                              # _BAD_REQUEST_LATCH_THRESHOLD
        # Streak of back-to-back TRANSIENT failures (an unclassified exception, or a
        # per-profile OpenerError -- see _register_transient_failure for exactly what counts
        # and _TRANSIENT_LATCH_THRESHOLD for why); reset on any outcome that proves the
        # opener pipeline actually works.
        self._consecutive_transient_failures = 0
        # Ring buffer of the most recent COMMITTED opener records (AUTO generations, plus
        # Observe suggestions only after a confirmed Like; see recent_openers_snapshot),
        # mirroring self.store.record_opener's permanent per-run record. WHY THIS EXISTS: a
        # bug report that says only "provider_calls=1" cannot
        # tell you whether that opener was about the right photo. Recording the model's own
        # `referenced` string alongside whether the call was anchored (see maybe_opener's
        # anchor parameter) is exactly the evidence needed to diagnose an out-of-place opener
        # after the fact -- this is the anchor mechanism's own paper trail. Guarded by
        # self._lock like every other piece of mutable state on this instance; maybe_opener
        # already runs its whole body under that lock, so the append there needs no extra
        # locking -- only recent_openers_snapshot (a reader that may run on a different
        # thread, e.g. while building a bug report) takes the lock itself.
        self.recent_openers: deque[dict] = deque(maxlen=_RECENT_OPENERS)
        # Privacy-preserving entropy scratch space.  Advisory suggestions are drafts until a
        # real decision lands, so they must not enter the diagnostic/durable opener trail.
        # The entropy guard still needs to avoid a run of identical outbound openings, though;
        # retain only its normalized leading n-gram, in memory, for this process lifetime.
        # It has no profile id, item, model, full opener, or store write.
        self._recent_opening_ngrams: deque[str] = deque(maxlen=_RECENT_OPENERS)
        # Ring buffer of the most recent REJECTED opener attempts (OpenerParseError), mirroring
        # recent_openers above but for the other outcome. WHY THIS EXISTS: self.store.
        # record_opener_rejection is the durable per-run AUTO record, but a bug report (see
        # bugreport.py's Recent opener rejections section) needs this data WITHOUT a BigQuery
        # round-trip, exactly like recent_openers exists so the report doesn't have to query
        # self.store.record_opener's table either. Appended in the OpenerParseError branch of
        # maybe_opener below for every AUTO rejected attempt (including the final one that
        # exhausts retries), regardless of whether persistence succeeds. Observe drafts
        # deliberately keep no profile-attributable rejection trail.
        # "Attempt" is exact: an unusable ENTROPY REGENERATION draw (see _apply_entropy_guard)
        # is deliberately absent from both this buffer and the store's ledger, because that
        # draw never gated a send, and counting it would inflate the very guard-firing rate
        # these records exist to measure.
        # Guarded by self._lock exactly like recent_openers (maybe_opener already runs its
        # whole body under that lock).
        self.recent_rejections: deque[dict] = deque(maxlen=_RECENT_REJECTIONS)
        self._lock = threading.RLock()

    def _register_transient_failure(self, exc: Exception, *, request_stop: bool = True) -> bool:
        """Count one more consecutive transient-class failure and, once
        _TRANSIENT_LATCH_THRESHOLD is reached, latch the service disabled and return True
        (so the caller can skip its own "swiping without" print -- _exhaust() already prints
        a clear stop line, and printing both would just be noise). Called from both places
        that can produce this class of failure (see maybe_opener): the final catch-all branch
        (an unclassified exception -- timeout, connection blip, an uncaught 5xx) and the
        OpenerError branch (a per-profile content problem, e.g. a refusal or -- see
        opener.py's item B fix -- a corrupt captured photo).

        Records exc so the eventual _exhaust() reason names the LAST failure actually seen,
        not just a bare count -- an operator staring at "3 consecutive opener failures" with
        no detail has to go spelunking in old log lines for the one that matters.

        request_stop is forwarded to _exhaust() unchanged -- see maybe_opener's advisory
        parameter. An ADVISORY call (a Hinge observe-mode suggestion) still needs its own
        streak of unusable responses to eventually stop wasting quota on a broken model/
        prompt, but must never ask the whole run to halt over what is, for that call, purely
        cosmetic (a human is deciding for themselves either way).
        """
        self._consecutive_transient_failures += 1
        if self._consecutive_transient_failures >= _TRANSIENT_LATCH_THRESHOLD:
            last = f"{type(exc).__name__}: {exc}"
            self._exhaust(
                f"{self._consecutive_transient_failures} consecutive opener calls failed in a "
                f"row without ever producing a usable opener (most recent: {last}) -- this "
                "looks like a systemic problem (a dead network, a provider outage, our own "
                "request building, or a corrupted capture pipeline), not one-off bad luck, so "
                "the run is stopping rather than silently sending bare likes with no opener "
                "for the rest of it",
                request_stop=request_stop,
            )
            return True
        return False

    def _leading_ngram_collision(self, opener: str) -> str:
        """The leading n-gram this opener SHARES with one already produced this run, or "" when
        it opens with words we have not used yet. Pure lookup over a bounded, in-memory n-gram
        scratch buffer; it decides nothing on its own, _apply_entropy_guard below owns every
        consequence.

        Deliberately NOT filtered by app, and it includes advisory n-grams: the fingerprint this
        guard exists to avoid is "every message this account sends opens the same way", and
        neither a woman comparing screenshots with a friend nor an anti-bot heuristic cares
        which worker produced a line.  Advisory text remains a non-durable draft until a Like,
        but its n-gram is sufficient for the run-local anti-repetition check.

        The n-gram is recomputed from each entry's stored text rather than cached in the entry:
        the buffer is bounded at _RECENT_OPENERS, so this is a dozen string compares, and
        keeping that dict schema stable matters more than the microseconds -- bugreport.py
        renders those same entries.
        """
        ngram = _leading_ngram(opener, _ENTROPY_NGRAM_WORDS)
        if not ngram:
            return ""
        if ngram in self._recent_opening_ngrams:
            return ngram
        return ""

    def _record_billed_draw(self, run_id: str, model: str, usage, *, note: str) -> None:
        """Track + persist the spend of a billed opener call whose text is NOT the one being
        sent.

        Spend is billing telemetry, rather than a preference/card decision.  It remains durable
        for every real provider call, including a discarded Observe draft, so run/day budgets
        never under-report actual usage.

        Deliberately does NOT call _exhaust on an unpriceable model, unlike every other
        tracker.record() call site in this file -- the entropy guard's contract is that it can
        never stop a run (see _apply_entropy_guard). Nothing is lost by that: the draw we KEEP
        is recorded a few lines later in maybe_opener's success branch, essentially always from
        the same model id, and that call site does exhaust on the identical KeyError. The only
        gap is the freak case where the cascade changed models between the two draws and only
        the discarded one is unpriceable, which is why that path still prints a loud warning
        rather than passing silently.
        """
        try:
            cost = self.tracker.record(model, usage)
        except KeyError:
            print(f"Warning: no budget.pricing entry for model '{model}'; the {note} was "
                  "billed but its spend cannot be tracked")
            cost = None
        except Exception as e:  # noqa: BLE001
            # Broader than the main path's KeyError-only guard, deliberately: this helper is
            # reached ONLY from the entropy guard, which is holding a perfectly good opener at
            # the time. Letting an accounting oddity (a result with no usage, a tracker that
            # raises something new) escape from here would turn a cosmetic style check into a
            # failed generation, which is precisely the inversion the guard must never cause.
            print(f"Warning: could not track the spend of the {note} "
                  f"({type(e).__name__}: {e})")
            cost = None
        try:
            self.store.record_spend(run_id, model, usage, cost)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: failed to persist the spend record for the {note} "
                  f"({_display_cost(cost)}): {e}")

    def _apply_entropy_guard(self, run_id: str, profile: Profile, result, *,
                             items: ItemRequest | None,
                             should_stop: Callable[[], bool] | None,
                             skip_models: frozenset[str],
                             deadline: float | None = None,
                             client_accepts_deadline: bool = False) -> tuple[object, str, bool]:
        """THE ENTROPY GUARD (ops/OPENER-REDESIGN.md 3.6). Given a parsed, usable opener, return
        `(result_to_send, colliding_ngram, regenerated)`: either the result handed in, or a
        second draw taken because the first one opened with words already sent this run.

        WHY IT EXISTS: shortening the openers compresses the output space, and the few-shot edit
        pairs in the style block make direct copying a live risk. Across a burner account
        sending uncapped volume, a run of messages that all open "Based on that X, I'm going to
        guess ..." is simultaneously a bot fingerprint and genuinely embarrassing if two matches
        ever compare screenshots. The check is a plain string comparison of leading n-grams (see
        _leading_ngram_collision) -- no semantics, no classifier, no second judge, and no
        constraint whatsoever on WHAT the model is allowed to write. It only ever asks for
        another draw.

        THREE THINGS IT MUST NEVER DO, and they are the whole design:
          1. Never raise. Every failure of the extra call is swallowed and the original opener
             is kept, because the original was already good enough to send.
          2. Never reject, never _exhaust(), never touch stop_requested or either latch counter.
             A stylistic near-miss is not evidence that the provider, the prompt, or the request
             pipeline is unhealthy, and stopping a run over one would be wildly disproportionate.
          3. Never consume the per-profile attempt budget. Note precisely what that means
             structurally: the call site is lexically INSIDE maybe_opener's
             `for attempt in range(...)` loop, on the success path, but that path returns
             unconditionally, so the extra draw can never produce another `attempt` iteration
             and can never push a profile toward the max_attempts exhaustion that stops the
             whole run. The budget exemption comes from the guard living on a returning path
             and never re-entering the loop, not from its line number. Anyone moving this call
             must preserve that property, not just its position. Doc 3.6 calls the requirement
             out explicitly: a hard rejection here "can stop a run over a stylistic near-miss,
             which is disproportionate".

        THE ADVISORY ASYMMETRY -- CORRECTED 2026-08-11. Read this before "restoring" the old
        behavior; the paragraph that used to live here argued the guard must be skipped OUTRIGHT
        when advisory is True, and that argument was WRONG. Do not resurrect it.

        The old reasoning: an advisory (Hinge observe-mode) call used a smaller attempt budget
        because a human may be waiting on the suggestion
        (see maybe_opener's advisory docstring paragraph), so a guard that consumed an attempt
        could disable suggestions over a stylistic near-miss. That premise is false for this guard as
        actually built: see THE THREE THINGS above, especially point 3. The extra draw sits on
        the SUCCESS path, lexically inside maybe_opener's `for attempt in range(...)` loop but
        returning unconditionally, so it structurally CANNOT produce another `attempt` iteration
        -- in AUTO or in advisory alike. There was never an attempt for advisory to lose. The old
        paragraph was reasoning about a cost this function's own construction already ruled out;
        it was carried over from ops/OPENER-REDESIGN.md 3.6's open question about a
        HARD-REJECTING guard design (one that never shipped) and never re-checked against the
        soft, budget-exempt design that actually did.

        Running the guard under advisory is not merely SAFE, it is REQUIRED, and skipping it was
        an active bug, not a conservative default. Hinge observe mode is the canary for auto --
        the owner's rule is that the opener shown to a human in observe must be byte-identical to
        what auto would type for the same profile, because observe exists to preview what auto is
        about to do at scale. A guard that runs in AUTO but is inert in observe makes the two
        modes diverge on exactly the profiles the guard exists to change: a collision that AUTO
        would quietly redraw around used to ship untouched to the human in observe, so observe
        stopped predicting what auto actually sends -- defeating the canary. This is not
        hypothetical: a live dry run with advisory=True and a shared OpenerService (so
        recent_openers accumulates across calls, exactly like a real observe session) produced 5
        openers where 4 opened with the identical phrase, because the guard never even looked.

        So: advisory now runs through the exact same collision check, extra draw, and
        accept-on-second-collision path as AUTO, with no branch on `advisory` anywhere in this
        function any more. What makes that safe is not "advisory is a lesser case needing its own
        carve-out" -- it is this guard's OWN invariants (points 1-3 above), which hold
        identically in both modes: it never raises, never rejects, never calls _exhaust(), and
        never touches stop_requested or either latch counter, regardless of advisory. There was
        never a failure mode here for advisory's bounded attempt budget to be exposed to.

        ONE extra draw, never a loop: if the second draw collides too, it is ACCEPTED and
        logged. A near-identical opener that gets sent is a much smaller problem than a retry
        storm, a stalled profile, or the sunk cost of a third billed call, and an unbounded
        "keep asking until it is different" loop is a spend hole with no ceiling. This part is
        unchanged by the correction above and applies identically in advisory and AUTO.
        """
        text = str(getattr(result, "opener", "") or "")
        collision = self._leading_ngram_collision(text)
        if not collision:
            return result, "", False

        # Below here we KNOW we would like another draw. Two reasons not to actually take one,
        # both about spending money we should not spend; in each case the original opener is
        # perfectly sendable, so this degrades to "ship the repetitive one" rather than to any
        # kind of failure.
        if should_stop is not None and should_stop():
            print(f"Opener: this draft repeats an earlier opening this run "
                  f"(\"{collision}\"), but the run is stopping -- sending it as is rather "
                  "than spending on another draw.")
            return result, collision, False
        if self.tracker.budget_reached():
            print(f"Opener: this draft repeats an earlier opening this run "
                  f"(\"{collision}\"), but the run budget is reached -- sending it as is "
                  "rather than spending on another draw.")
            return result, collision, False
        if deadline is not None and time.monotonic() >= deadline:
            print(f"Opener: this draft repeats an earlier opening this run "
                  f"(\"{collision}\"), but the advisory deadline has passed -- sending it "
                  "as is rather than starting a stale regeneration.")
            return result, collision, False

        print(f"Opener: this draft opens with words already sent this run (\"{collision}\"); "
              "asking once for a different opening. This extra call is deliberately EXEMPT "
              "from the per-profile attempt budget -- see _apply_entropy_guard.")
        # Written to be read by the MODEL, so: no em dash, no hyphen, plain ASCII, and no
        # suggestion that the previous draft was bad. It was not; it was fine and merely
        # familiar. Telling the model it was "rejected" (the wording the real retry path uses,
        # where the text genuinely was unusable) would push it to change the wrong things.
        retry_hint = (
            f"Your previous draft was fine, but it opened with the same words as an opener "
            f"already sent to someone else recently: \"{collision}\". Write a different "
            f"opener for this profile that does not open with those words. Every other rule "
            f"is unchanged."
        )
        try:
            # `items` rides along because the second draw is the
            # SAME request with a different retry_hint. Dropping it here would silently switch
            # the regeneration to the raw-frame shape -- the model would be answering about a
            # different set of images, in a different index space, and whichever draw survived
            # would carry the other one's numbering. The entropy guard is about the opening
            # WORDS and must change nothing else about the request.
            generate_kwargs = dict(items=items, should_stop=should_stop,
                                   skip_models=skip_models)
            if client_accepts_deadline:
                generate_kwargs["deadline"] = deadline
            second = self.client.generate(profile, self.style, retry_hint=retry_hint,
                                          **generate_kwargs)
        except OpenerDeadlineExceeded as e:
            # The first draw is valid and this second one is a cosmetic refinement. An expiry
            # is therefore a normal reason to retain that first draft, never a provider failure.
            # A late 2xx can nevertheless have consumed provider quota, so record its usage
            # exactly once before retaining the first draft.
            if e.usage is not None and e.model:
                self._record_billed_draw(run_id, e.model, e.usage,
                                         note="stale entropy regeneration response")
            print("Opener: the entropy regeneration reached the advisory deadline; keeping the "
                  "original opener, which was already good enough to send.")
            return result, collision, False
        except OpenerParseError as e:
            # Billed but unusable. Record the spend (real money, see _record_billed_draw) and
            # keep the original opener. Deliberately NOT written to the opener_rejections
            # ledger: that table answers "how often do the deterministic SEND guards reject an
            # attempt", and this draw was never gating a send -- counting it there would inflate
            # exactly the statistic it exists to measure. The console line below plus the
            # `entropy_collision` field on the ring-buffer entry are this path's paper trail.
            self._record_billed_draw(run_id, e.model, e.usage,
                                     note="rejected entropy regeneration draw")
            print(f"Opener: the entropy regeneration came back unusable ({e}); keeping the "
                  "original opener, which was already good enough to send.")
            return result, collision, False
        except Exception as e:  # noqa: BLE001
            # Anything else at all: a stop signal mid-cascade (OpenerAborted), a per-profile
            # OpenerError, an HTTP failure, capacity exhaustion, a fake client that cannot cope
            # with a second call. NONE of them move a latch counter, set last_skip_reason, or
            # exhaust: we are holding a perfectly good opener, so there is no failure to report
            # to anyone -- only a cosmetic improvement we did not get.
            print(f"Opener: the entropy regeneration failed ({type(e).__name__}: {e}); "
                  "keeping the original opener, which was already good enough to send.")
            return result, collision, False

        second_text = str(getattr(second, "opener", "") or "")
        if not second_text:
            # This second call was real and billed, exactly like the first -- record it before
            # anything else. Defensive in practice: a real GeminiOpener._parse() raises
            # REASON_EMPTY_AFTER_SANITIZE on an empty opener, so only a fake or future client
            # can land `second` here with blank text. But _record_billed_draw's own invariant
            # ("every real call being recorded exactly once") is unconditional, not contingent
            # on which client made the call, so it is recorded anyway.
            self._record_billed_draw(run_id, getattr(second, "model", "") or "",
                                     getattr(second, "usage", None),
                                     note="empty entropy regeneration draw")
            # A client that returns a result with no text at all would turn a cosmetic guard
            # into a commentless like, the one outcome this whole file exists to prevent.
            print("Opener: the entropy regeneration returned no opener text; keeping the "
                  "original opener.")
            return result, collision, False

        # The first draft is being thrown away, but it was BILLED. Record it here so the money
        # is accounted for exactly once; maybe_opener records the surviving draw itself.
        self._record_billed_draw(run_id, getattr(result, "model", "") or "",
                                 getattr(result, "usage", None),
                                 note="entropy regenerated opener draft")
        again = self._leading_ngram_collision(second_text)
        if again:
            # ONE extra draw only -- accept and be loud about it. (A second draw that repeats
            # the FIRST draft's opening is covered by this same check: the first draft collided
            # with the buffer, so anything sharing its n-gram collides with the buffer too.)
            print(f"Opener: the regenerated opener still opens with \"{again}\"; SENDING it "
                  "anyway. The guard asks once and never loops -- a near-repeat that goes out "
                  "is a smaller problem than a retry storm or an unbounded spend.")
        return second, collision, True

    def maybe_opener(self, run_id: str, app: str, profile: Profile, *,
                      items: ItemRequest | None = None,
                      should_stop: Callable[[], bool] | None = None,
                      advisory: bool = False, stage: bool = False) -> "OpenerPick | None":
        """Return an OpenerPick (text + the MODEL ITEM INDEX the opener is about and the item
        to like -- 1-based, see OpenerPick), or None (disabled / budget
        reached / out of credit / permanent provider error / every retry attempt used up /
        a per-profile OpenerError / a single sub-latch 400 or transient failure / the run
        stopping via should_stop).

        advisory (default False -- unchanged AUTO-mode behavior): True for Hinge's pre-action
        Observe suggestion (see worker.py's _ObserveSuggestion). It is a private draft until a
        confirmed Like: manual Observe may display it for a person, while the reviewed bridge
        may send that exact current draft only after its own sheet checks. Two consequences
        follow directly from its advisory role:
          1. It uses the shorter advisory_max_attempts budget and advisory_deadline_s. The
             service still enters its first attempt for legacy-client compatibility, but a
             deadline-aware Gemini client receives the same absolute cutoff before any request.
             If waiting for the shared lock or preparing images has already consumed it, Gemini
             issues no stale first request. Every later fallback model and the optional
             entropy-regeneration draw share that cutoff; each request is capped to the
             remaining time, and a response arriving after it is discarded as a stale advisory
             result rather than counted as a provider failure.
          2. Every exhaustion path below routes through _exhaust(..., request_stop=False)
             instead of the default request_stop=True. disabled/exhausted_reason are still
             set exactly as for an AUTO exhaustion (so a systematically broken model/prompt
             still stops burning quota on further suggestions), but stop_requested is left
             alone: an advisory suggestion failing is COSMETIC (the person or reviewed
             controller can still make a preference action without this draft), and ending
             the whole Observe session over it would sacrifice the session's entire purpose
             (collecting real training labels) for something that was never going to force an
             action. See worker.py's _ObserveSuggestion for the call site and this class's
             own _exhaust() for the matching request_stop plumbing.

        items (default None): ops/OPENER-REDESIGN.md 5.2/5.7's item-crop request shape -- one
        cropped image per numbered profile item, the unnumbered context crops after them, her
        name as text and the capture's truncation flag, built by
        opener.ItemRequest.from_profile() from what the driver enumerated. When present it
        REPLACES profile.photos as the model's view of her, which is the whole point: image k
        IS item k, so the number the model answers with means something. Forwarded to
        self.client.generate(...) unconditionally below: a client that silently dropped this
        kwarg would fall back
        to sending raw scroll frames, where one card appears in several frames and one frame can
        hold two cards, and the returned item number would be confidently meaningless. A
        TypeError out of the call is the correct failure there.

        Both auto and observe use `items`; the same crops are reused verbatim on every attempt
        for this profile, so only `retry_hint` changes after a rejected response.

        should_stop (BUG 1, adversarial audit): a cheap, non-blocking "is the run stopping?"
        check -- in practice worker.py's threading.Event.is_set for the shared stop flag.
        Threaded into every self.client.generate(...) call below (so GeminiOpener can abort
        BETWEEN models mid-cascade -- see its own should_stop docstring) and also checked
        directly by this method BETWEEN RETRY ATTEMPTS, since a fake/simple OpenerClient
        cannot be relied on to implement its own should_stop cascade the way GeminiOpener
        does. Either check raising/observing a stop returns None IMMEDIATELY: no further
        retries, no _exhaust() call, and neither latch counter moves -- a deliberate
        shutdown is not evidence the provider or this service is unhealthy, and treating it
        as one would poison exhausted_reason with a misleading "provider failed" story for
        an operator who just clicked Stop. Pre-fix, nothing on this path ever consulted a
        stop signal at all: this method holds self._lock across every attempt (up to
        max_attempts), each of which could cascade across every configured model at
        request_timeout_s apiece, so a Stop click could be silently ignored for as long as
        max_attempts * len(models) * request_timeout_s (5 * 7 * 90s = 3150s, ~52 minutes,
        against the shipped config) while this service's lock stayed held, blocking every
        other worker sharing it.

        THE OWNER'S RULE: a commentless like is normally not acceptable. If the AI's response
        is bad, redo the prompt -- spending more usage is fine. The narrow exception is a
        provider safety-policy block in AUTO: the ranker has already decided to like that
        profile, so the worker may carry out that decision without a comment. Most billed responses that
        didn't parse into a usable opener (OpenerParseError) are RETRIED for THIS profile,
        up to self.max_attempts times, each retry telling the model what was wrong with its
        previous attempt (retry_hint) so a re-ask can actually do better -- this is worth
        doing because a parse failure is usually stochastic (the same request often
        succeeds on a re-ask), unlike a deterministic failure (see below). Gemini safety
        blocks are deliberately not resent: the current profile is skipped with a visible
        reason while the next profile remains eligible. Only once
        max_attempts consecutive attempts have ALL failed does that stop looking like
        one-off bad luck and start looking like something actually broken (a bad prompt, a
        degenerate model, a schema bug) -- at that point the whole run stops via
        _exhaust(), exactly like the owner's instruction: "if after 5 attempts it's still a
        bad response, stop the automation -- that means something is wrong."

        ONE further billed call can happen per profile, and it is NOT an attempt: after a
        successful parse, the entropy guard (ops/OPENER-REDESIGN.md 3.6, implemented in
        _apply_entropy_guard) may ask once more when the opener opens with the same words as
        one already produced this run. It sits deliberately OUTSIDE the attempt loop, so a
        stylistic near-miss can never push a profile toward the max_attempts exhaustion that
        stops the run; it never raises, never rejects, and never latches anything; and -- as of
        the 2026-08-11 correction -- it now runs IDENTICALLY on an advisory call, not skipped:
        its exemption from the attempt budget was never conditional on advisory in the first
        place (it lives on the success path, outside the loop, in either mode), so there was
        never a cost for skipping it to avoid, and running it in observe is REQUIRED, not merely
        tolerated -- observe is the canary for auto (the opener shown to a human there must be
        byte-identical to what auto would send), and a guard that only fired in AUTO made observe
        stop predicting auto on exactly the profiles it exists to change. Both draws' spend is
        recorded either way. See _apply_entropy_guard for the full reasoning, especially the
        ADVISORY ASYMMETRY paragraph, which documents the old (wrong) design explicitly so it
        does not get "simplified" back in.

        Every OTHER kind of per-call failure is deliberately NOT retried:
          - OpenerError (not a parse error) is almost always deterministic -- e.g. a
            corrupt captured photo that fails to decode while fitting the request to the
            inline-size budget (see opener.py's _fit_images_to_budget). Re-asking would
            resend the exact same bad payload and fail identically, so retrying it here
            would be pure wasted spend; this SERVICE just returns None and stays enabled
            for the next profile.
          - An HTTP-level failure (429/404/5xx/a transport error) is not retried HERE at
            all: GeminiOpener.generate() already cascades those across every configured
            model internally before it ever raises to this method (see its class
            docstring) -- by the time an exception reaches here, the whole model cascade
            has already run its course for this call, so retrying THAT would just repeat
            work generate() already did.

        The budget check, every provider call for this profile (including retries), and
        spend recording are all serialized under one lock acquisition, so concurrent app
        workers cannot interleave with a retry sequence and overspend the shared run
        budget. The budget is also re-checked after EVERY attempt's spend is recorded
        (not just once, before the first attempt): the owner's retry rule never overrides
        the run budget, so a budget crossed mid-retry stops the retries immediately with
        "run budget reached" rather than continuing to spend on attempt after attempt.

        Every path that returns None because of a global exhaustion (budget reached, a
        latch tripped, a permanent provider error, GeminiCapacityExhausted, every retry
        attempt failing) goes through _exhaust(), which records exhausted_reason and, for
        AUTO, sets stop_requested. Every OTHER path that returns None (a per-profile OpenerError, or
        a single sub-latch 400/transient failure) instead records last_skip_reason: this
        call failed, but the service is still enabled and expects to succeed again next
        profile. Callers that must never send a like with no opener (see worker.py's
        _auto_loop) need to distinguish the two, since only the second kind leaves
        self.disabled False. A should_stop-triggered None joins the SECOND group (per-call,
        service stays enabled): the run stopping says nothing about whether the opener
        pipeline itself is healthy, so it must not be reported or counted like one.
        """
        advisory_started_at = time.monotonic() if advisory else None
        advisory_deadline_at = (
            advisory_started_at + self.advisory_deadline_s
            if advisory_started_at is not None else None
        )
        client_accepts_deadline = advisory and _accepts_generate_deadline(self.client)
        with self._lock:
            # Per-call permission, never sticky. A successful call or any unrelated failure
            # after a safety-blocked profile must restore the ordinary no-bare-like rule.
            self.last_skip_allows_commentless_like = False
            if self.disabled:
                return None
            if self.tracker.budget_reached():
                self._exhaust("run budget reached", request_stop=not advisory)
                return None

            # Local policy only: the service is shared by advisory and AUTO callers, so an
            # observe suggestion must never mutate the full autonomous retry budget.
            effective_max_attempts = (
                self.advisory_max_attempts if advisory else self.max_attempts)
            # Which models have already produced an unusable response FOR THIS PROFILE, so a
            # retry can steer GeminiOpener's cascade away from re-hitting the same (often
            # scarcest-quota) model that just failed -- see opener.py's generate() skip_models
            # docstring. Local to this call/profile by construction (a fresh call stack frame
            # per maybe_opener() invocation): it can never leak into the NEXT profile's call,
            # which gets its own empty set here.
            failed_models: set[str] = set()

            retry_hint = ""   # "" means first attempt; a retry fills this in below
            for attempt in range(1, effective_max_attempts + 1):
                if (advisory and attempt > 1 and advisory_deadline_at is not None and
                        time.monotonic() >= advisory_deadline_at):
                    self.last_skip_reason = (
                        "opener advisory retry deadline reached after "
                        f"{attempt - 1}/{effective_max_attempts} attempt(s); no further "
                        "attempt was started for this profile"
                    )
                    print(
                        "Opener: advisory retry deadline reached after "
                        f"{attempt - 1}/{effective_max_attempts} attempt(s); showing no "
                        "suggestion for this profile, while future profiles remain eligible."
                    )
                    return None
                if should_stop is not None and should_stop():
                    # Checked BETWEEN retry attempts (including before the very first one),
                    # independent of whatever the client itself does internally -- a simple/
                    # fake OpenerClient cannot be relied on to implement GeminiOpener's own
                    # per-model should_stop cascade, so this method's own retry loop must not
                    # depend on that. Returns immediately: no _exhaust(), no latch movement,
                    # no retry -- this is a deliberate shutdown, not a failure of any kind
                    # (see this method's should_stop docstring paragraph).
                    self.last_skip_reason = (
                        "opener call skipped before this attempt: the run is stopping "
                        "(should_stop signaled), not a provider failure"
                    )
                    return None
                try:
                    # items is forwarded unconditionally, and the
                    # cost of getting it wrong is higher: a client that quietly ignored it would
                    # send profile.photos instead, and the model's item_index would then count
                    # scroll frames while every consumer downstream reads it as an item number
                    # (doc 5.2/5.7). Loud TypeError over silent renumbering.
                    generate_kwargs = dict(
                        items=items,
                        should_stop=should_stop,
                        skip_models=frozenset(failed_models),
                    )
                    if client_accepts_deadline:
                        # GeminiOpener receives the absolute deadline, not another relative
                        # budget. It checks it between every fallback model and limits an
                        # already-starting request to the remaining time.
                        generate_kwargs["deadline"] = advisory_deadline_at
                    result = self.client.generate(profile, self.style, retry_hint=retry_hint,
                                                  **generate_kwargs)
                except OpenerDeadlineExceeded as e:
                    # A stale advisory suggestion is deliberately cosmetic. This is neither a
                    # provider timeout nor a profile/request failure: do not retry, latch,
                    # disable the service, or ask the observation run to stop. A late 2xx can
                    # still be billed, however, so retain its accounting facts exactly once.
                    if not advisory:
                        raise
                    if e.usage is not None and e.model:
                        self._record_billed_draw(run_id, e.model, e.usage,
                                                 note="stale advisory opener response")
                    self.last_skip_reason = str(e)
                    print("Opener: advisory deadline reached during the model cascade; showing "
                          "no suggestion for this profile, while future profiles remain eligible.")
                    return None
                except OpenerAborted as e:
                    # The client itself aborted mid-call (e.g. GeminiOpener's cascade caught a
                    # stop signal BETWEEN models, after already issuing at least one request
                    # this attempt). Same contract as the loop-top check above: return
                    # immediately, no retry, no _exhaust(), no latch movement -- see BUG 1's
                    # fix rationale and OpenerAborted's own docstring for why this must never
                    # be treated like an ordinary OpenerError.
                    self.last_skip_reason = f"opener call aborted mid request: {e}"
                    return None
                except OpenerParseError as e:
                    # The call reached the API and was billed (it returned usage) even though
                    # the body didn't parse into a usable opener. Record the spend like a
                    # normal call -- EVERY attempt is a real billed call, retries included, so
                    # this happens on each pass through this branch, not just the first.
                    # Reaching the API at all (billed usage came back) proves the request shape
                    # itself is fine, so this breaks any streak of bad-request failures too --
                    # and, for the same reason, any streak of transient failures: a real
                    # end-to-end round trip through transport/auth/request-building just
                    # succeeded, which is exactly the "pipeline works" signal that latch resets
                    # on (see _TRANSIENT_LATCH_THRESHOLD).
                    self._consecutive_bad_requests = 0
                    self._consecutive_transient_failures = 0
                    # D: this model just produced an unusable response FOR THIS PROFILE -- steer
                    # any further retry attempt's cascade away from re-hitting it first (see
                    # opener.py's generate() skip_models docstring). Reset to an empty set at the
                    # top of every maybe_opener() call, so this never survives past the current
                    # profile.
                    failed_models.add(e.model)
                    # Record WHY this attempt produced no opener before any of the exhaustion
                    # branches below have a chance to overwrite it with a global cause -- this is
                    # a per-call reason, not a global one (see last_skip_reason's docstring), and
                    # must describe what actually happened here (an unparseable body) regardless
                    # of whether this also turns out to be the attempt that exhausts the run.
                    self.last_skip_reason = (
                        f"opener response could not be parsed into a usable opener "
                        f"(attempt {attempt}/{effective_max_attempts}): {e}"
                    )
                    try:
                        cost = self.tracker.record(e.model, e.usage)
                    except KeyError:
                        # Same guard as the success path: the API echoed a model with no
                        # budget.pricing entry, so this (real, billed) spend can't be tracked.
                        self._exhaust(f"no budget.pricing entry for model '{e.model}'; "
                                      "spend can no longer be tracked", request_stop=not advisory)
                        cost = None
                    try:
                        self.store.record_spend(run_id, e.model, e.usage, cost)
                    except Exception as store_exc:  # noqa: BLE001
                        print(f"Warning: failed to persist opener spend record "
                              f"({_display_cost(cost)}): {store_exc}")
                    attempt_outcome = (
                        "safety-blocked" if e.reason_code in
                        (REASON_PROMPT_BLOCKED, REASON_RESPONSE_BLOCKED) else "unparseable"
                    )
                    result_noun = "provider result" if attempt_outcome == "safety-blocked" else "response"
                    print(f"Opener attempt {attempt}/{effective_max_attempts}: {attempt_outcome} "
                          f"{result_noun} (billed {_display_cost(cost)}): {e}")
                    # Durable record of EVERY rejected attempt, not just successes (see
                    # ranker/bigquery_store.py's opener_rejections table and opener.py's
                    # OpenerParseError docstring for reason_code/raw_opener semantics).
                    # Recorded HERE, before the retry/exhaust decision below, so the FINAL
                    # attempt -- the one that calls _exhaust() and stops the run -- is
                    # captured too, not only the attempts that go on to a further retry.
                    # Without this, only SUCCEEDED openers were ever persisted, so there was
                    # no way to ask how often any guard fires or whether the deterministic
                    # detectors (scaffolding, sentence-count, ...) are too strict or too
                    # loose. Guarded exactly like record_spend just above: a store outage
                    # must never take down opener generation, the one invariant this whole
                    # file exists to protect.
                    if not advisory:
                        try:
                            self.store.record_opener_rejection(
                                run_id, app, e.model, attempt, e.reason_code, str(e), e.raw_opener)
                        except Exception as store_exc:  # noqa: BLE001
                            print(f"Warning: failed to persist opener rejection record: {store_exc}")
                    # In-memory mirror of the row just above, independent of the store call's
                    # success -- see recent_rejections' docstring in __init__ for why this
                    # exists (the bug report's Recent opener rejections section reads this,
                    # not the store, exactly like recent_openers/recent_openers_snapshot).
                    if not advisory:
                        self.recent_rejections.append({
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "app": app,
                            "model": e.model,
                            "attempt": attempt,
                            "reason_code": e.reason_code,
                            "reason": str(e),
                            "raw_opener": e.raw_opener,
                        })

                    if self.disabled:
                        # _exhaust() already ran above (the unpriceable-model guard) -- the
                        # service is globally disabled now, so there is nothing left to retry.
                        return None
                    if self.tracker.budget_reached():
                        # Re-checked here, not just before the first attempt: the retry rule
                        # never overrides the run budget, so a cap crossed by THIS attempt's
                        # spend must stop further retries immediately.
                        self._exhaust("run budget reached", request_stop=not advisory)
                        return None
                    if e.reason_code in (REASON_PROMPT_BLOCKED, REASON_RESPONSE_BLOCKED):
                        # Gemini identified this as a content-policy block, not malformed JSON
                        # or a stochastic bad draw.  Retrying another model with the same
                        # profile captures would simply resubmit the content Gemini withheld;
                        # do not evade that decision by rewriting, stripping, or isolating the
                        # profile.  The blocked attempt has already been billed and recorded
                        # above. Keep the service healthy for the next profile. In AUTO the
                        # ranker has already made the like decision, so this one explicit
                        # outcome permits a commentless like; Observe displays the reason and
                        # remains live without changing its manual controls.
                        blocked_stage = (
                            "the request" if e.reason_code == REASON_PROMPT_BLOCKED
                            else "the generated response"
                        )
                        self.last_skip_reason = (
                            f"Gemini's safety policy withheld {blocked_stage} for this profile; "
                            "it was not retried and no opener will be used for it. AUTO may "
                            "proceed with its existing like decision without a comment. Future "
                            "profiles remain eligible."
                        )
                        self.last_skip_allows_commentless_like = not advisory
                        action = ("AUTO will proceed with its existing like decision without a "
                                  "comment" if not advisory else "Observe will show no suggestion")
                        print(
                            "Opener: Gemini safety policy withheld content for this profile; "
                            f"not retrying the same captured content. {action}; future profiles "
                            "remain eligible."
                        )
                        return None
                    if attempt >= effective_max_attempts:
                        # Every attempt for this profile came back unusable. Per the owner's
                        # explicit rule, that many failures in a row is no longer one-off bad
                        # luck -- it means something is actually wrong -- so the whole run
                        # stops rather than ever falling back to a bare/commentless like.
                        # Advisory exhaustion disables further suggestions to prevent an
                        # unbounded per-profile spend loop, but request_stop=False keeps the
                        # human observation session itself running.
                        self._exhaust(
                            f"{attempt} consecutive AI opener attempts for this profile were "
                            f"all rejected as unusable (most recent: {e}) -- retrying a bad "
                            "response is expected, but this many failures in a row means "
                            "something is actually wrong (a broken prompt, a degenerate model, "
                            "a schema bug), so opener generation is being disabled rather than "
                            "keep spending on retries that keep failing",
                            request_stop=not advisory,
                        )
                        return None

                    # Retry: tell the model exactly what was wrong with its last attempt so a
                    # re-ask can actually fix it, rather than blindly repeating the same draw.
                    retry_hint = (
                        f"Your previous attempt was rejected and NOT sent: {e}. Write a new "
                        "opener that fixes this."
                    )
                    print(f"Opener: retrying this profile "
                          f"(attempt {attempt + 1}/{effective_max_attempts})...")
                    continue
                except OpenerError as e:
                    # Same class of per-profile failure, but raised without usage attached, so
                    # there is no billed amount to record -- just skip the opener this once.
                    # NOT retried, unlike OpenerParseError above: this is almost always
                    # deterministic (e.g. a corrupt photo that fails to decode -- see
                    # opener.py's _fit_images_to_budget), so a re-ask would resend the exact
                    # same bad payload and fail identically, making a retry pure wasted spend.
                    self._consecutive_bad_requests = 0
                    # Deliberately COUNTS toward the transient-failure latch rather than resetting
                    # it -- the opposite of OpenerParseError just above. Unlike OpenerParseError,
                    # reaching this branch does NOT prove the request reached (or was billed by)
                    # the provider: opener.py can raise OpenerError before any network call at all
                    # (e.g. its _fit_images_to_budget step failing to decode a corrupt photo -- see
                    # item B), so it proves nothing about whether the pipeline is healthy. A single
                    # corrupt screenshot is normal and must not escalate on its own -- but a
                    # SYSTEMICALLY corrupt capture pipeline (every `adb screencap` producing
                    # truncated PNGs, say) would surface as a long run of exactly this error, one
                    # per profile, and that is precisely the silent-forever degradation this latch
                    # exists to catch.
                    if self._register_transient_failure(e, request_stop=not advisory):
                        # Latched permanent by _register_transient_failure -- exhausted_reason
                        # (a global cause) now names this, so last_skip_reason (a per-call cause)
                        # is deliberately left alone rather than shadowing it with this profile's
                        # individual symptom.
                        pass
                    else:
                        self.last_skip_reason = f"opener call failed for this profile: {e}"
                        print(f"Opener: {e}; swiping without an opener for this profile.")
                    return None
                except Exception as e:  # noqa: BLE001
                    # HTTP-level / provider-level failures land here, and none of them are
                    # retried BY THIS METHOD: GeminiOpener.generate() already cascades a
                    # 429/404/5xx/transport failure across every configured model internally
                    # before it ever raises (see its class docstring) -- so by the time an
                    # exception reaches this branch, the whole model cascade already ran its
                    # course for this call, and looping again here would just repeat work
                    # generate() already did rather than genuinely try something new.
                    if isinstance(e, GeminiCapacityExhausted):
                        # Use the exception's own text rather than a generic line: GeminiOpener
                        # builds it per model and distinguishes "every model is out of DAILY
                        # quota, nothing works until the midnight Pacific reset" from "the
                        # cascade fell through with a transient per-minute cap in it, so a
                        # restart shortly may just work". This string becomes the hub's stop
                        # reason, and those two situations need opposite operator responses.
                        self._exhaust(str(e), request_stop=not advisory)
                        return None
                    # Permanent classification MUST run before the bad-request streak counter
                    # below: an invalid/revoked Gemini API key surfaces as an HTTP 400 (see
                    # _is_invalid_gemini_api_key), and it must latch immediately on the first
                    # occurrence like a 401 would, not spend 2 more profiles feeding the
                    # malformed-request streak and then get reported with the wrong diagnosis.
                    reason = _permanent_reason(e)
                    if reason is not None:
                        # Permanent (bad key/model): retrying it per-profile is pure waste, so
                        # stop attempting openers for the rest of the run instead.
                        self._exhaust(reason, request_stop=not advisory)
                        return None
                    if _is_bad_request(e):
                        # Only latch permanent after a STREAK of consecutive 400s -- see
                        # _BAD_REQUEST_LATCH_THRESHOLD's docstring for why a lone 400 must not
                        # kill openers for the whole run (it can be that profile's own bad
                        # capture, not a systemic request problem).
                        self._consecutive_bad_requests += 1
                        if self._consecutive_bad_requests >= _BAD_REQUEST_LATCH_THRESHOLD:
                            self._exhaust(
                                f"The opener provider rejected {self._consecutive_bad_requests} opener "
                                "requests in a row as malformed (bad param/schema) -- this looks "
                                "like a systemic request problem, not a one-off bad photo",
                                request_stop=not advisory,
                            )
                            return None
                        # Below the latch: this is a per-call reason (see last_skip_reason's
                        # docstring), not yet a global exhaustion -- the service is still enabled
                        # and expected to serve the next profile normally.
                        self.last_skip_reason = (
                            f"opener request for this profile was rejected as malformed "
                            f"(HTTP 400): {e}"
                        )
                        print(f"Opener skipped (400 on this profile's request -- could be a bad "
                              f"capture; {self._consecutive_bad_requests}/"
                              f"{_BAD_REQUEST_LATCH_THRESHOLD} in a row before treating it as "
                              f"permanent, swiping without): {e}")
                        return None
                    self._consecutive_bad_requests = 0
                    # Transient network/timeout/rate-limit errors: skip this profile's opener
                    # but keep the service enabled so subsequent profiles can still get openers
                    # -- UNLESS this is now a streak (see _TRANSIENT_LATCH_THRESHOLD): an audit of
                    # this project demonstrated 50 consecutive bare RuntimeErrors sailing through
                    # this exact branch with the service left enabled and nothing on the hub
                    # showing anything was wrong, each one a live profile that got a bare like
                    # with no opener.
                    if self._register_transient_failure(e, request_stop=not advisory):
                        # Latched -- exhausted_reason now carries the global cause; leave
                        # last_skip_reason (a per-call cause) alone, same reasoning as the
                        # OpenerError branch above.
                        pass
                    else:
                        self.last_skip_reason = (
                            f"opener call failed for this profile (transient error): "
                            f"{type(e).__name__}: {e}"
                        )
                        print(f"Opener skipped (transient error, swiping without): "
                              f"{type(e).__name__}: {e}")
                    return None

                # Success -- whether on the first attempt or after one or more retries.
                self._consecutive_bad_requests = 0   # success -- request shape is fine
                self._consecutive_transient_failures = 0   # success -- the pipeline works end to end
                # A successful, end-to-end call proves the LAST failure (if any) is over -- clear
                # it so a stale per-call reason from a prior profile never lingers and gets
                # misread as describing this one (see last_skip_reason's docstring).
                self.last_skip_reason = None

                # ENTROPY GUARD (ops/OPENER-REDESIGN.md 3.6, and see _apply_entropy_guard for
                # the full rationale, INCLUDING THE 2026-08-11 CORRECTION: this now runs
                # identically whether or not the call is advisory -- it takes no `advisory`
                # argument at all any more, deliberately, because nothing about its behavior may
                # ever depend on that flag (see the ADVISORY ASYMMETRY paragraph in its
                # docstring for why the OLD advisory-skips-it design was wrong, not merely
                # conservative). It may hand back a SECOND draw taken because this one opened
                # with words already sent this run. That extra call must not consume the
                # per-profile attempt budget whose exhaustion stops the whole run, and it does
                # not: these lines are lexically inside the `for attempt in ...` loop, but they
                # sit on the SUCCESS path, which returns unconditionally a few lines below, so no
                # extra draw can ever cause another `attempt` iteration -- true in AUTO and in
                # advisory alike. Keep the guard on a returning path if it ever moves.
                #
                # It runs BEFORE any spend/opener recording below for one reason: exactly one
                # `openers` row per profile must exist, and it must hold the text that actually
                # gets sent. The draft the guard discards was still billed, so the guard records
                # THAT spend itself -- the money is tracked either way, while the opener row is
                # written once, below, for whichever draw survives.
                result, entropy_collision, entropy_regenerated = self._apply_entropy_guard(
                    run_id, profile, result,
                    items=items, should_stop=should_stop,
                    skip_models=frozenset(failed_models),
                    deadline=(advisory_deadline_at if client_accepts_deadline else None),
                    client_accepts_deadline=client_accepts_deadline)

                try:
                    cost = self.tracker.record(result.model, result.usage)
                except KeyError:
                    # The API already ran (real credits spent) but its response echoed a model
                    # string with no budget.pricing entry, so spend can't be accounted for.
                    # Degrade the same way as budget-reached/out-of-credit rather than crash —
                    # continuing to spend with no way to track it would silently break the
                    # budget-enforcement contract the rest of this service is built around.
                    # cost=None (not 0.0): the real cost was nonzero, just unrecoverable, and
                    # a fabricated $0.00 would misreport actual spend in the stored record.
                    self._exhaust(f"no budget.pricing entry for model '{result.model}'; "
                                  "spend can no longer be tracked", request_stop=not advisory)
                    cost = None
                # getattr defaults here for the same defensive reason as the pre-existing
                # getattr on the old referenced_index (this block used to read only that
                # field): a simple/fake OpenerClient used by a test, or some future call site,
                # need not populate every field on the OpenerResult it constructs, and a bare
                # AttributeError from a call that otherwise fully succeeded would be a strange
                # way for this method to fail. Read HERE, above the store calls, rather than
                # after them, because `angle` and `item_description` are both part of the
                # persisted opener row below.
                #
                # `item_index` REPLACES `referenced_index` and is a different quantity, not a
                # rename: 1-based over the NUMBERED ITEMS the model was shown, where the old
                # field was 0-based over raw scroll frames (see OpenerPick's docstring and
                # ops/OPENER-REDESIGN.md 5.1/5.7). The default is ITEM_INDEX_ABSENT rather than
                # a bare 0 for the same reason: 0 is out of band under the new contract, and a
                # stand-in result that never set the field must not read as "she picked item 1".
                # Deliberately NO getattr fallback to "referenced_index": silently accepting an
                # old 0-based frame index here and treating it as an item number is precisely
                # the reinterpretation doc 5.3 exists to prevent, so a stale producer degrades
                # to ABSENT (loud, unusable) instead of to a plausible wrong item.
                item_index = getattr(result, "item_index", ITEM_INDEX_ABSENT)
                # WHICH LIST that number counts, straight off the result rather than inferred
                # from anything here. The getattr default is the UNTRANSLATABLE space, matching
                # OpenerResult's own default: a client that does not state its space gets a
                # pick nothing will convert into a tap, which is the safe direction. Reading it
                # from `result` (not from, say, whether `items` was passed) keeps one producer
                # of this fact -- generate(), which built the payload -- instead of two that
                # can drift.
                index_space = str(getattr(result, "index_space", INDEX_SPACE_MODEL_ITEMS)
                                  or INDEX_SPACE_MODEL_ITEMS)
                referenced = str(getattr(result, "referenced", "") or "")
                # The model's own words for what this opener is DOING -- telemetry only, never
                # read to make a decision (see OpenerPick.angle). "" when the model omitted it
                # or returned a non-string; opener.py has already stripped it.
                angle = str(getattr(result, "angle", "") or "")
                # The model's own short description of the ITEM it picked (doc 5.7). Read
                # unconditionally, with no branch on `advisory` anywhere: auto logs it, observe
                # displays it, and the moment one mode stops asking for it the two modes stop
                # issuing the same request, which is the canary property observe exists for.
                item_description = str(getattr(result, "item_description", "") or "")
                # Output of opener.py's deterministic redundancy MONITOR: content words this
                # opener restated from its own `referenced` note (doc 3.7). isinstance-checked
                # rather than trusted, so a fake/older result carrying a string or None here
                # degrades to [] instead of being iterated character by character into twelve
                # bogus markers. LOG ONLY, and that is not a soft preference: this monitor is a
                # documented LOWER BOUND on redundancy (a terse `referenced` defeats it
                # entirely), it has never been calibrated against real data, and five
                # consecutive rejections stop the whole run -- gating on it would make an
                # uncalibrated metric a run-killer. It never rejects anything here, ever.
                raw_markers = getattr(result, "redundancy_markers", None)
                redundancy_markers = ([str(m) for m in raw_markers]
                                      if isinstance(raw_markers, (list, tuple)) else [])
                if redundancy_markers:
                    # opener.py prints its own line when it computes these, but that fires for
                    # every PARSED draft, including one the entropy guard then throws away.
                    # This line fires only for the opener that is actually about to be sent,
                    # which is the population the offline calibration in doc 3.7 needs to count.
                    print(f"Opener: the opener being sent restates "
                          f"{len(redundancy_markers)} word(s) from its own `referenced` note "
                          f"({'; '.join(redundancy_markers)}). Logged only, never a rejection.")
                staged_for_action = advisory or stage
                try:
                    self.store.record_spend(run_id, result.model, result.usage, cost)
                    # angle and item_description ride along as the 6th and 7th POSITIONAL
                    # arguments: both stores declare them as trailing `angle: str = ""` /
                    # `item_description: str = ""` parameters (see ranker/store.py and
                    # ranker/bigquery_store.py, each with its own column migration), and a
                    # positional call keeps working against the fakes in the test suite that
                    # accept *a. `referenced` is the defensively-read local above rather than
                    # result.referenced: a result missing that attribute should persist an empty
                    # note, not blow up mid-try and get reported as a failure to persist SPEND,
                    # which is a misleading thing to print about an AttributeError on
                    # `referenced` (the spend row above it was already written fine).
                    #
                    if not staged_for_action:
                        self.store.record_opener(run_id, app, result.model, result.opener,
                                                 referenced, angle, item_description)
                except Exception as e:  # noqa: BLE001
                    # Spend was already tracked in-memory by CostTracker (or deliberately
                    # marked unrecoverable above); store failure is non-fatal.
                    print(f"Warning: failed to persist opener spend record "
                          f"({_display_cost(cost)}): {e}")
                if self.tracker.budget_reached():
                    self._exhaust("run budget reached", request_stop=not advisory)
                # A diagnostic opener entry is committed only for AUTO, or later for a confirmed
                # Observe Like.  An advisory suggestion is a draft and must not make an
                # abandoned profile look acted-on in a bug report.
                #
                # The three newer fields all exist to make a redesign VISIBLE in a bug report
                # rather than only in a console line nobody kept: `angle` is the model's own
                # account of what it was doing, `redundancy_markers` is the over-description
                # monitor's verdict on the text actually sent, and `entropy_collision` /
                # `entropy_regenerated` record whether this opener repeated an earlier opening
                # and whether a second draw was taken -- reflecting a REAL collision check on
                # every call, advisory included as of the 2026-08-11 correction (previously
                # hardcoded to "" / False on an advisory call because the guard was skipped
                # outright there; see _apply_entropy_guard's ADVISORY ASYMMETRY paragraph for why
                # that was wrong). Extra keys are safe here: bugreport.py reads this dict key by
                # key with .get defaults, never by unpacking or exact comparison.
                #
                # The "index" KEY KEEPS ITS NAME AND CHANGES ITS MEANING, which is worth
                # knowing before comparing two bug reports across this commit: entries written
                # before 2026-08-12 hold a 0-based index into the raw scroll frames, entries
                # after hold the 1-based MODEL ITEM INDEX (doc 5.1/5.7, and OpenerPick's
                # docstring). The key was not renamed because bugreport.py reads it by that
                # name and a report is a human-read artefact, but `item_description` sitting
                # beside it is what makes the two eras distinguishable at a glance: it is
                # simply absent from every pre-change entry.
                #
                # "index_space" rides alongside it so a report never has to be dated to be
                # read: it names the list "index" counts, which is the one fact a bare small
                # integer cannot carry and whose absence is what let the two spaces be confused
                # in the first place.
                recent_entry = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "app": app,
                    "model": result.model,
                    # This remains an advisory-generation record even though it is appended
                    # only after a confirmed Like.  The flag describes how the opener was
                    # produced (Observe vs AUTO), not whether the staged row was committed.
                    "advisory": bool(advisory),
                    "index": item_index,
                    "index_space": index_space,
                    "referenced": referenced,
                    "angle": angle,
                    "item_description": item_description,
                    "redundancy_markers": redundancy_markers,
                    "entropy_collision": entropy_collision,
                    "entropy_regenerated": entropy_regenerated,
                    "opener": result.opener,
                }
                if not staged_for_action:
                    self.recent_openers.append(recent_entry)
                ngram = _leading_ngram(result.opener, _ENTROPY_NGRAM_WORDS)
                if ngram:
                    self._recent_opening_ngrams.append(ngram)
                # Positional, matching OpenerPick's field order: text, index (the MODEL ITEM
                # INDEX now -- see that dataclass's docstring), referenced, angle,
                # item_description. `index_space` is passed by KEYWORD rather than as a sixth
                # positional: it is the field that makes `index` interpretable at all, and a
                # bare trailing string in a five-argument positional call is exactly the kind
                # of thing a later edit drops or reorders without noticing.
                staged = (_StagedOpenerRecord(
                    run_id, app, result.model, result.opener, referenced, angle,
                    item_description, recent_entry) if staged_for_action else None)
                return OpenerPick(result.opener, item_index, referenced, angle, item_description,
                                  index_space=index_space, _staged_record=staged)
            # Unreachable in practice: __init__ now rejects any max_attempts that isn't an
            # int >= 1 (see BUG 2), so range(1, effective_max_attempts + 1) always yields at
            # least one iteration, and
            # every iteration above returns -- from the should_stop check, a non-retried
            # exception, the success branch, or _exhaust() on the final retry attempt. Before
            # that guard existed, OpenerService(max_attempts=0) made this loop body never run at
            # all: 0 API calls, disabled stayed False, stop_requested stayed False, no reason was
            # ever recorded, and control fell straight through to this exact line on every call
            # -- reporting perfectly healthy while producing zero openers, forever. Kept as a
            # defensive fallback so this method's return type stays honest even if that invariant
            # is ever broken by a future edit.
            return None  # pragma: no cover

    def commit_opener(self, pick: OpenerPick, *, profile_id: str = "", decision: str = "like",
                      decision_source: str = "", decision_created_at: object | None = None,
                      pre_send_evidence: dict[str, object] | None = None) -> bool:
        """Persist one staged AUTO/Observe opener after a landed Like, exactly once.

        This is intentionally a separate, explicit commit from generation: neither opening a
        profile nor generating a suggestion says that the account acted.  Pass, Stop, resync,
        and every pre-tap refusal simply drop the in-memory envelope with the card.
        """
        with self._lock:
            record = getattr(pick, "_staged_record", None)
            if record is None:
                return False
            try:
                _record_staged_opener(
                    self.store, record, pick, profile_id=profile_id, decision=decision,
                    decision_source=decision_source, decision_created_at=decision_created_at,
                    pre_send_evidence=pre_send_evidence)
            except Exception as exc:  # noqa: BLE001 -- a store outage must not erase a real Like
                print(f"Warning: failed to persist committed opener after landed like: {exc}")
                return False
            self.recent_openers.append(dict(record.recent_entry))
            pick._staged_record = None
            return True

    def commit_advisory_opener(self, pick: OpenerPick, **lineage) -> bool:
        """Compatibility alias for Observe callers; commit semantics are now generic."""
        return self.commit_opener(pick, **lineage)

    def recent_openers_snapshot(self) -> list[dict]:
        """A copy of the most recent committed opener records -- see recent_openers'
        docstring in __init__ for exactly what each entry records and why (the request item
        space and the model's own `referenced` claim diagnose an out-of-place opener).

        Returns list(self.recent_openers) under self._lock rather than handing back
        self.recent_openers itself: a bug-report reader running on another thread must never
        iterate a deque that a live worker thread is concurrently appending to (maybe_opener
        holds this same lock across its whole body, including the append), and a plain list is
        also a stable, JSON-serializable snapshot rather than a live view that keeps changing
        under the reader's feet."""
        with self._lock:
            return list(self.recent_openers)

    def recent_rejections_snapshot(self) -> list[dict]:
        """A copy of the most recent REJECTED opener attempts -- see recent_rejections'
        docstring in __init__ for exactly what each entry records. Same reasoning as
        recent_openers_snapshot above: returns a plain list copy under self._lock rather than
        the live deque, so a bug-report reader on another thread never iterates a deque a
        worker thread is concurrently appending to."""
        with self._lock:
            return list(self.recent_rejections)

    def _exhaust(self, reason: str, *, request_stop: bool = True) -> None:
        """Flip the service permanently disabled and record WHY (exhausted_reason), so an
        operator staring at a stopped or degraded run sees the actual cause instead of a
        generic line. Every AUTO-path exhaustion (the module docstring's list: budget
        reached, a latch tripped, a permanent provider error, GeminiCapacityExhausted, every
        retry attempt failing) reaches here with the default request_stop=True.

        request_stop=False is for an ADVISORY call only (see maybe_opener's advisory
        parameter): a Hinge Observe draft is still pre-action, so a bad response cannot
        justify forcing the session itself to stop. Ending the whole Observe session over it
        would sacrifice real training labels for something purely cosmetic -- the owner's rule
        is "stop the AUTOMATION", and a draft has not made an action.
        disabled/exhausted_reason are still set exactly as for an AUTO exhaustion (a
        systematically broken model/prompt must still stop
        burning further quota on suggestions no one will ever see used), and this method
        remains first-writer-wins for exhausted_reason regardless of which kind of call gets
        there first -- only the stop_requested side effect is conditional.
        """
        with self._lock:
            if not self.disabled:
                print(f"Opener: {reason} -> " + (
                    "stopping all workers" if request_stop else
                    "disabling further opener attempts this run (advisory suggestion only "
                    "-- the observe session continues; see maybe_opener's advisory parameter)"
                ))
            # First-writer-wins: once exhausted, later calls into _exhaust (e.g. a second
            # worker's own opener call failing right behind the first, now that the service
            # is disabled) must not clobber the reason an operator actually needs -- the
            # original cause, not whatever incidental error a later caller happened to hit.
            if self.exhausted_reason is None:
                self.exhausted_reason = reason
            self.disabled = True
            # request_stop=True (the default) is unconditional otherwise: there is no longer a
            # configuration that keeps AUTOMATION going once the service has given up (see this
            # class's module docstring for why the old budget.on_exhausted=
            # "swipe_without_opener" mode was removed outright). request_stop=False (advisory)
            # deliberately leaves stop_requested untouched -- see this method's own docstring.
            if request_stop:
                self.stop_requested = True
