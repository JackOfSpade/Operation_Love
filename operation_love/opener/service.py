"""Shared, budget-aware opener generation across all app workers.

One OpenerService is shared by every worker (Bumble, Hinge, ...) so the per-run
spend cap is GLOBAL, not per-app. Thread-safe. Handles the ways opener generation
ends for a run: the configured run budget, the actual Anthropic out-of-credit
error, and a permanent provider failure (bad/missing API key, no model access,
retired model id, or -- only after several back-to-back malformed-request
errors -- a broken request schema/params) -- each flips the service to disabled
and (if on_exhausted="stop") asks the supervisor to stop all workers.

A LONE 400 (BadRequestError) is treated as transient, not permanent:
AnthropicOpener._content() embeds that specific profile's own captured photos in
every request, so a single 400 can be caused by a profile-specific payload
problem (a truncated/corrupt PNG from a flaky `adb screencap`, an oversized
image, too many photos) rather than by the request itself being malformed. Only
a run of consecutive 400s (see _BAD_REQUEST_LATCH_THRESHOLD) -- which a genuine
schema/param bug produces deterministically, since it re-fires on literally the
next call regardless of that call's content -- latches the service permanently
disabled.

A per-call failure (refusal, or truncated/malformed structured output -- OpenerError /
OpenerParseError) is handled far more narrowly: it is almost always specific to THAT
profile's content, so the swipe just proceeds without an opener. The call is still billed,
so its spend is recorded before degrading.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from ..costing import CostTracker, is_out_of_credit
from ..perception.capture import Profile
from .opener import OpenerClient, OpenerError, OpenerParseError


@dataclass
class OpenerPick:
    """An opener plus which profile item (0-based index, capture order) it is about, so the
    driver can attach the comment to the RIGHT photo/prompt instead of always the first."""
    text: str
    index: int = 0


# How many consecutive BadRequestErrors (HTTP 400) to require before treating the failure
# as permanent and latching the service disabled. A genuinely broken request (bad schema,
# bad param) is deterministic: it re-fires on the very next call no matter which profile's
# content is sent, so it still latches within a couple of profiles. A profile-specific
# payload problem (corrupt screenshot, oversized image) is very unlikely to repeat back to
# back across unrelated profiles' photos, so requiring a streak protects it from a single
# unlucky capture while still catching a systemic problem quickly.
_BAD_REQUEST_LATCH_THRESHOLD = 3


def _permanent_reason(exc: Exception) -> str | None:
    """Classify a provider error as PERMANENT (will fail identically on every remaining
    profile this run: bad/missing key, no model access, retired model id) vs. TRANSIENT
    (rate limit, 5xx, connection blip -- worth retrying on the next profile). Returns an
    operator-facing explanation of what's wrong, or None if exc isn't a recognized permanent
    failure. Deliberately does NOT classify BadRequestError -- see _is_bad_request and
    OpenerService._consecutive_bad_requests, since a single 400 can be caused by that
    profile's own captured photos rather than by a systemic request problem. The anthropic
    SDK is optional here, so import lazily and degrade to "can't classify -> treat as
    transient" if it isn't installed."""
    try:
        import anthropic
    except ImportError:
        return None
    if isinstance(exc, anthropic.AuthenticationError):
        return "ANTHROPIC_API_KEY is missing or invalid"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "ANTHROPIC_API_KEY does not have access to the configured opener model"
    if isinstance(exc, anthropic.NotFoundError):
        return "opener.model in config.yaml is not a valid/available model id"
    return None


def _is_bad_request(exc: Exception) -> bool:
    """True if exc is Anthropic's 400 BadRequestError. Split out from _permanent_reason
    because a 400 is only sometimes permanent -- see _BAD_REQUEST_LATCH_THRESHOLD. Same
    lazy-import/optional-SDK handling as _permanent_reason."""
    try:
        import anthropic
    except ImportError:
        return False
    return isinstance(exc, anthropic.BadRequestError)


class OpenerService:
    def __init__(self, client: OpenerClient | None, tracker: CostTracker, store,
                 style: str, on_exhausted: str = "stop"):
        self.client = client
        self.tracker = tracker
        self.store = store
        self.style = style
        self.on_exhausted = on_exhausted
        self.disabled = client is None
        self.stop_requested = False     # set when opener generation is exhausted (budget /
                                         # credit / permanent provider error) and on_exhausted="stop"
        self._consecutive_bad_requests = 0   # streak of back-to-back 400s; see
                                              # _BAD_REQUEST_LATCH_THRESHOLD
        self._lock = threading.RLock()

    def maybe_opener(self, run_id: str, app: str, profile: Profile) -> "OpenerPick | None":
        """Return an OpenerPick (text + referenced item index), or None (disabled / budget
        reached / out of credit / permanent provider error / unparseable response).

        The budget check, provider call, and spend recording are serialized so
        concurrent app workers cannot all pass the pre-call budget check and
        overspend the shared run budget at once.
        """
        with self._lock:
            if self.disabled:
                return None
            if self.tracker.budget_reached():
                self._exhaust("run budget reached")
                return None
            try:
                result = self.client.generate(profile, self.style)
            except OpenerParseError as e:
                # The call reached the API and was billed (it returned usage) even though
                # the body didn't parse into a usable opener -- record the spend like a
                # normal call, then degrade to swiping without an opener this time only.
                # Reaching the API at all (billed usage came back) proves the request shape
                # itself is fine, so this breaks any streak of bad-request failures too.
                self._consecutive_bad_requests = 0
                try:
                    cost = self.tracker.record(e.model, e.usage)
                except KeyError:
                    # Same guard as the success path: the API echoed a model with no
                    # budget.pricing entry, so this (real, billed) spend can't be tracked.
                    self._exhaust(f"no budget.pricing entry for model '{e.model}'; "
                                  "spend can no longer be tracked")
                    cost = None
                try:
                    self.store.record_spend(run_id, e.model, e.usage, cost)
                except Exception as store_exc:  # noqa: BLE001
                    print(f"Warning: failed to persist opener spend record (${cost:.4f}): {store_exc}")
                print(f"Opener skipped (unparseable response, billed ${cost:.4f}, "
                      f"swiping without): {e}")
                if self.tracker.budget_reached():
                    self._exhaust("run budget reached")
            except OpenerError as e:
                # Same class of per-profile failure, but raised without usage attached, so
                # there is no billed amount to record -- just skip the opener this once.
                self._consecutive_bad_requests = 0
                print(f"Opener: {e}; swiping without an opener for this profile.")
                return None
            except Exception as e:  # noqa: BLE001
                if is_out_of_credit(e):
                    self._exhaust("Claude credit exhausted")
                    return None
                if _is_bad_request(e):
                    # Only latch permanent after a STREAK of consecutive 400s -- see
                    # _BAD_REQUEST_LATCH_THRESHOLD's docstring for why a lone 400 must not
                    # kill openers for the whole run (it can be that profile's own bad
                    # capture, not a systemic request problem).
                    self._consecutive_bad_requests += 1
                    if self._consecutive_bad_requests >= _BAD_REQUEST_LATCH_THRESHOLD:
                        self._exhaust(
                            f"Anthropic API rejected {self._consecutive_bad_requests} opener "
                            "requests in a row as malformed (bad param/schema) -- this looks "
                            "like a systemic request problem, not a one-off bad photo"
                        )
                        return None
                    print(f"Opener skipped (400 on this profile's request -- could be a bad "
                          f"capture; {self._consecutive_bad_requests}/"
                          f"{_BAD_REQUEST_LATCH_THRESHOLD} in a row before treating it as "
                          f"permanent, swiping without): {e}")
                    return None
                self._consecutive_bad_requests = 0
                reason = _permanent_reason(e)
                if reason is not None:
                    # Permanent (bad key/model): retrying it per-profile is pure waste, so
                    # stop attempting openers for the rest of the run instead.
                    self._exhaust(reason)
                    return None
                # Transient network/timeout/rate-limit errors: skip this profile's opener
                # but keep the service enabled so subsequent profiles can still get openers.
                print(f"Opener skipped (transient error, swiping without): "
                      f"{type(e).__name__}: {e}")
                return None

            self._consecutive_bad_requests = 0   # success -- request shape is fine

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
                              "spend can no longer be tracked")
                cost = None
            try:
                self.store.record_spend(run_id, result.model, result.usage, cost)
                self.store.record_opener(run_id, app, result.model, result.opener, result.referenced)
            except Exception as e:  # noqa: BLE001
                # Spend was already tracked in-memory by CostTracker (or deliberately
                # marked unrecoverable above); store failure is non-fatal.
                print(f"Warning: failed to persist opener spend record (${cost}): {e}")
            if self.tracker.budget_reached():
                self._exhaust("run budget reached")
            return OpenerPick(result.opener, getattr(result, "referenced_index", 0))

    def _exhaust(self, reason: str) -> None:
        with self._lock:
            if not self.disabled:
                action = "stopping all workers" if self.on_exhausted == "stop" else "swiping without openers"
                print(f"Opener: {reason} -> {action}")
            self.disabled = True
            if self.on_exhausted == "stop":
                self.stop_requested = True
