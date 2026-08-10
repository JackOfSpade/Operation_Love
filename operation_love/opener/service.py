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
commentless like is never acceptable. If the AI's response is bad, redo the prompt --
spending more usage is fine. Only if a profile's response is STILL bad after
max_attempts tries does that mean something is actually wrong, and only then does the
whole run stop. The old budget.on_exhausted="swipe_without_opener" configuration was
removed outright rather than left dormant -- the same reasoning that removed the
Anthropic provider path applies here: a setting that can silently reintroduce a banned
behavior is worse than no setting at all. Every exhaustion path below now
unconditionally sets stop_requested; there is no remaining way to configure this
service to keep swiping once it gives up on a profile or on the run.

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
help. OpenerParseError (a billed response that didn't parse into a usable opener) is
RETRIED, up to self.max_attempts times, with the model told what was wrong on each
retry -- see maybe_opener's docstring for the full loop and why: a parse failure is
usually stochastic (the model drew a bad sample), so a re-ask genuinely has a decent
chance of producing something usable, and the owner's explicit instruction is to keep
trying rather than fall back to a bare like. OpenerError (refusal, or a corrupt photo
that can't even be encoded) is NOT retried -- it is almost always specific to THAT
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
OpenerError (a refusal, or -- see opener.py's item B fix -- a corrupt captured photo) is,
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

import threading
from dataclasses import dataclass
from typing import Callable

from ..costing import CostTracker
from ..perception.capture import Profile
from .opener import (
    GeminiAPIError,
    GeminiCapacityExhausted,
    OpenerAborted,
    OpenerClient,
    OpenerError,
    OpenerParseError,
)


@dataclass
class OpenerPick:
    """An opener plus which profile item (0-based index, capture order) it is about, so the
    driver can attach the comment to the RIGHT photo/prompt instead of always the first."""
    text: str
    index: int = 0


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
    def __init__(self, client: OpenerClient | None, tracker: CostTracker, store,
                 style: str, max_attempts: int = 5):
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
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError(
                f"OpenerService max_attempts must be an int >= 1, got {max_attempts!r}")
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
        self.disabled = client is None
        self.stop_requested = False     # set whenever _exhaust() runs (budget reached / out
                                         # of credit / permanent provider error / max_attempts
                                         # consecutive bad responses for one profile) -- see
                                         # _exhaust; there is no configuration that keeps a run
                                         # going once this is set (the owner's rule requires
                                         # halting, not silently degrading to bare likes).
        # Human-readable cause of the FIRST exhaustion (see _exhaust) -- e.g. "run budget
        # reached" or "all configured Gemini models exhausted their free-tier quota; no
        # opener capacity remains". Every worker sharing this service reads the same value,
        # so whichever app happened to trigger it, every worker's hub status shows the one
        # true original cause rather than each other's follow-on symptoms (e.g. a second
        # worker's own maybe_opener() call failing merely because the service is already
        # disabled). None until the service has actually exhausted once.
        self.exhausted_reason: str | None = None
        # Human-readable cause of the LAST maybe_opener() call that returned None WITHOUT
        # exhausting the service -- an unparseable response (OpenerParseError), a per-profile
        # OpenerError, a single sub-latch HTTP 400, or a single sub-latch transient failure
        # (see maybe_opener). Unlike exhausted_reason this is deliberately LAST-writer-wins
        # and per-call, not first-writer-wins and run-lifetime: exhausted_reason answers "why
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
        self.last_skip_reason: str | None = None
        self._consecutive_bad_requests = 0   # streak of back-to-back 400s; see
                                              # _BAD_REQUEST_LATCH_THRESHOLD
        # Streak of back-to-back TRANSIENT failures (an unclassified exception, or a
        # per-profile OpenerError -- see _register_transient_failure for exactly what counts
        # and _TRANSIENT_LATCH_THRESHOLD for why); reset on any outcome that proves the
        # opener pipeline actually works.
        self._consecutive_transient_failures = 0
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

    def maybe_opener(self, run_id: str, app: str, profile: Profile, *,
                      should_stop: Callable[[], bool] | None = None,
                      advisory: bool = False) -> "OpenerPick | None":
        """Return an OpenerPick (text + referenced item index), or None (disabled / budget
        reached / out of credit / permanent provider error / every retry attempt used up /
        a per-profile OpenerError / a single sub-latch 400 or transient failure / the run
        stopping via should_stop).

        advisory (default False -- unchanged AUTO-mode behavior): True for Hinge's observe-
        mode post-heart suggestion (see worker.py's on_like_intent), where the opener is only
        DISPLAYED next to the app's own comment sheet -- the human retypes and sends it (or
        their own words) themselves. Two consequences follow directly from that:
          1. Exactly ONE attempt is made (effective_max_attempts becomes 1 regardless of
             self.max_attempts): a human is sitting there waiting on this call, so a multi-
             attempt retry storm is a UX problem here, not a spend-protecting safeguard --
             there is no autonomous send to protect from a bad response in the first place.
          2. Every exhaustion path below routes through _exhaust(..., request_stop=False)
             instead of the default request_stop=True. disabled/exhausted_reason are still
             set exactly as for an AUTO exhaustion (so a systematically broken model/prompt
             still stops burning quota on further suggestions), but stop_requested is left
             alone: an advisory suggestion failing is COSMETIC (the human keeps swiping and
             typing their own messages with the app's own controls either way), and ending
             the whole observe session over it would sacrifice the session's entire purpose
             (collecting real training labels) for something that was never going to touch
             automation. See worker.py's _wait_for_observed_decision for the call site and
             this class's own _exhaust() for the matching request_stop plumbing.

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

        THE OWNER'S RULE: a commentless like is never acceptable. If the AI's response is
        bad, redo the prompt -- spending more usage is fine. So a billed response that
        didn't parse into a usable opener (OpenerParseError) is RETRIED for THIS profile,
        up to self.max_attempts times, each retry telling the model what was wrong with its
        previous attempt (retry_hint) so a re-ask can actually do better -- this is worth
        doing because a parse failure is usually stochastic (the same request often
        succeeds on a re-ask), unlike a deterministic failure (see below). Only once
        max_attempts consecutive attempts have ALL failed does that stop looking like
        one-off bad luck and start looking like something actually broken (a bad prompt, a
        degenerate model, a schema bug) -- at that point the whole run stops via
        _exhaust(), exactly like the owner's instruction: "if after 5 attempts it's still a
        bad response, stop the automation -- that means something is wrong."

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
        attempt failing) goes through _exhaust(), which records exhausted_reason and sets
        stop_requested. Every OTHER path that returns None (a per-profile OpenerError, or
        a single sub-latch 400/transient failure) instead records last_skip_reason: this
        call failed, but the service is still enabled and expects to succeed again next
        profile. Callers that must never send a like with no opener (see worker.py's
        _auto_loop) need to distinguish the two, since only the second kind leaves
        self.disabled False. A should_stop-triggered None joins the SECOND group (per-call,
        service stays enabled): the run stopping says nothing about whether the opener
        pipeline itself is healthy, so it must not be reported or counted like one.
        """
        with self._lock:
            if self.disabled:
                return None
            if self.tracker.budget_reached():
                self._exhaust("run budget reached", request_stop=not advisory)
                return None

            # advisory: exactly ONE attempt, never self.max_attempts -- see this method's
            # advisory docstring paragraph. Local variable, not a mutation of self.max_attempts:
            # the instance-level setting is shared by every caller (including concurrent AUTO
            # calls on another app), so a suggestion call must never shrink it for anyone else.
            effective_max_attempts = 1 if advisory else self.max_attempts
            # Which models have already produced an unusable response FOR THIS PROFILE, so a
            # retry can steer GeminiOpener's cascade away from re-hitting the same (often
            # scarcest-quota) model that just failed -- see opener.py's generate() skip_models
            # docstring. Local to this call/profile by construction (a fresh call stack frame
            # per maybe_opener() invocation): it can never leak into the NEXT profile's call,
            # which gets its own empty set here.
            failed_models: set[str] = set()

            retry_hint = ""   # "" means first attempt; a retry fills this in below
            for attempt in range(1, effective_max_attempts + 1):
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
                    result = self.client.generate(profile, self.style, retry_hint=retry_hint,
                                                  should_stop=should_stop,
                                                  skip_models=frozenset(failed_models))
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
                    print(f"Opener attempt {attempt}/{effective_max_attempts}: unparseable "
                          f"response (billed {_display_cost(cost)}): {e}")

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
                    if attempt >= effective_max_attempts:
                        # Every attempt for this profile came back unusable. Per the owner's
                        # explicit rule, that many failures in a row is no longer one-off bad
                        # luck -- it means something is actually wrong -- so the whole run
                        # stops rather than ever falling back to a bare/commentless like.
                        # (advisory: effective_max_attempts is 1, so this fires after the single
                        # allowed attempt -- see this method's advisory docstring paragraph --
                        # and request_stop=not advisory keeps that a display-only degradation.)
                        self._exhaust(
                            f"{attempt} consecutive AI opener attempts for this profile were "
                            f"all rejected as unusable (most recent: {e}) -- retrying a bad "
                            "response is expected, but this many failures in a row means "
                            "something is actually wrong (a broken prompt, a degenerate model, "
                            "a schema bug), so the run is stopping rather than keep spending on "
                            "retries that keep failing",
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
                try:
                    self.store.record_spend(run_id, result.model, result.usage, cost)
                    self.store.record_opener(run_id, app, result.model, result.opener, result.referenced)
                except Exception as e:  # noqa: BLE001
                    # Spend was already tracked in-memory by CostTracker (or deliberately
                    # marked unrecoverable above); store failure is non-fatal.
                    print(f"Warning: failed to persist opener spend record "
                          f"({_display_cost(cost)}): {e}")
                if self.tracker.budget_reached():
                    self._exhaust("run budget reached", request_stop=not advisory)
                return OpenerPick(result.opener, getattr(result, "referenced_index", 0))
            # Unreachable in practice: __init__ now rejects any max_attempts that isn't an
            # int >= 1 (see BUG 2), so range(1, effective_max_attempts + 1) -- 1 for an advisory
            # call, self.max_attempts otherwise -- always yields at least one iteration, and
            # every iteration above returns -- from the should_stop check, a non-retried
            # exception, the success branch, or _exhaust() on the final retry attempt. Before
            # that guard existed, OpenerService(max_attempts=0) made this loop body never run at
            # all: 0 API calls, disabled stayed False, stop_requested stayed False, no reason was
            # ever recorded, and control fell straight through to this exact line on every call
            # -- reporting perfectly healthy while producing zero openers, forever. Kept as a
            # defensive fallback so this method's return type stays honest even if that invariant
            # is ever broken by a future edit.
            return None  # pragma: no cover

    def _exhaust(self, reason: str, *, request_stop: bool = True) -> None:
        """Flip the service permanently disabled and record WHY (exhausted_reason), so an
        operator staring at a stopped or degraded run sees the actual cause instead of a
        generic line. Every AUTO-path exhaustion (the module docstring's list: budget
        reached, a latch tripped, a permanent provider error, GeminiCapacityExhausted, every
        retry attempt failing) reaches here with the default request_stop=True.

        request_stop=False is for an ADVISORY call only (see maybe_opener's advisory
        parameter): a Hinge observe-mode suggestion that a human retypes and sends
        themselves has no automation behind it for a bad response to threaten, so ending the
        whole observe session over it would sacrifice the session's entire purpose (real
        training labels) for something purely cosmetic -- the owner's rule is "stop the
        AUTOMATION", and observe has none. disabled/exhausted_reason are still set exactly
        as for an AUTO exhaustion (a systematically broken model/prompt must still stop
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
