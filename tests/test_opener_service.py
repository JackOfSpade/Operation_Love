"""OpenerService branch logic: disabled / budget / provider failures / stop-vs-continue.

The service is the GLOBAL, budget-aware gate for openers shared by every worker.
It's exercised indirectly elsewhere; this pins its own decision branches with
lightweight fakes (no provider SDK/network).
"""
import math
import threading
from types import SimpleNamespace

import pytest

import operation_love.opener.service as service_mod
from operation_love.opener.opener import (
    INDEX_SPACE_MODEL_ITEMS,
    INDEX_SPACE_PROFILE_PHOTOS,
    ITEM_INDEX_ABSENT,
    GeminiAPIError,
    GeminiCapacityExhausted,
    ItemRequest,
    OpenerAborted,
    OpenerError,
    OpenerParseError,
    REASON_PROMPT_BLOCKED,
    REASON_RESPONSE_BLOCKED,
)
from operation_love.opener.service import OpenerPick, OpenerService


class _Res:
    model = "gemini-x"
    usage = "usage"
    opener = "hey, that hiking photo is great"
    referenced = "hiking"
    # The MODEL ITEM INDEX (ops/OPENER-REDESIGN.md 5.1/5.7): 1-based, over the numbered items
    # the model was shown, and the item to like as well as the item written about. Replaces
    # `referenced_index`, which was 0-based over raw scroll frames -- a different quantity, not
    # a rename, which is why nothing in this file may treat one as the other.
    item_index = 2
    # The model's own free-text words for what its opener is DOING -- pure telemetry, never
    # branched on (see OpenerPick.angle / OpenerResult.angle). Present on this fake so the
    # tests below can pin that it survives all the way through OpenerPick, the persisted
    # `openers` row, and the recent_openers ring buffer rather than being silently dropped by
    # one of the three. Deliberately NOT set on every other fake in this file: the "" default
    # for a result that omits the field is its own pinned contract further down.
    angle = "guessing where the hike was"
    # The model's own description of the ITEM it picked -- carried in BOTH modes (doc 5.7),
    # logged by auto and displayed by observe. Same "present here, absent elsewhere" split as
    # `angle`: the "" default for a result that omits it is pinned separately below.
    item_description = "a photo of her on a hike"


class _Client:
    """exc, if set, is raised on EVERY call -- the original single-shot fake behavior,
    still right for every exception type that maybe_opener() does NOT retry (OpenerError,
    GeminiAPIError, GeminiCapacityExhausted, ...).

    exc_sequence, when given, instead scripts EXACTLY what happens on each successive
    call within a single retry sequence: None means "return a normal success", anything
    else is raised. This is what OpenerParseError-based tests need, since maybe_opener()
    now retries that specific error internally -- a fixed `exc` would just keep failing
    every retry and exhaust the service, which most of these tests are not about.

    retry_hints records the retry_hint every call was made with, so tests can pin the
    "" (first attempt) vs non-empty (a retry, telling the model what was wrong last time)
    contract directly.

    should_stops records the should_stop callable every call was made with, so BUG 1 tests
    can pin that maybe_opener() actually threads its own should_stop argument through to
    the client on every call (see opener.py's GeminiOpener.generate, which uses this same
    parameter to abort mid-cascade).

    skip_models_seen records the skip_models frozenset every call was made with, so item D's
    tests can pin that maybe_opener() accumulates the model named by each OpenerParseError
    (e.model) and passes it through on the NEXT attempt -- see opener.py's GeminiOpener.
    generate, which uses this same parameter to steer its cascade away from a model that
    already failed to parse for this profile.

    opener_texts, when given, scripts the OPENER TEXT of each successive SUCCESSFUL return
    (success #0 gets opener_texts[0], and the final entry repeats forever once the script runs
    out). Without it EVERY success returns the identical _Res.opener, which the entropy guard
    (service.py's _apply_entropy_guard, ops/OPENER-REDESIGN.md 3.6) reads as a leading-n-gram
    collision the moment a second AUTO call succeeds on the same service -- so a test that
    wants two successful AUTO calls WITHOUT the guard firing must make their openings differ,
    and a test about the guard itself must make them collide on purpose. Both directions are
    exercised below.
    """
    def __init__(self, exc=None, exc_sequence=None, opener_texts=None):
        self.exc = exc
        self.exc_sequence = list(exc_sequence) if exc_sequence is not None else None
        self.opener_texts = list(opener_texts) if opener_texts is not None else None
        self.successes = 0
        self.calls = 0
        self.retry_hints = []
        self.should_stops = []
        self.skip_models_seen = []
        self.items = []

    def _success(self):
        res = _Res()
        if self.opener_texts:
            # Clamped rather than popped, so a script that runs out keeps answering (with its
            # last text) instead of silently reverting to _Res.opener and changing which
            # openers collide half way through a test.
            res.opener = self.opener_texts[min(self.successes, len(self.opener_texts) - 1)]
        self.successes += 1
        return res

    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        self.retry_hints.append(retry_hint)
        self.should_stops.append(should_stop)
        self.skip_models_seen.append(skip_models)
        self.items.append(items)
        if self.exc_sequence is not None:
            step = self.exc_sequence.pop(0) if self.exc_sequence else None
            if step is not None:
                raise step
            return self._success()
        if self.exc:
            raise self.exc
        return self._success()


class _Tracker:
    """budget_reached() answers come from a queue so pre-call vs post-call differ."""
    def __init__(self, reached=None):
        self._q = list(reached or [])
        self.recorded = []

    def budget_reached(self):
        return self._q.pop(0) if self._q else False

    def record(self, model, usage):
        self.recorded.append((model, usage))
        return 0.01


class _Store:
    def __init__(self):
        self.spend = []
        self.openers = []
        self.opener_kwargs = []
        self.rejections = []

    def record_spend(self, *a):
        self.spend.append(a)

    def record_opener(self, *a, **kw):
        self.openers.append(a)
        self.opener_kwargs.append(kw)

    def record_opener_rejection(self, *a):
        self.rejections.append(a)


def test_disabled_without_client():
    s = OpenerService(None, _Tracker(), _Store(), "casual")
    assert s.disabled is True
    assert s.maybe_opener("r", "bumble", object()) is None


def test_generates_records_and_stays_enabled():
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")
    out = s.maybe_opener("r", "bumble", object())
    # OpenerPick.index now carries the MODEL ITEM INDEX straight off the result, unshifted:
    # no +1/-1 anywhere between OpenerResult.item_index and the pick (doc 5.3 -- a silently
    # reinterpreted integer is the bug class this contract is written against).
    assert out.text == _Res.opener and out.index == _Res.item_index and c.calls == 1
    assert out.item_description == _Res.item_description
    assert len(st.spend) == 1 and len(st.openers) == 1     # both spend + opener persisted
    # The persisted `openers` row, pinned POSITIONALLY and in full: both stores declare
    # record_opener(run_id, app, model, opener, referenced, angle="", item_description="")
    # with those two trailing, and service.py passes all seven positionally. A vague len()==1
    # here would not notice a telemetry column silently going out empty (or the arguments
    # transposing, which with two same-typed trailing strings is easy and invisible), which is
    # exactly the drift that would take the offline analysis dark without failing anything.
    assert st.openers[0] == ("r", "bumble", "gemini-x", _Res.opener, _Res.referenced,
                             _Res.angle, _Res.item_description)
    assert t.recorded == [("gemini-x", "usage")]
    assert s.disabled is False and s.stop_requested is False


def test_budget_reached_before_call_skips_and_stops():
    c, t, st = _Client(), _Tracker([True]), _Store()
    s = OpenerService(c, t, st, "casual")
    assert s.maybe_opener("r", "bumble", object()) is None
    assert c.calls == 0                                     # never hit the provider
    assert s.disabled is True and s.stop_requested is True


def test_exhaust_records_the_reason_for_the_hub_to_show():
    # WS-opener-reason: _exhaust() used to only print() its cause, leaving nothing an
    # operator-facing status snapshot could surface -- an opener-exhaustion stop rendered
    # identically to a manual Stop click. exhausted_reason is the durable record of it.
    c, t, st = _Client(), _Tracker([True]), _Store()
    s = OpenerService(c, t, st, "casual")
    assert s.exhausted_reason is None                       # nothing has happened yet
    s.maybe_opener("r", "bumble", object())
    assert s.exhausted_reason == "run budget reached"


def test_exhaust_first_reason_wins_when_exhausted_twice():
    # A second, unrelated exhaustion (e.g. another worker's opener call failing right
    # behind the first, now that the service is already disabled) must not clobber the
    # ORIGINAL cause -- that's the one the operator actually needs to see.
    s = OpenerService(_Client(), _Tracker(), _Store(), "casual")
    s._exhaust("run budget reached")
    s._exhaust("a different worker's opener call also failed")
    assert s.exhausted_reason == "run budget reached"
    assert s.disabled is True and s.stop_requested is True


def test_budget_reached_after_call_disables_but_returns_this_opener():
    c, t, st = _Client(), _Tracker([False, True]), _Store()  # ok pre-call, exhausted post-call
    s = OpenerService(c, t, st, "casual")
    out = s.maybe_opener("r", "bumble", object())
    assert out.text == _Res.opener                          # the in-flight opener still returns
    assert s.disabled is True and s.stop_requested is True  # but no more after this


class _NoPricingTracker(_Tracker):
    """record() raises KeyError, like the real CostTracker when the API echoes back
    a model string with no budget.pricing entry."""
    def record(self, model, usage):
        raise KeyError(f"No pricing configured for model {model!r}")


def test_unpriceable_model_disables_but_returns_this_opener():
    c, t, st = _Client(), _NoPricingTracker([False]), _Store()
    s = OpenerService(c, t, st, "casual")
    out = s.maybe_opener("r", "bumble", object())
    assert out.text == _Res.opener                          # already-spent credits aren't wasted
    # cost recorded as None (unknown), NOT a fabricated 0.0 -- the call had a real,
    # nonzero cost that just couldn't be priced; 0.0 would misreport actual spend.
    assert len(st.spend) == 1 and st.spend[0][-1] is None
    assert s.disabled is True and s.stop_requested is True   # but no more openers until pricing is fixed


def test_opener_error_skips_this_swipe_without_disabling_the_service():
    from operation_love.opener.opener import OpenerError

    c, t, st = _Client(exc=OpenerError("Gemini returned no usable opener for this profile")), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")

    assert s.maybe_opener("r", "bumble", object()) is None     # this swipe: no opener
    assert s.disabled is False and s.stop_requested is False   # service stays live...
    assert len(st.spend) == 0 and len(st.openers) == 0         # ...and nothing was billed/recorded

    c.exc = None                                                # next profile: a normal response
    out = s.maybe_opener("r", "bumble", object())
    assert out.text == _Res.opener                             # subsequent calls are unaffected


def test_transient_error_degrades_gracefully():
    """Transient network/timeout errors skip this profile's opener but leave the service
    enabled so subsequent profiles can still try (the run does not halt)."""
    s = OpenerService(_Client(exc=RuntimeError("timeout")), _Tracker([False]), _Store(), "casual")
    result = s.maybe_opener("r", "bumble", object())
    assert result is None                                   # swipe without opener this time
    assert s.disabled is False                              # service stays enabled (not permanent)
    assert s.stop_requested is False                        # run keeps going


def test_all_gemini_models_exhausted_stops_automation(capsys):
    """Exhausting every configured model stops the whole run, and the provider's OWN
    diagnosis is what reaches the operator -- not a generic line written here.

    GeminiOpener builds that message per model and distinguishes "every model is out of
    DAILY quota, so nothing works until the midnight Pacific reset" from "the cascade fell
    through with a transient per-minute cap in it, so restarting shortly may just work".
    Those need opposite operator responses, and this string is what the hub shows as the
    run's stop reason, so the service must forward it verbatim rather than flatten both
    cases into one hardcoded sentence."""
    reason = ("every configured Gemini model has exhausted its per-day free-tier quota "
              "(gemini-3.6-flash, gemini-2.5-flash-lite); free-tier daily quota resets at "
              "midnight Pacific")
    c = _Client(exc=GeminiCapacityExhausted(reason))
    s = OpenerService(c, _Tracker([False]), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True
    assert s.exhausted_reason == reason          # what the hub renders as the stop reason
    output = capsys.readouterr().out
    assert reason in output                       # forwarded verbatim, nothing paraphrased
    assert "stopping all workers" in output


def test_transient_gemini_exhaustion_reason_is_not_reworded_into_a_daily_quota_message(capsys):
    """The companion to the test above: when the cascade fell through on transient
    per-minute throttling, the operator must NOT be told to wait for the daily reset. The
    service adds no wording of its own, so this holds for whatever GeminiOpener reports."""
    reason = ("no configured Gemini model could serve the request: gemini-3.6-flash "
              "(per-minute throttle). At least one of these is a transient per-minute cap "
              "rather than a per-day exhaustion, so restarting in a minute may well succeed")
    s = OpenerService(_Client(exc=GeminiCapacityExhausted(reason)), _Tracker([False]),
                      _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.exhausted_reason == reason
    output = capsys.readouterr().out
    assert "midnight Pacific" not in output
    assert "restarting in a minute" in output


@pytest.mark.parametrize("http_code", [401, 403])
def test_permanent_gemini_errors_disable_service(http_code):
    error = GeminiAPIError(http_code, "ERROR", "provider rejected request")
    s = OpenerService(_Client(exc=error), _Tracker([False]), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True


def test_404_is_no_longer_classified_permanent_the_cascade_owns_it_now():
    """A model id being retired (HTTP 404) used to disable the WHOLE service, on the theory
    that a bad model id fails identically on every remaining profile. That was wrong: a 404
    means only ONE configured model is unavailable, and GeminiOpener.generate() now retires
    just that model and cascades to the next configured one on its own (see opener.py). A
    bare 404 GeminiAPIError should therefore no longer reach _permanent_reason's old
    whole-service-disabling branch at all -- if one somehow does reach maybe_opener()
    directly, it must be treated like any other non-permanent provider error (transient,
    service stays enabled) rather than reintroducing the old behavior."""
    error = GeminiAPIError(404, "NOT_FOUND", "model retired")
    assert service_mod._permanent_reason(error) is None

    s = OpenerService(_Client(exc=error), _Tracker([False]), _Store(), "casual")
    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is False and s.stop_requested is False


def test_gemini_400_requires_consecutive_profile_latch(capsys):
    error = GeminiAPIError(400, "INVALID_ARGUMENT", "bad image or request")
    client = _Client(exc=error)
    s = OpenerService(client, _Tracker(), _Store(), "casual")

    for _ in range(service_mod._BAD_REQUEST_LATCH_THRESHOLD - 1):
        assert s.maybe_opener("r", "hinge", object()) is None
        assert s.disabled is False
    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True
    assert "opener provider rejected" in capsys.readouterr().out


def test_gemini_invalid_api_key_400_latches_immediately_not_after_a_streak(capsys):
    """Google returns an invalid/revoked API key as HTTP 400 INVALID_ARGUMENT (message
    "API key not valid..."), not 401 -- pre-fix this fell into the ordinary bad-request
    streak path and took _BAD_REQUEST_LATCH_THRESHOLD profiles to latch, misreporting a
    dead key as "the provider rejected N requests in a row as malformed". It must instead
    be recognized as PERMANENT on the very first occurrence, with the GEMINI_API_KEY
    message."""
    error = GeminiAPIError(400, "INVALID_ARGUMENT",
                            "API key not valid. Please pass a valid API key.")
    s = OpenerService(_Client(exc=error), _Tracker([False]), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True     # latched on the FIRST call
    assert s.exhausted_reason == "GEMINI_API_KEY is not valid or has been revoked"
    assert "opener provider rejected" not in capsys.readouterr().out   # not the streak path


def test_gemini_ordinary_400_still_requires_the_full_streak_even_with_invalid_key_checking():
    """The invalid-key 400 detection must not swallow every 400 -- an ordinary
    malformed-payload 400 (e.g. a corrupt screenshot) keeps its existing 3-in-a-row
    latch behaviour, same as test_gemini_400_requires_consecutive_profile_latch."""
    error = GeminiAPIError(400, "INVALID_ARGUMENT", "bad image or request")
    client = _Client(exc=error)
    s = OpenerService(client, _Tracker(), _Store(), "casual")

    for _ in range(service_mod._BAD_REQUEST_LATCH_THRESHOLD - 1):
        assert s.maybe_opener("r", "hinge", object()) is None
        assert s.disabled is False
    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True
    assert s.exhausted_reason != "GEMINI_API_KEY is not valid or has been revoked"


def test_gemini_api_key_invalid_status_shape_latches_immediately_too():
    """Google reports a dead/revoked key TWO different ways: the message-based shape covered
    by test_gemini_invalid_api_key_400_latches_immediately_not_after_a_streak above ("API key
    not valid. Please pass a valid API key."), and a second shape that instead carries the
    machine-readable status API_KEY_INVALID with differently-worded prose that does NOT
    contain the phrase "api key not valid" anywhere. _is_invalid_gemini_api_key checks BOTH
    exc.message AND exc.status for either phrase (see its docstring) specifically so this
    second shape is not missed -- nothing else in this suite ever constructs a GeminiAPIError
    with status API_KEY_INVALID and a message lacking "api key not valid", so deleting that
    half of the check (the `or "api_key_invalid" in haystack` branch) would change nothing
    else in the suite while silently letting this real error shape fall through to the
    ordinary 3-strikes malformed-request streak (see test_gemini_400_requires_consecutive_
    profile_latch): two more profiles burned on a dead key, then reported to an operator as
    "the provider rejected 3 requests in a row as malformed" instead of the correct
    diagnosis."""
    error = GeminiAPIError(400, "API_KEY_INVALID",
                            "The provided key could not be authenticated for this project.")
    s = OpenerService(_Client(exc=error), _Tracker([False]), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True     # latched on the FIRST call
    assert s.exhausted_reason == "GEMINI_API_KEY is not valid or has been revoked"


def test_a_success_between_bad_request_400s_resets_the_streak():
    """Companion to test_a_success_between_transient_failures_resets_the_streak below, but
    for the OTHER latch: _consecutive_bad_requests. maybe_opener()'s success branch resets
    it to 0 (the request shape is proven fine) -- remove that reset and this test is exactly
    the mutation that breaks: 400, 400, SUCCESS, 400, 400 would latch the service permanently
    disabled on the fourth call, because the post-success pair of 400s would be miscounted
    as calls 3 and 4 of one unbroken streak instead of a fresh streak of 2.

    Interleaves 400 x(threshold-1), a real SUCCESS, then 400 x(threshold-1) again -- still
    enabled -- and only THEN one more 400 to prove the latch logic itself still works (the
    streak reaching the threshold from a true zero, not that latching is broken entirely)."""
    error = GeminiAPIError(400, "INVALID_ARGUMENT", "bad image or request")
    c = _Client(exc=error)
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    for _ in range(service_mod._BAD_REQUEST_LATCH_THRESHOLD - 1):
        assert s.maybe_opener("r", "hinge", object()) is None
        assert s.disabled is False

    c.exc = None                                     # SUCCESS -- must clear the streak
    out = s.maybe_opener("r", "hinge", object())
    assert out.text == _Res.opener and s.disabled is False

    c.exc = error
    for _ in range(service_mod._BAD_REQUEST_LATCH_THRESHOLD - 1):
        assert s.maybe_opener("r", "hinge", object()) is None
        assert s.disabled is False                  # streak restarted from zero, not resumed
    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True   # latch still works from a real streak


@pytest.mark.parametrize("http_code", [408, 500, 503])
def test_transient_gemini_errors_keep_service_enabled(http_code):
    error = GeminiAPIError(http_code, "UNAVAILABLE", "try again")
    s = OpenerService(_Client(exc=error), _Tracker([False]), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is False and s.stop_requested is False


# ---------------------------------------------------------------------------------------
# E: a streak of consecutive TRANSIENT failures (unclassified exceptions, or per-profile
# OpenerErrors) must eventually escalate rather than degrading silently forever. An audit
# demonstrated 50 consecutive bare RuntimeErrors sailing through with disabled=False,
# stop_requested=False, exhausted_reason=None throughout -- each one a live Hinge profile
# that got a bare like with no opener and nothing on the hub to show it.
# ---------------------------------------------------------------------------------------

def test_two_consecutive_transient_failures_do_not_latch():
    """Below _TRANSIENT_LATCH_THRESHOLD, the service must stay live -- a one-off timeout or
    connection blip is normal and must not kill an otherwise-healthy run."""
    c = _Client(exc=RuntimeError("timeout"))
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    for _ in range(service_mod._TRANSIENT_LATCH_THRESHOLD - 1):
        assert s.maybe_opener("r", "hinge", object()) is None
        assert s.disabled is False and s.stop_requested is False


def test_third_consecutive_transient_failure_latches_and_names_the_streak(capsys):
    """The Nth consecutive transient failure (N = _TRANSIENT_LATCH_THRESHOLD) must stop the
    service AND the run, with a reason naming the streak length and the last error seen --
    not a generic line the operator has to go digging in old logs to explain."""
    c = _Client(exc=RuntimeError("connection reset"))
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    for _ in range(service_mod._TRANSIENT_LATCH_THRESHOLD - 1):
        assert s.maybe_opener("r", "hinge", object()) is None
        assert s.disabled is False        # not latched yet

    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True
    assert str(service_mod._TRANSIENT_LATCH_THRESHOLD) in s.exhausted_reason
    assert "RuntimeError" in s.exhausted_reason
    assert "connection reset" in s.exhausted_reason
    output = capsys.readouterr().out
    assert "consecutive opener calls failed" in output


def test_a_success_between_transient_failures_resets_the_streak():
    """A successful generate() call proves the pipeline is healthy end to end, so it must
    zero the streak -- two more failures afterward must NOT be treated as calls 3 and 4 of
    an unbroken run."""
    c = _Client(exc=RuntimeError("timeout"))
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    for _ in range(service_mod._TRANSIENT_LATCH_THRESHOLD - 1):
        s.maybe_opener("r", "hinge", object())
    assert s.disabled is False

    c.exc = None                                    # next call succeeds
    out = s.maybe_opener("r", "hinge", object())
    assert out.text == _Res.opener and s.disabled is False

    c.exc = RuntimeError("timeout again")
    for _ in range(service_mod._TRANSIENT_LATCH_THRESHOLD - 1):
        assert s.maybe_opener("r", "hinge", object()) is None
        assert s.disabled is False                  # streak restarted from zero, not resumed
    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True


def test_openerparse_error_between_transient_failures_resets_the_streak():
    """OpenerParseError means the call actually reached the API and was billed -- proof the
    transport/auth/request-building are all fine -- so it must reset the streak exactly like
    a full success does, even though the FIRST attempt that hits it also (separately) failed
    to produce a usable opener. maybe_opener() now retries an OpenerParseError itself, so this
    is modeled as one bad response that succeeds on its retry (the realistic case under the
    owner's retry-until-good-or-N-strikes rule): the streak still resets on the very first,
    still-failing attempt (see the reset at the top of the OpenerParseError branch), before the
    retry ever runs, and the overall call still returns a usable opener."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc=RuntimeError("timeout"))
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    for _ in range(service_mod._TRANSIENT_LATCH_THRESHOLD - 1):
        s.maybe_opener("r", "hinge", object())
    assert s.disabled is False

    c.exc = None
    c.exc_sequence = [parse_error]        # one bad response, then a normal success on retry
    out = s.maybe_opener("r", "hinge", object())
    assert out.text == _Res.opener
    assert s.disabled is False

    c.exc_sequence = None
    c.exc = RuntimeError("timeout again")
    for _ in range(service_mod._TRANSIENT_LATCH_THRESHOLD - 1):
        assert s.maybe_opener("r", "hinge", object()) is None
        assert s.disabled is False                  # streak restarted from zero, not resumed
    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True


def test_consecutive_opener_errors_count_toward_the_transient_latch_instead_of_resetting():
    """Deliberate design decision (documented in service.py's _register_transient_failure):
    unlike OpenerParseError, a plain OpenerError does NOT prove the call ever reached the
    provider -- opener.py can raise it before any network call at all (e.g. a corrupt photo
    failing to decode while fitting the request to the inline-size budget). A single bad
    photo is normal and must not escalate alone, but a systemically corrupt capture pipeline
    would surface as a long run of exactly this error, one per profile -- precisely the
    silent-forever degradation this latch exists to catch, so OpenerError COUNTS toward the
    same streak rather than resetting it."""
    corrupt_photo_error = OpenerError("Gemini opener: photo index 0 could not be decoded")
    c = _Client(exc=corrupt_photo_error)
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    for _ in range(service_mod._TRANSIENT_LATCH_THRESHOLD - 1):
        assert s.maybe_opener("r", "hinge", object()) is None
        assert s.disabled is False

    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True
    assert "OpenerError" in s.exhausted_reason


def test_mixing_openererror_and_generic_transient_failures_shares_one_streak():
    """The two failure classes feed the SAME counter -- an operator does not care whether
    the run of failures was all corrupt photos, all timeouts, or a mix; either way it is a
    run of opener calls that never produced a usable opener."""
    c = _Client(exc=OpenerError("bad photo"))
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None   # #1: OpenerError
    assert s.disabled is False
    c.exc = RuntimeError("timeout")
    assert s.maybe_opener("r", "hinge", object()) is None   # #2: generic transient
    assert s.disabled is False
    c.exc = OpenerError("bad photo again")
    assert s.maybe_opener("r", "hinge", object()) is None   # #3: latches
    assert s.disabled is True and s.stop_requested is True


class _FailingStore(_Store):
    def record_spend(self, *a):
        raise RuntimeError("database unavailable")


class _FailingRejectionStore(_Store):
    def record_opener_rejection(self, *a):
        raise RuntimeError("rejections table unavailable")


def test_parse_error_returns_none_and_formats_unknown_cost_safely(capsys):
    """The unpriceable-model guard inside the OpenerParseError retry branch must _exhaust()
    (and therefore stop the run -- unconditional now that on_exhausted no longer exists;
    see service.py's _exhaust) immediately, without waiting for a retry, and must format
    the unrecoverable cost safely even though the store also fails to persist it."""
    parse_error = OpenerParseError("bad JSON", "usage", "unpriced-model")
    s = OpenerService(
        _Client(exc=parse_error), _NoPricingTracker([False, False]), _FailingStore(),
        "casual",
    )

    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True
    output = capsys.readouterr().out
    assert "an unknown amount" in output
    assert "unparseable response" in output


# ---------------------------------------------------------------------------------------
# last_skip_reason: the per-thread, per-call companion to exhausted_reason (first-writer-wins,
# run-lifetime). It exists so a caller that must never send a like with no
# opener (worker.py's _auto_loop) can report the ACTUAL cause of a per-call failure that
# leaves the service otherwise healthy, instead of a generic line. See maybe_opener's and
# last_skip_reason's own docstrings in service.py for the exact contract: set on every
# early-return-None path that does NOT go through _exhaust() (unparseable response,
# OpenerError, a sub-latch 400, a sub-latch transient failure), and cleared on success.
# ---------------------------------------------------------------------------------------

def test_last_skip_reason_starts_unset():
    s = OpenerService(_Client(), _Tracker(), _Store(), "casual")
    assert s.last_skip_reason is None
    assert s.last_skip_allows_commentless_like is False


def test_last_skip_reason_is_isolated_between_concurrent_worker_threads():
    """A worker must read the reason for its own call after the service lock is released.

    The former shared scalar was last-writer-wins across workers, so another app could replace
    it before the first worker published its warning. The public attribute now remains local to
    the calling thread while exhausted_reason stays deliberately global.
    """
    s = OpenerService(_Client(), _Tracker(), _Store(), "casual")
    written = threading.Barrier(3)
    reads = {}

    def worker(name):
        s.last_skip_reason = f"{name} failed"
        written.wait()
        reads[name] = s.last_skip_reason

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("hinge", "bumble")]
    for thread in threads:
        thread.start()
    written.wait()
    for thread in threads:
        thread.join()

    assert reads == {"hinge": "hinge failed", "bumble": "bumble failed"}
    assert s.last_skip_reason is None


def test_last_skip_reason_set_on_unparseable_response():
    """A single OpenerParseError attempt sets last_skip_reason to name the failure -- even
    though (unlike an OpenerError/400/transient failure) a lone OpenerParseError can no
    longer leave maybe_opener() returning with the service still enabled by itself: it is
    retried internally (see maybe_opener), so a fake that keeps failing every attempt (as
    here) runs the retry loop out to exhaustion. last_skip_reason is still set, correctly,
    on that final failing attempt right before exhausted_reason takes over -- this pins
    that it names the actual failure even in the exhausting case, not a generic message."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    s = OpenerService(_Client(exc=parse_error), _Tracker(), _Store(), "casual")
    s.maybe_opener("r", "hinge", object())
    assert s.last_skip_reason is not None and "bad JSON" in s.last_skip_reason
    assert s.disabled is True and s.stop_requested is True   # exhausted after max_attempts


def test_last_skip_reason_set_on_opener_error():
    s = OpenerService(_Client(exc=OpenerError("bad photo")), _Tracker(), _Store(), "casual")
    s.maybe_opener("r", "hinge", object())
    assert s.last_skip_reason is not None and "bad photo" in s.last_skip_reason


def test_last_skip_reason_set_on_a_single_sub_latch_400():
    error = GeminiAPIError(400, "INVALID_ARGUMENT", "bad image or request")
    s = OpenerService(_Client(exc=error), _Tracker(), _Store(), "casual")
    s.maybe_opener("r", "hinge", object())
    assert s.disabled is False                     # below the latch -- still enabled
    assert s.last_skip_reason is not None and "400" in s.last_skip_reason


def test_last_skip_reason_set_on_a_single_sub_latch_transient_failure():
    s = OpenerService(_Client(exc=RuntimeError("timeout")), _Tracker(), _Store(), "casual")
    s.maybe_opener("r", "hinge", object())
    assert s.disabled is False
    assert s.last_skip_reason is not None and "timeout" in s.last_skip_reason


def test_last_skip_reason_cleared_on_a_successful_call():
    c = _Client(exc=RuntimeError("timeout"))
    s = OpenerService(c, _Tracker([False]), _Store(), "casual")
    s.maybe_opener("r", "hinge", object())
    assert s.last_skip_reason is not None           # set by the failure

    c.exc = None                                    # next call succeeds
    out = s.maybe_opener("r", "hinge", object())
    assert out.text == _Res.opener
    assert s.last_skip_reason is None               # cleared -- no stale reason lingers


def test_last_skip_reason_not_set_when_service_is_disabled_from_construction():
    """opener.enabled=false -> OpenerService(client=None, ...) is disabled from the start.
    That short-circuit returns None without ever reaching a per-call failure branch, so it
    must not fabricate a last_skip_reason -- there is nothing per-call to report."""
    s = OpenerService(None, _Tracker(), _Store(), "casual")
    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.last_skip_reason is None


def test_the_latching_call_records_exhausted_reason_not_last_skip_reason():
    """The Nth (latching) consecutive transient failure goes through _exhaust() -- the
    GLOBAL-exhaustion path -- not the per-call last_skip_reason path. A caller that finds
    disabled=True after a call must read exhausted_reason (the global cause), not
    last_skip_reason (which answers a narrower question that no longer applies once the
    service has given up for the rest of the run)."""
    c = _Client(exc=RuntimeError("connection reset"))
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    for _ in range(service_mod._TRANSIENT_LATCH_THRESHOLD - 1):
        s.maybe_opener("r", "hinge", object())
    assert s.disabled is False

    s.maybe_opener("r", "hinge", object())          # latches
    assert s.disabled is True and s.stop_requested is True
    assert s.exhausted_reason is not None
    assert str(service_mod._TRANSIENT_LATCH_THRESHOLD) in s.exhausted_reason


# ---------------------------------------------------------------------------------------
# THE OWNER'S RULE: "I do not want commentless likes. If the response from the AI is bad,
# redo the prompt. I'd rather spend more usage than go with a bad response. If after 5
# attempts it's still a bad response, stop the automation -- that means something is
# wrong." maybe_opener() implements this as a retry loop scoped to OpenerParseError only
# (see the method's own docstring for why every other failure type is NOT retried here).
# ---------------------------------------------------------------------------------------

def test_bad_response_succeeding_on_retry_returns_the_opener_and_bills_both_attempts():
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc_sequence=[parse_error])   # attempt 1 fails, attempt 2 succeeds
    t = _Tracker()
    st = _Store()
    s = OpenerService(c, t, st, "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.text == _Res.opener and c.calls == 2
    # BOTH attempts are real billed calls -- every one of them must reach
    # tracker.record/store.record_spend, not just the one that finally succeeded.
    assert t.recorded == [("gemini-x", "usage"), ("gemini-x", "usage")]
    assert len(st.spend) == 2
    assert len(st.openers) == 1               # only the successful attempt produces an opener
    assert s.disabled is False and s.stop_requested is False


@pytest.mark.parametrize("reason_code", [REASON_PROMPT_BLOCKED, REASON_RESPONSE_BLOCKED])
def test_gemini_safety_block_is_recorded_once_then_skips_only_this_profile(reason_code):
    blocked = OpenerParseError("Gemini withheld content", "usage", "gemini-x",
                               reason_code=reason_code)
    c = _Client(exc=blocked)
    t = _Tracker()
    st = _Store()
    s = OpenerService(c, t, st, "casual")

    assert s.maybe_opener("r", "hinge", object()) is None

    # The provider call is still billable/auditable, but a policy block must never trigger the
    # ordinary malformed-output retry storm or disable an otherwise healthy run.
    assert c.calls == 1
    assert t.recorded == [("gemini-x", "usage")]
    assert len(st.spend) == len(st.rejections) == 1
    assert st.rejections[0][4] == reason_code
    assert s.disabled is False and s.stop_requested is False
    assert s.exhausted_reason is None
    assert "safety policy withheld" in s.last_skip_reason
    assert "not retried" in s.last_skip_reason
    assert "Future profiles remain eligible" in s.last_skip_reason
    assert s.last_skip_allows_commentless_like is True

    c.exc = None
    next_card = s.maybe_opener("r", "hinge", object())

    assert next_card is not None and next_card.text == _Res.opener
    assert s.last_skip_reason is None
    assert s.last_skip_allows_commentless_like is False


def test_advisory_gemini_safety_block_leaves_observation_and_future_cards_live():
    blocked = OpenerParseError("Gemini withheld content", "usage", "gemini-x",
                               reason_code=REASON_PROMPT_BLOCKED)
    c = _Client(exc_sequence=[blocked, None])
    st = _Store()
    s = OpenerService(c, _Tracker(), st, "casual")

    assert s.maybe_opener("r", "hinge", object(), advisory=True) is None

    assert c.calls == 1
    assert s.disabled is False and s.stop_requested is False
    assert s.exhausted_reason is None
    # Advisory drafts are still intentionally not durable opener/rejection rows, but their
    # spend and the operator-facing per-profile reason are retained.
    assert len(st.spend) == 1 and st.rejections == []
    assert "safety policy withheld" in s.last_skip_reason
    assert s.last_skip_allows_commentless_like is False

    next_card = s.maybe_opener("r", "hinge", object(), advisory=True)

    assert next_card is not None and next_card.text == _Res.opener
    assert c.calls == 2
    assert len(st.spend) == 2
    assert s.disabled is False and s.stop_requested is False
    assert s.last_skip_reason is None
    assert s.last_skip_allows_commentless_like is False


# ---------------------------------------------------------------------------------------
# Durable rejection recording (opener_rejections): before this, a rejected attempt was
# printed to the console and then lost forever -- BigQuery only ever recorded SUCCEEDED
# openers. Every OpenerParseError attempt must now reach self.store.record_opener_rejection
# BEFORE the retry/exhaust decision, including the final attempt that stops the run, and a
# store failure on that call must never take down opener generation (same contract as
# record_spend just above it in service.py).
# ---------------------------------------------------------------------------------------

def test_rejection_recorded_with_reason_code_and_raw_opener_before_a_successful_retry():
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x",
                                   reason_code="bad_json", raw_opener="{not valid json")
    c = _Client(exc_sequence=[parse_error])   # attempt 1 fails, attempt 2 succeeds
    st = _Store()
    s = OpenerService(c, _Tracker(), st, "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.text == _Res.opener
    assert len(st.rejections) == 1            # only the failing attempt, not the success
    run_id, app, model, attempt, reason_code, reason, raw_opener = st.rejections[0]
    assert (run_id, app, model, attempt) == ("r", "hinge", "gemini-x", 1)
    assert reason_code == "bad_json" and raw_opener == "{not valid json"
    assert "bad JSON" in reason


def test_rejection_recorded_for_every_attempt_including_the_final_exhausting_one():
    parse_error = OpenerParseError("still bad JSON", "usage", "gemini-x",
                                   reason_code="scaffolding", raw_opener="Here's the opener: hi")
    c = _Client(exc=parse_error)              # every attempt fails, forever
    st = _Store()
    s = OpenerService(c, _Tracker(), st, "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True   # exhausted via _exhaust()

    assert len(st.rejections) == s.max_attempts == 5
    attempts = [row[3] for row in st.rejections]
    assert attempts == [1, 2, 3, 4, 5]                        # including the final one
    assert all(row[4] == "scaffolding" for row in st.rejections)


def test_rejection_store_failure_does_not_break_a_successful_retry():
    """A store that raises on record_opener_rejection must not prevent the retry loop from
    still returning a usable opener -- matching record_spend's own guard just above it."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc_sequence=[parse_error])
    s = OpenerService(c, _Tracker(), _FailingRejectionStore(), "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.text == _Res.opener


def test_rejection_store_failure_does_not_break_exhaustion():
    """Same guard, in the exhausting case: the run must still stop for the real reason (5
    consecutive bad responses) even though every one of those attempts also failed to
    persist to the (broken) rejections store."""
    parse_error = OpenerParseError("still bad JSON", "usage", "gemini-x")
    c = _Client(exc=parse_error)
    s = OpenerService(c, _Tracker(), _FailingRejectionStore(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.disabled is True and s.stop_requested is True
    assert "still bad JSON" in s.exhausted_reason


def test_recent_rejections_snapshot_mirrors_the_stored_rows_in_memory():
    """recent_rejections_snapshot() is the in-memory ring buffer a bug report reads without a
    store round-trip (see service.py's recent_rejections docstring in __init__) -- it must be
    populated from the exact same OpenerParseError data as the durable store row, independent
    of the store call's own success or failure."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x",
                                   reason_code="bad_json", raw_opener="{oops")
    c = _Client(exc_sequence=[parse_error])
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    s.maybe_opener("r", "hinge", object())

    snapshot = s.recent_rejections_snapshot()
    assert len(snapshot) == 1
    entry = snapshot[0]
    assert entry["app"] == "hinge" and entry["model"] == "gemini-x"
    assert entry["attempt"] == 1
    assert entry["reason_code"] == "bad_json" and entry["raw_opener"] == "{oops"
    assert "bad JSON" in entry["reason"]


def test_five_consecutive_bad_responses_exhaust_and_name_the_attempt_count():
    parse_error = OpenerParseError("still bad JSON", "usage", "gemini-x")
    c = _Client(exc=parse_error)              # every attempt fails, forever
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert c.calls == s.max_attempts == 5
    assert s.disabled is True and s.stop_requested is True
    assert str(s.max_attempts) in s.exhausted_reason
    assert "still bad JSON" in s.exhausted_reason


def test_max_attempts_is_configurable_and_honoured_by_the_retry_loop():
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc=parse_error)
    s = OpenerService(c, _Tracker(), _Store(), "casual", max_attempts=2)

    assert s.maybe_opener("r", "hinge", object()) is None
    assert c.calls == 2
    assert s.disabled is True and s.stop_requested is True
    assert "2" in s.exhausted_reason


def test_retry_hint_is_empty_on_the_first_attempt_and_non_empty_after():
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc_sequence=[parse_error])   # attempt 1 fails, attempt 2 succeeds
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.text == _Res.opener
    assert c.retry_hints[0] == ""                      # first attempt: no prior failure to report
    assert c.retry_hints[1] != ""                       # retry: told what was wrong last time
    assert "bad JSON" in c.retry_hints[1]               # names the actual prior failure


def test_opener_error_is_never_retried():
    """Unlike OpenerParseError, a plain OpenerError is almost always deterministic (e.g. a
    corrupt photo that fails to decode) -- retrying would just resend the identical bad
    payload and fail identically, so it is not retried at all: exactly one call."""
    c = _Client(exc=OpenerError("Gemini opener: photo index 0 could not be decoded"))
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert c.calls == 1
    assert s.disabled is False and s.stop_requested is False


def test_http_level_failure_is_never_retried_by_this_method():
    """An HTTP-level failure is not retried HERE at all -- GeminiOpener.generate() already
    cascades a 400/429/404/5xx across every configured model internally before it ever
    raises to this method (see opener.py's GeminiOpener docstring), so maybe_opener() must
    not loop again on top of that: exactly one call reaches this service per profile."""
    error = GeminiAPIError(500, "UNAVAILABLE", "try again")
    c = _Client(exc=error)
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert c.calls == 1
    assert s.disabled is False and s.stop_requested is False


def test_budget_reached_mid_retry_stops_retrying_and_reports_budget_reason():
    """The owner's retry-until-good-or-5-strikes rule never overrides the run budget: a cap
    crossed by an in-progress retry sequence's own spend must stop further retries right
    away, with the budget (not the retry count) as the reported reason."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc=parse_error)              # would keep failing every attempt
    t = _Tracker([False, True])               # pre-call OK, then over budget right after attempt 1
    s = OpenerService(c, t, _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert c.calls == 1                       # stopped before a second attempt
    assert s.disabled is True and s.stop_requested is True
    assert s.exhausted_reason == "run budget reached"


# ---------------------------------------------------------------------------------------
# should_stop -- BUG 1 (adversarial audit): nothing on this path ever consulted a stop
# signal, so a Stop click during a live retry/cascade sequence was silently ignored for as
# long as max_attempts * len(models) * request_timeout_s (~52 minutes against the shipped
# config) while THIS service's own lock stayed held the whole time, blocking every other
# worker sharing it. Threaded into every client.generate() call and also checked directly
# between retry attempts (a simple/fake client cannot be relied on to implement its own
# should_stop cascade the way GeminiOpener does).
# ---------------------------------------------------------------------------------------

def test_should_stop_is_threaded_into_every_client_generate_call():
    """maybe_opener() must forward its own should_stop argument to the client on every
    call -- this is what lets GeminiOpener abort BETWEEN models mid-cascade (see opener.py's
    should_stop docstring); a fake/simple client that ignores it is unaffected either way."""
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")
    marker = lambda: False    # noqa: E731 -- identity sentinel, not meant to be reused
    s.maybe_opener("r", "hinge", object(), should_stop=marker)
    assert c.should_stops == [marker]


def test_should_stop_checked_between_retry_attempts_stops_without_poisoning_health():
    """A stop discovered BETWEEN retry attempts (independent of anything the client itself
    does) must end the call immediately: exactly 1 API call is made (attempt 1, which fails
    with a normally-retryable OpenerParseError), stop_requested stays False, disabled stays
    False, and neither latch counter moves -- a deliberate shutdown is not evidence the
    provider or this service is unhealthy, and must not be reported or counted like one."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc_sequence=[parse_error])   # attempt 1 fails (retryable); attempt 2 would succeed
    s = OpenerService(c, _Tracker(), _Store(), "casual")
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        # False for the check before attempt 1 (it must still run -- the retry sequence was
        # already in flight when Stop was clicked), True from then on: simulating a stop
        # discovered right after attempt 1's (unusable) response came back.
        return calls["n"] > 1

    assert s.maybe_opener("r", "hinge", object(), should_stop=should_stop) is None
    assert c.calls == 1                                     # attempt 2 never happened
    assert s.disabled is False and s.stop_requested is False
    assert s.exhausted_reason is None
    assert s._consecutive_bad_requests == 0
    assert s._consecutive_transient_failures == 0
    assert s.last_skip_reason is not None and "stopping" in s.last_skip_reason.lower()


def test_opener_aborted_from_the_client_does_not_poison_exhausted_reason():
    """If the CLIENT itself raises OpenerAborted (e.g. GeminiOpener catching a stop signal
    mid-cascade, after already issuing one model's request this attempt), the service must
    treat it exactly like the loop-top should_stop check above -- not like an ordinary
    OpenerError: no retry, no _exhaust(), no latch movement, and exhausted_reason must stay
    unset. An operator reading the hub after clicking Stop must never see a fabricated
    'provider failed' story in its place."""
    aborted = OpenerAborted(
        "Opener cascade aborted before requesting 'gemini-second': the run is stopping "
        "(should_stop signaled), not a provider failure")
    c = _Client(exc=aborted)
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    # should_stop=lambda: False here so the loop-top check does NOT itself short-circuit --
    # this test is specifically about the client raising OpenerAborted mid-call, not about
    # this service's own between-attempts check (covered by the test above).
    assert s.maybe_opener("r", "hinge", object(), should_stop=lambda: False) is None
    assert c.calls == 1                                     # not retried
    assert s.disabled is False and s.stop_requested is False
    assert s.exhausted_reason is None
    assert s._consecutive_bad_requests == 0
    assert s._consecutive_transient_failures == 0
    assert s.last_skip_reason is not None and "stopping" in s.last_skip_reason.lower()


def test_should_stop_defaults_to_none_and_never_short_circuits_when_omitted():
    """Sanity companion: every pre-existing call site in this file omits should_stop, and
    that must keep behaving exactly as before -- should_stop=None is the "not stopping"
    case, checked with an `is not None` guard rather than calling a missing callable."""
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")
    out = s.maybe_opener("r", "hinge", object())
    assert out.text == _Res.opener and c.calls == 1
    assert c.should_stops == [None]


# ---------------------------------------------------------------------------------------
# BUG 2 (adversarial audit): OpenerService(max_attempts=0) made range(1, 1) empty, so
# maybe_opener()'s retry loop body never ran at all: 0 API calls, disabled stayed False,
# stop_requested stayed False, no reason was ever recorded -- reporting perfectly healthy
# while producing zero openers, forever. config.validate() already keeps this out of the
# shipped app, but OpenerService is a public class constructed directly all over the tests
# (and by any future direct call site), so the invariant must hold here too.
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad_max_attempts", [0, -1, 16, True, "5", 2.5], ids=[
    "zero", "negative", "over_ceiling", "bool_true", "string", "float"])
def test_max_attempts_must_be_a_positive_int(bad_max_attempts):
    """bool is rejected explicitly even though it IS an int subclass in Python (True == 1):
    silently coercing max_attempts=True into 1 would be a confusing accident, not a real
    configuration choice."""
    with pytest.raises(ValueError, match="max_attempts"):
        OpenerService(_Client(), _Tracker(), _Store(), "casual", max_attempts=bad_max_attempts)


@pytest.mark.parametrize("bad_attempts", [0, -1, 16, True, "3", 2.5])
def test_advisory_max_attempts_must_be_a_bounded_positive_int(bad_attempts):
    with pytest.raises(ValueError, match="advisory_max_attempts"):
        OpenerService(
            _Client(), _Tracker(), _Store(), "casual",
            max_attempts=15, advisory_max_attempts=bad_attempts,
        )


@pytest.mark.parametrize("bad_deadline", [
    0,
    -1,
    301,
    True,
    "60",
    math.nan,
    math.inf,
    -math.inf,
    pytest.param(10 ** 10_000, id="huge_int"),
])
def test_advisory_deadline_must_be_finite_positive_and_bounded(bad_deadline):
    with pytest.raises(ValueError, match="advisory_deadline_s"):
        OpenerService(
            _Client(), _Tracker(), _Store(), "casual", advisory_deadline_s=bad_deadline,
        )


@pytest.mark.parametrize("bad_style", [None, 1, True, ["casual"]])
def test_style_must_be_a_string(bad_style):
    with pytest.raises(ValueError, match="style"):
        OpenerService(_Client(), _Tracker(), _Store(), bad_style)


# ---------------------------------------------------------------------------------------
# A: advisory=True (Hinge's observe-mode pre-action suggestion) -- a display-only failure
# must never end the observe session. maybe_opener(advisory=True) uses its shorter, time-bounded
# retry policy and routes exhaustion through _exhaust(request_stop=False):
# disabled/exhausted_reason are still set (spend stays protected), but stop_requested is left
# alone. The default (advisory=False, every existing test in this file) must be completely
# unaffected -- none of those tests pass advisory at all, so they pin that on their own.
# ---------------------------------------------------------------------------------------

def test_advisory_uses_its_shorter_retry_budget_not_auto_max_attempts():
    """Observe retries transiently bad model output, but not for AUTO's full budget."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc=parse_error)                # would keep failing every attempt forever
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object(), advisory=True) is None
    assert c.calls == s.advisory_max_attempts == 3       # not the AUTO default of 5
    assert s.disabled is True                    # spend still protected
    assert s.stop_requested is False              # but the run itself was never asked to stop


def test_advisory_bad_response_can_succeed_on_retry():
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc_sequence=[parse_error])
    s = OpenerService(c, _Tracker(), _Store(), "casual",
                      max_attempts=5, advisory_max_attempts=3)

    out = s.maybe_opener("r", "hinge", object(), advisory=True)

    assert out.text == _Res.opener
    assert c.calls == 2
    assert c.retry_hints[0] == "" and "bad JSON" in c.retry_hints[1]
    assert s.disabled is False and s.stop_requested is False


def test_advisory_deadline_stops_retries_without_disabling_future_profiles(monkeypatch):
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc=parse_error)
    ticks = iter((100.0, 161.0))
    monkeypatch.setattr(service_mod.time, "monotonic", lambda: next(ticks))
    s = OpenerService(c, _Tracker(), _Store(), "casual",
                      advisory_max_attempts=3, advisory_deadline_s=60)

    assert s.maybe_opener("r", "hinge", object(), advisory=True) is None

    assert c.calls == 1
    assert s.disabled is False and s.stop_requested is False
    assert "deadline" in s.last_skip_reason


def test_advisory_deadline_never_prevents_the_first_attempt(monkeypatch):
    c = _Client()
    ticks = iter((100.0, 10_000.0))
    monkeypatch.setattr(service_mod.time, "monotonic", lambda: next(ticks))
    s = OpenerService(c, _Tracker(), _Store(), "casual",
                      advisory_max_attempts=3, advisory_deadline_s=1)

    assert s.maybe_opener("r", "hinge", object(), advisory=True).text == _Res.opener
    assert c.calls == 1


def test_advisory_exhaustion_disables_but_never_requests_stop(capsys):
    """The owner's rule is 'stop the AUTOMATION' -- in observe mode surfacing an advisory
    suggestion, there is no automation for a bad response to threaten (the human decides and
    sends for themselves either way), so exhausting the retry budget must degrade quietly
    rather than asking every worker sharing this service to halt."""
    parse_error = OpenerParseError("still bad JSON", "usage", "gemini-x")
    c = _Client(exc=parse_error)
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object(), advisory=True) is None
    assert s.disabled is True
    assert s.exhausted_reason is not None and "still bad JSON" in s.exhausted_reason
    assert s.stop_requested is False
    output = capsys.readouterr().out
    assert "stopping all workers" not in output
    assert "advisory suggestion only" in output


def test_advisory_budget_reached_before_call_disables_but_does_not_stop():
    c, t, st = _Client(), _Tracker([True]), _Store()   # budget already reached before this call
    s = OpenerService(c, t, st, "casual")

    assert s.maybe_opener("r", "hinge", object(), advisory=True) is None
    assert c.calls == 0
    assert s.disabled is True and s.exhausted_reason == "run budget reached"
    assert s.stop_requested is False


def test_advisory_bad_request_streak_still_latches_but_does_not_stop():
    """The 400 streak latch (item-for-item the same mechanism AUTO uses) must still trip
    after _BAD_REQUEST_LATCH_THRESHOLD consecutive advisory calls -- spend protection is not
    optional just because the call is advisory -- but must not ask the run to stop."""
    error = GeminiAPIError(400, "INVALID_ARGUMENT", "bad image or request")
    c = _Client(exc=error)
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    for _ in range(service_mod._BAD_REQUEST_LATCH_THRESHOLD - 1):
        assert s.maybe_opener("r", "hinge", object(), advisory=True) is None
        assert s.disabled is False
    assert s.maybe_opener("r", "hinge", object(), advisory=True) is None
    assert s.disabled is True
    assert s.stop_requested is False


def test_advisory_transient_streak_still_latches_but_does_not_stop():
    c = _Client(exc=RuntimeError("connection reset"))
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    for _ in range(service_mod._TRANSIENT_LATCH_THRESHOLD - 1):
        assert s.maybe_opener("r", "hinge", object(), advisory=True) is None
        assert s.disabled is False
    assert s.maybe_opener("r", "hinge", object(), advisory=True) is None
    assert s.disabled is True
    assert s.stop_requested is False


def test_advisory_success_is_unaffected():
    """The happy path must work identically under advisory -- only the FAILURE handling
    changes."""
    s = OpenerService(_Client(), _Tracker([False, False]), _Store(), "casual")
    out = s.maybe_opener("r", "hinge", object(), advisory=True)
    assert out.text == _Res.opener
    assert s.disabled is False and s.stop_requested is False


def test_advisory_draft_keeps_billing_but_writes_no_opener_or_rejection_rows():
    """A generated Observe suggestion can be abandoned without either a Pass or Like.  Its
    provider call remains durable billing telemetry, but must leave no durable or reportable
    profile/opener accounting behind.  AUTO retains the ordinary durable path."""
    success_store = _Store()
    success = OpenerService(_Client(), _Tracker([False]), success_store, "casual")
    assert success.maybe_opener("r", "hinge", object(), advisory=True) is not None
    assert len(success_store.spend) == 1
    assert success_store.openers == []
    assert success.recent_openers_snapshot() == []

    rejection_store = _Store()
    rejected = OpenerService(
        _Client(exc=OpenerParseError("bad JSON", "usage", "gemini-x")),
        _Tracker([False]), rejection_store, "casual")
    assert rejected.maybe_opener("r", "hinge", object(), advisory=True) is None
    assert len(rejection_store.spend) == rejected.advisory_max_attempts == 3
    assert rejection_store.openers == []
    assert rejection_store.rejections == []
    assert rejected.recent_rejections_snapshot() == []


def test_confirmed_like_commits_staged_advisory_opener_exactly_once():
    """The durable opener row is tied to the confirmed Like boundary, not generation.  A
    repeated commit is harmless and cannot duplicate an opener record."""
    store = _Store()
    service = OpenerService(_Client(), _Tracker([False]), store, "casual")

    pick = service.maybe_opener("run", "hinge", object(), advisory=True)
    assert pick is not None
    assert store.openers == []
    assert service.recent_openers_snapshot() == []

    assert service.commit_advisory_opener(pick) is True
    assert len(store.openers) == 1
    assert len(service.recent_openers_snapshot()) == 1
    assert service.recent_openers_snapshot()[0]["advisory"] is True
    assert service.commit_advisory_opener(pick) is False
    assert len(store.openers) == 1


def test_committed_opener_carries_exact_landed_action_lineage():
    store = _Store()
    service = OpenerService(_Client(), _Tracker([False]), store, "casual")
    pick = service.maybe_opener("run", "hinge", object(), advisory=True)

    assert service.commit_advisory_opener(
        pick, profile_id="profile-opaque", decision="like", decision_source="manual",
        decision_created_at=123.0) is True
    assert store.opener_kwargs == [{
        "profile_id": "profile-opaque", "decision": "like", "decision_source": "manual",
        "decision_created_at": 123.0, "model_item_index": 2,
    }]


def test_staged_auto_opener_is_not_committed_until_the_landed_like_boundary():
    """AUTO passes stage=True; a generated string alone is never an acted-on opener row."""
    store = _Store()
    service = OpenerService(_Client(), _Tracker([False]), store, "casual")
    pick = service.maybe_opener("run", "hinge", object(), stage=True)
    assert pick is not None
    assert len(store.spend) == 1
    assert store.openers == [] and service.recent_openers_snapshot() == []
    assert service.commit_opener(pick) is True
    assert len(store.openers) == 1 and len(service.recent_openers_snapshot()) == 1


def test_advisory_default_is_false_so_every_existing_call_site_is_unaffected():
    """Sanity pin: advisory defaults to False, so every pre-existing call in this file (and
    every AUTO-mode call site in worker.py) keeps today's max_attempts-retries-then-stops
    behavior exactly as it was before advisory existed."""
    parse_error = OpenerParseError("still bad JSON", "usage", "gemini-x")
    c = _Client(exc=parse_error)
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert c.calls == s.max_attempts == 5
    assert s.disabled is True and s.stop_requested is True


# ---------------------------------------------------------------------------------------
# D: a parse-failure retry must not re-hit the model that just failed. OpenerService
# accumulates the failing model (OpenerParseError.model) for the CURRENT profile only and
# passes it as skip_models on the next attempt within the SAME maybe_opener() call -- see
# opener.py's GeminiOpener.generate() for how the cascade actually honours the set.
# ---------------------------------------------------------------------------------------

def test_first_attempt_always_sees_an_empty_skip_models():
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")
    s.maybe_opener("r", "hinge", object())
    assert c.skip_models_seen == [frozenset()]


def test_retry_skips_the_model_that_just_failed_to_parse():
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc_sequence=[parse_error])   # attempt 1 fails (model gemini-x), attempt 2 succeeds
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.text == _Res.opener and c.calls == 2
    assert c.skip_models_seen[0] == frozenset()             # attempt 1: nothing failed yet
    assert c.skip_models_seen[1] == frozenset({"gemini-x"})  # attempt 2: steer away from gemini-x


def test_skip_models_accumulates_across_multiple_failed_attempts():
    """A THIRD attempt must skip BOTH models that already failed for this profile, not just
    the most recent one."""
    fail_x = OpenerParseError("bad JSON from x", "usage", "gemini-x")
    fail_y = OpenerParseError("bad JSON from y", "usage", "gemini-y")
    c = _Client(exc_sequence=[fail_x, fail_y])   # attempts 1, 2 fail; attempt 3 succeeds
    s = OpenerService(c, _Tracker(), _Store(), "casual", max_attempts=5)

    out = s.maybe_opener("r", "hinge", object())

    assert out.text == _Res.opener and c.calls == 3
    assert c.skip_models_seen[0] == frozenset()
    assert c.skip_models_seen[1] == frozenset({"gemini-x"})
    assert c.skip_models_seen[2] == frozenset({"gemini-x", "gemini-y"})


def test_skip_models_never_leaks_across_profiles():
    """Reset per profile: a SECOND, separate maybe_opener() call (a new profile) must start
    with an empty skip set again, even though the FIRST profile accumulated a failing model.

    The two profiles are given DIFFERENT opener texts on purpose. This test is about
    skip_models only, and identical text would additionally trip the entropy guard on profile
    2 (a leading-n-gram collision with profile 1's opener), whose budget-exempt extra draw is
    a real, deliberate second entry in skip_models_seen -- see the entropy-guard section at
    the bottom of this file, which pins that draw's own skip_models directly."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc_sequence=[parse_error],
                opener_texts=["hey, that hiking photo is great",
                              "i bet that lake was colder than it looks"])
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    s.maybe_opener("r", "hinge", object())                    # profile 1
    assert c.skip_models_seen == [frozenset(), frozenset({"gemini-x"})]

    c.skip_models_seen = []                                   # isolate profile 2's own calls
    s.maybe_opener("r", "hinge", object())                    # profile 2 -- fresh call, no exc queued
    assert c.skip_models_seen == [frozenset()]                 # NOT frozenset({"gemini-x"})


def test_advisory_first_attempt_passes_an_empty_skip_models():
    """Advisory attempt 1 starts with no failed model -- skip_models must be
    empty (nothing has failed yet within THIS call) exactly like a first AUTO attempt."""
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")
    s.maybe_opener("r", "hinge", object(), advisory=True)
    assert c.skip_models_seen == [frozenset()]


# ---------------------------------------------------------------------------------------
# Retired anchor request shape: passing an anchor must fail before a client request is made.
# ---------------------------------------------------------------------------------------

def test_retired_anchor_argument_is_not_accepted_by_the_service():
    anchor_bytes = b"...png bytes..."
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")

    with pytest.raises(TypeError):
        s.maybe_opener("r", "hinge", object(), anchor=anchor_bytes)
    assert c.calls == 0


@pytest.mark.skip(reason="retired anchor request shape")
def test_same_anchor_is_passed_again_on_every_retry_attempt_after_a_parse_error():
    """A malformed first response is retried (see maybe_opener's OpenerParseError branch),
    but retrying is about fixing the TEXT, not about re-deciding what the message is about --
    what the message attaches to (the anchor) does not change just because the previous
    attempt's opener text was rejected. If a retry ever passed a different anchor, or None,
    the corrected opener could end up describing a different photo than the one it will
    actually be displayed under."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    anchor_bytes = b"anchor-frame-bytes"
    c = _Client(exc_sequence=[parse_error, parse_error])   # attempts 1, 2 fail; attempt 3 succeeds
    s = OpenerService(c, _Tracker(), _Store(), "casual", max_attempts=5)

    out = s.maybe_opener("r", "hinge", object(), anchor=anchor_bytes)

    assert out.text == _Res.opener
    assert c.calls == 3
    assert c.anchors == [anchor_bytes, anchor_bytes, anchor_bytes]


@pytest.mark.skip(reason="retired anchor request shape")
def test_omitting_anchor_yields_none_at_the_client():
    """Every pre-existing call site (and every test above this section) omits anchor, so it
    must default to None at the client, not some other sentinel -- an OpenerClient
    implementation that branches on `anchor is None` needs that exact contract to keep
    behaving as an ordinary, unanchored request."""
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")

    s.maybe_opener("r", "hinge", object())

    assert c.anchors == [None]


def test_item_request_is_forwarded_to_the_client_on_every_attempt():
    """Doc 5.2/5.7's crop shape reaches the model through the same unconditional forwarding as
    the anchor, and for a sharper reason: a service that quietly dropped it would send the raw
    scroll frames instead, and the model's item number would then count frames while every
    consumer downstream reads it as an item number. Reused verbatim across retries -- what the
    model is LOOKING at does not change because the previous attempt's wording was rejected."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    items = ItemRequest(name="Sarah", items=[b"crop-1", b"crop-2"], context=[b"vitals"])
    c = _Client(exc_sequence=[parse_error])          # attempt 1 fails, attempt 2 succeeds
    s = OpenerService(c, _Tracker(), _Store(), "casual", max_attempts=5)

    out = s.maybe_opener("r", "hinge", object(), items=items)

    assert out.text == _Res.opener
    assert c.items == [items, items]


def test_omitting_the_item_request_yields_none_at_the_client():
    """Every pre-existing call site omits it, so it must default to None at the client -- an
    OpenerClient that branches on `items is None` needs that exact contract to keep behaving as
    an ordinary frame-shaped request."""
    c = _Client()
    s = OpenerService(c, _Tracker([False, False]), _Store(), "casual")

    s.maybe_opener("r", "hinge", object())

    assert c.items == [None]


def test_the_entropy_regeneration_reuses_the_same_item_request():
    """The extra draw is the SAME request with a different retry_hint (doc 3.6): it is about the
    opening WORDS and must change nothing else. Dropping the crops here would silently switch the
    regeneration to the frame shape, so the two draws would be answering about different images
    in different index spaces and whichever one survived would carry the other's numbering."""
    items = ItemRequest(name="Sarah", items=[b"crop-1"])
    c = _Client(opener_texts=["Same words here about one thing.",
                              "Different words entirely about it."])
    s = OpenerService(c, _Tracker(), _Store(), "casual")
    s.maybe_opener("r", "hinge", object(), items=items)      # seeds the leading n-gram
    c.opener_texts = ["Same words here about something else.",
                      "Different words entirely about that."]
    c.successes = 0

    s.maybe_opener("r", "hinge", object(), items=items)

    assert c.calls == 3, "one first draw, one seeding call before it, and one regeneration"
    assert c.items == [items, items, items]


def test_referenced_is_populated_from_the_client_result():
    """OpenerPick.referenced echoes the client's own OpenerResult.referenced verbatim -- it's
    what lets the hub show an operator WHICH profile detail the model believed its opener was
    about, so a mismatch against the anchored item becomes visible instead of silent."""
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.referenced == _Res.referenced


def test_referenced_defaults_to_empty_string_when_the_client_result_has_no_such_attribute():
    """service.py reads referenced via getattr(result, "referenced", "") specifically so a
    minimal/older OpenerClient result that doesn't populate the field doesn't blow up with an
    AttributeError -- it degrades to an empty string instead."""
    class _MinimalClient:
        def generate(self, profile, style, retry_hint="", *, anchor=None, items=None,
                 should_stop=None,
                     skip_models=frozenset()):
            return SimpleNamespace(model="gemini-x", usage="usage",
                                   opener="hey there", item_index=1)

    s = OpenerService(_MinimalClient(), _Tracker([False, False]), _Store(), "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.text == "hey there"
    assert out.referenced == ""


def test_angle_is_populated_from_the_client_result_and_persisted():
    """OpenerPick.angle echoes the client's own OpenerResult.angle verbatim, and the same
    value is written as the 6th positional argument of store.record_opener. It is PURE
    TELEMETRY -- nothing in the service, the worker, or any driver branches on it (see
    OpenerPick.angle's docstring and ops/OPENER-REDESIGN.md 3.5, which is also why it is
    deliberately free text and not an enum) -- so nothing else in this suite would notice it
    being dropped somewhere between the client and the `openers` table."""
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.angle == _Res.angle
    assert st.openers[0][5] == _Res.angle


def test_item_description_is_populated_from_the_client_result_and_persisted():
    """ops/OPENER-REDESIGN.md 5.7: `item_description` rides the same telemetry path as `angle`
    -- verbatim onto OpenerPick, 7th positional argument of store.record_opener, and into the
    ring buffer -- and it is written in BOTH modes. It must never collapse into `referenced`:
    that column is the DETAIL the opener reacts to and is what the redundancy monitor (3.7)
    compares the opener against, while this one says what the ITEM is."""
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.item_description == _Res.item_description
    assert st.openers[0][6] == _Res.item_description
    assert st.openers[0][4] == _Res.referenced           # still its own separate column
    assert s.recent_openers_snapshot()[0]["item_description"] == _Res.item_description


def test_advisory_item_description_is_transient_until_a_real_decision():
    """Observe must generate the same item-aware request as AUTO, but a suggestion by itself
    is not a dating decision.  It therefore returns the complete pick for the live UI while
    leaving no durable opener or diagnostic profile trail behind; billing remains separate."""
    c, t, st = _Client(), _Tracker([False, False, False, False]), _Store()
    s = OpenerService(c, t, st, "casual")

    auto = s.maybe_opener("r", "hinge", object())
    advisory = s.maybe_opener("r", "hinge", object(), advisory=True)

    assert advisory.item_description == auto.item_description == _Res.item_description
    assert [row[6] for row in st.openers] == [_Res.item_description]
    # The advisory draft collides with AUTO's leading n-gram, so its one entropy redraw is
    # a second real provider call; billing records all three calls without recording the draft.
    assert len(st.spend) == 3
    assert [e["item_description"] for e in s.recent_openers_snapshot()] == [_Res.item_description]


def test_angle_defaults_to_empty_string_when_the_client_result_has_no_such_attribute():
    """Companion to the `referenced` degradation test above, for the newer fields: service.py
    reads angle via getattr(result, "angle", "") for the same reason (a minimal/older
    OpenerClient result need not populate every field), and the store still gets 6th and 7th
    positional arguments -- empty strings, never missing arguments and never None, since both
    stores declare the columns as STRING/TEXT."""
    class _AngielessClient:
        def generate(self, profile, style, retry_hint="", *, anchor=None, items=None,
                 should_stop=None,
                     skip_models=frozenset()):
            return SimpleNamespace(model="gemini-x", usage="usage", opener="hey there",
                                   referenced="the lake", item_index=1)

    st = _Store()
    s = OpenerService(_AngielessClient(), _Tracker([False, False]), st, "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.text == "hey there" and out.referenced == "the lake"
    assert out.angle == "" and out.item_description == ""
    assert st.openers[0] == ("r", "hinge", "gemini-x", "hey there", "the lake", "", "")
    assert s.recent_openers_snapshot()[0]["angle"] == ""
    assert s.recent_openers_snapshot()[0]["item_description"] == ""


def test_item_index_degrades_to_absent_rather_than_to_the_first_item():
    """A result carrying no item_index at all must NOT read as "she picked item 1". Under the
    1-based contract (doc 5.7) 0 is out of band by construction, which is the whole reason
    ITEM_INDEX_ABSENT exists -- and there is deliberately no getattr fallback to the old
    `referenced_index`, so a stale producer degrades loudly to ABSENT instead of having its
    0-based frame index quietly reinterpreted as an item number (doc 5.3)."""
    class _IndexlessClient:
        def generate(self, profile, style, retry_hint="", *, anchor=None, items=None,
                 should_stop=None,
                     skip_models=frozenset()):
            return SimpleNamespace(model="gemini-x", usage="usage", opener="hey there",
                                   referenced="the lake", referenced_index=7)

    s = OpenerService(_IndexlessClient(), _Tracker([False, False]), _Store(), "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.index == ITEM_INDEX_ABSENT
    assert out.index != 7                               # the old field is not consulted
    assert s.recent_openers_snapshot()[0]["index"] == ITEM_INDEX_ABSENT


# --- OpenerPick.capture_order_index: the ONE crossing between the model's index space and
# the driver's. Between 2026-08-12 and this regression the worker skipped it entirely and
# handed the model's 1-based item number to a 0-based capture-order parameter, where an
# in-range value resolved to a real, adjacent, WRONG frame and was reported on target.

def test_capture_order_index_converts_a_profile_photo_pick_and_only_that_one():
    """The legacy frame shape numbers profile.photos 1..N and the driver's capture order is
    the same list 0-based, so the conversion is exact and derivable: item k is frame k-1."""
    assert OpenerPick("hi", 1, index_space=INDEX_SPACE_PROFILE_PHOTOS).capture_order_index == 0
    assert OpenerPick("hi", 3, index_space=INDEX_SPACE_PROFILE_PHOTOS).capture_order_index == 2


def test_capture_order_index_refuses_the_model_item_space_until_the_driver_table_exists():
    """A crop-shape item number can only be navigated through doc 5.3's driver-owned
    translation table (model index -> heart ordinal), which is a later workflow. With no table
    there is no honest conversion, so this returns None rather than inventing one -- and None
    is also what an unrecognised space gets, so a new space is safe by default."""
    assert OpenerPick("hi", 3, index_space=INDEX_SPACE_MODEL_ITEMS).capture_order_index is None
    assert OpenerPick("hi", 3, index_space="something_new").capture_order_index is None
    assert OpenerPick("hi", 3).capture_order_index is None      # the default is the safe one


def test_capture_order_index_never_turns_absent_into_the_first_frame():
    """ITEM_INDEX_ABSENT is 0, and 0 is a perfectly legal FIRST FRAME in the driver's space.
    Passing it through would convert "the model could not pick an item" into "the opener is
    about item 1", which is exactly the confident wrong send this crossing exists to stop."""
    for space in (INDEX_SPACE_PROFILE_PHOTOS, INDEX_SPACE_MODEL_ITEMS):
        pick = OpenerPick("hi", ITEM_INDEX_ABSENT, index_space=space)
        assert pick.capture_order_index is None, space


def test_maybe_opener_states_the_index_space_on_the_pick_and_in_the_ring_buffer():
    """The space is READ OFF the result, not inferred here, so there is one producer of the
    fact (generate(), which built the payload). It reaches both consumers: the pick the driver
    acts on, and the bug-report ring buffer, where a bare small integer is otherwise
    uninterpretable."""
    class _SpacedClient:
        def generate(self, profile, style, retry_hint="", *, anchor=None, items=None,
                 should_stop=None,
                     skip_models=frozenset()):
            return SimpleNamespace(model="gemini-x", usage="usage", opener="hey there",
                                   referenced="the lake", item_index=2,
                                   index_space=INDEX_SPACE_PROFILE_PHOTOS)

    s = OpenerService(_SpacedClient(), _Tracker([False, False]), _Store(), "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.index_space == INDEX_SPACE_PROFILE_PHOTOS
    assert out.capture_order_index == 1
    assert s.recent_openers_snapshot()[0]["index_space"] == INDEX_SPACE_PROFILE_PHOTOS


def test_a_result_that_states_no_index_space_yields_an_unusable_pick_not_a_guessed_one():
    """A fake or a stale client that never sets the field must not have its number converted
    into a tap. The getattr default is the untranslatable space, so being wrong here means "no
    target", never "target 1" -- the same fail-safe direction as OpenerResult's own default."""
    s = OpenerService(_Client(), _Tracker([False, False]), _Store(), "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.index == _Res.item_index                 # the number still round-trips
    assert out.index_space == INDEX_SPACE_MODEL_ITEMS
    assert out.capture_order_index is None              # but nothing will target from it


# ---------------------------------------------------------------------------------------
# recent_openers_snapshot(): the ring buffer of the most recent SUCCESSFUL opener
# generations, independent of self.store.record_opener's permanent per-run record -- see
# __init__'s recent_openers docstring. Records advisory mode alongside the model's own
# referenced/index/opener fields.
# ---------------------------------------------------------------------------------------

def test_recent_openers_snapshot_excludes_unacted_advisory_suggestions():
    """The diagnostic trail follows real decisions, not drafts.  Advisory suggestions retain
    just a process-local n-gram for duplicate-opening protection and never expose the profile
    detail or message through the bug-report snapshot."""
    c = _Client(opener_texts=["hey, that hiking photo is great",
                              "so that lake looked freezing today"])
    t, st = _Tracker([False, False, False, False]), _Store()
    s = OpenerService(c, t, st, "casual")

    s.maybe_opener("r", "hinge", object())
    s.maybe_opener("r", "bumble", object(), advisory=True)

    snap = s.recent_openers_snapshot()
    assert len(snap) == 1
    auto_entry = snap[0]
    assert auto_entry["advisory"] is False
    assert auto_entry["opener"] == "hey, that hiking photo is great"
    assert auto_entry["referenced"] == _Res.referenced
    # The ring buffer's "index" key kept its NAME and changed its MEANING: it is the
    # 1-based MODEL ITEM INDEX now (doc 5.1/5.7), carried through unshifted from the
    # client's own result. `item_description` beside it is what makes an entry written
    # after this change distinguishable from one written before, in a bug report.
    assert auto_entry["index"] == _Res.item_index
    assert auto_entry["item_description"] == _Res.item_description
    assert auto_entry["angle"] == _Res.angle
    assert auto_entry["entropy_collision"] == ""
    assert auto_entry["entropy_regenerated"] is False
    assert auto_entry["redundancy_markers"] == []


def test_recent_openers_snapshot_is_capped_and_drops_the_oldest():
    """Only the last _RECENT_OPENERS successful generations survive -- append more than the
    cap and the OLDEST entries must be gone, with the newest ones retained in order (newest
    last), not some arbitrary subset."""
    class _CountingClient:
        def __init__(self):
            self.calls = 0

        def generate(self, profile, style, retry_hint="", *, anchor=None, items=None,
                 should_stop=None,
                     skip_models=frozenset()):
            self.calls += 1
            return SimpleNamespace(model="gemini-x", usage="usage",
                                   opener=f"opener #{self.calls}",
                                   referenced=f"item {self.calls}",
                                   item_index=self.calls)

    cap = service_mod._RECENT_OPENERS
    n = cap + 5
    c = _CountingClient()
    s = OpenerService(c, _Tracker(), _Store(), "casual")   # empty queue -> budget_reached() always False

    for _ in range(n):
        s.maybe_opener("r", "hinge", object())

    snap = s.recent_openers_snapshot()
    assert len(snap) == cap
    openers = [entry["opener"] for entry in snap]
    # The oldest 5 calls (#1..#5) were dropped; the newest `cap` calls remain, oldest-of-
    # those-first / newest-last (append order), never re-sorted or reversed.
    assert openers == [f"opener #{i}" for i in range(n - cap + 1, n + 1)]


def test_recent_openers_snapshot_returns_a_plain_list_not_the_live_deque():
    """recent_openers_snapshot() must hand back a stable copy: a bug-report reader on another
    thread must never iterate the live deque a worker thread is concurrently appending to,
    and mutating the returned list must not affect the service's own internal state."""
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")
    s.maybe_opener("r", "hinge", object())

    snap = s.recent_openers_snapshot()
    assert type(snap) is list
    assert snap is not s.recent_openers

    snap.append({"fake": "entry"})
    snap.clear()
    # The service's own ring buffer is untouched by mutating the returned snapshot.
    assert len(s.recent_openers_snapshot()) == 1


def test_a_failed_call_records_nothing_in_the_ring_buffer():
    """Only SUCCESSFUL opener generations land in recent_openers -- a failed call (an
    OpenerError here) must leave the buffer untouched, so the buffer can never imply a
    message was produced when none was."""
    c = _Client(exc=OpenerError("Gemini returned no usable opener for this profile"))
    s = OpenerService(c, _Tracker([False]), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()) is None
    assert s.recent_openers_snapshot() == []


# ---------------------------------------------------------------------------------------
# THE ENTROPY GUARD (service.py's _apply_entropy_guard, ops/OPENER-REDESIGN.md 3.6).
#
# Why it exists: Part A shortens the openers, which compresses the output space, and the
# few-shot edit pairs in the style block make direct copying a live risk. Across a burner
# account sending uncapped volume, a run of messages that all open "Based on that X, I'm
# going to guess ..." is both a bot fingerprint and genuinely embarrassing if two matches
# compare screenshots. The check is a plain leading-n-gram string comparison against the
# recent_openers ring buffer (opener.py's _leading_ngram) -- no semantics, no classifier, no
# second judge, and no constraint on WHAT the model may write. It only ever asks once more.
#
# The three properties below are the whole design, and every one of them is the kind of thing
# a later "simplification" would quietly invert:
#   1. It runs IDENTICALLY under advisory=True and advisory=False -- no branch on advisory
#      anywhere in _apply_entropy_guard. CORRECTED 2026-08-11: this file used to pin the
#      opposite ("skipped OUTRIGHT under advisory, not merely made budget-exempt"), on the
#      theory that a guard consuming an attempt there would consume observe's only allowed
#      attempt. That theory was wrong for THIS guard: point 2 below (the extra draw never
#      consumes the attempt budget) holds in every mode, so there was never an attempt for
#      advisory to lose, and skipping it broke the observe-is-auto-canary property instead
#      (see the advisory tests below) -- a live dry run with a shared OpenerService produced
#      5 advisory openers, 4 with an identical leading phrase, because the guard never looked.
#      Do not restore the old "skip under advisory" behavior; that is the regression this
#      section's advisory tests exist to catch.
#   2. A collision buys exactly ONE extra draw, and that draw does NOT consume the per-profile
#      attempt budget whose exhaustion stops the whole run -- true in AUTO and advisory alike,
#      since advisory forces effective_max_attempts == 1 but the guard's draw was never wired
#      to the attempt loop at all.
#   3. If the second draw collides too it is ACCEPTED and logged. Never a loop, never a
#      rejection, never a stop. Also true in either mode.
# ---------------------------------------------------------------------------------------

# Two openers that deliberately share NO leading words, so switching between them is what
# makes a collision happen or not happen in the tests below. _leading_ngram(..., 4) reduces
# them to "based on that ridgeline" and "you look like you".
_NGRAM_A = "based on that ridgeline i would guess norway"
_NGRAM_A_LEADING = "based on that ridgeline"
_NGRAM_B = "you look like you were freezing out there"


def test_advisory_collision_regenerates_exactly_once_and_sends_the_new_opener(capsys):
    """THE CANARY-PRESERVING CASE -- if this ever regresses, it does so silently, so guard it
    directly. An advisory (observe-mode) opener that opens with words already sent this run
    buys exactly ONE more budget-exempt draw, and the NEW text is what gets shown to the
    operator -- item-for-item the same as the AUTO happy path
    (test_auto_collision_regenerates_exactly_once_and_sends_the_new_opener above), because the
    guard no longer branches on advisory at all.

    This is the whole point of the 2026-08-11 correction: observe mode is supposed to be the
    canary for auto (the owner's rule is that the opener shown to a human in observe must be
    byte-identical to what auto would send), so a collision that AUTO would quietly redraw
    around must be redrawn around here too -- not shipped to the human untouched, which is
    exactly what the old "skip under advisory" behavior did, and exactly what a live dry run
    caught (5 advisory openers, 4 sharing a leading phrase, because the guard never looked)."""
    c = _Client(opener_texts=[_NGRAM_A, _NGRAM_A, _NGRAM_B])
    st = _Store()
    s = OpenerService(c, _Tracker(), st, "casual")

    first = s.maybe_opener("r", "hinge", object(), advisory=True)   # seeds the buffer with A
    assert first.text == _NGRAM_A
    capsys.readouterr()
    out = s.maybe_opener("r", "hinge", object(), advisory=True)     # draft A collides -> redraw

    assert out.text == _NGRAM_B                                     # the SECOND draw is shown
    assert c.calls == 3                                             # 1 + 1 attempt + 1 redraw
    assert s.disabled is False and s.stop_requested is False
    assert s.exhausted_reason is None
    output = capsys.readouterr().out
    assert f'opens with words already sent this run ("{_NGRAM_A_LEADING}")' in output
    assert "EXEMPT from the per-profile attempt budget" in output


def test_advisory_regeneration_does_not_consume_the_retry_budget(capsys):
    """Property 2, stated as the accounting rule it actually is, under advisory specifically:
    if the guard's redraw were ever mistakenly wired into maybe_opener's retry loop, a
    colliding advisory opener could consume the deliberately small failure budget and disable
    suggestions over a stylistic near-miss. It must not: the service stays enabled,
    stop_requested stays False, exhausted_reason stays None, and a third call is still served."""
    c = _Client(opener_texts=[_NGRAM_A, _NGRAM_A, _NGRAM_B, _NGRAM_B])
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    s.maybe_opener("r", "hinge", object(), advisory=True)           # seeds the buffer with A
    out = s.maybe_opener("r", "hinge", object(), advisory=True)     # collides -> redraw to B

    assert out.text == _NGRAM_B
    assert s.disabled is False
    assert s.stop_requested is False
    assert s.exhausted_reason is None
    # The single attempt was not consumed by the guard's redraw, so a further advisory call
    # still gets a real suggestion instead of None -- the exact failure the old "skip under
    # advisory" design was trying (wrongly) to prevent, and the exact failure this correction
    # must not reintroduce from the other direction.
    third = s.maybe_opener("r", "hinge", object(), advisory=True)
    assert third is not None


def test_advisory_second_colliding_draw_is_accepted_not_rejected(capsys):
    """Property 3 holds under advisory too: if the redraw ALSO collides, it is sent to the
    operator anyway, loudly, rather than looping for a third draw or being withheld -- a
    near-repeat surfaced to the human (who can see it and retype) is a much smaller problem
    than an unbounded spend hole or a suggestion silently disappearing."""
    c = _Client(opener_texts=[_NGRAM_A])       # every success repeats the SAME opener
    st = _Store()
    s = OpenerService(c, _Tracker(), st, "casual")

    s.maybe_opener("r", "hinge", object(), advisory=True)          # seeds the buffer with A
    capsys.readouterr()
    out = s.maybe_opener("r", "hinge", object(), advisory=True)    # A collides, redraw is A too

    assert out.text == _NGRAM_A                                    # accepted, not withheld
    assert c.calls == 3                                             # ONE redraw only, never a loop
    assert s.stop_requested is False
    assert s.disabled is False
    assert s.exhausted_reason is None
    output = capsys.readouterr().out
    assert f'still opens with "{_NGRAM_A_LEADING}"' in output
    assert "SENDING it anyway" in output


def test_auto_collision_regenerates_exactly_once_and_sends_the_new_opener(capsys):
    """Property 2, the happy path: an AUTO opener that opens with words already sent this run
    buys exactly ONE more draw, and the NEW text is what gets returned and persisted."""
    c = _Client(opener_texts=[_NGRAM_A, _NGRAM_A, _NGRAM_B])
    st = _Store()
    s = OpenerService(c, _Tracker(), st, "casual")

    first = s.maybe_opener("r", "hinge", object())               # seeds the buffer with A
    assert first.text == _NGRAM_A
    out = s.maybe_opener("r", "hinge", object())                 # draft A collides -> redraw

    assert out.text == _NGRAM_B                                   # the SECOND draw is sent
    assert c.calls == 3                                           # 1 + 1 attempt + 1 redraw
    assert s.disabled is False and s.stop_requested is False
    # Exactly one `openers` row per profile, holding the text actually sent -- never the
    # discarded draft, and never two rows for one profile.
    assert len(st.openers) == 2
    assert [row[3] for row in st.openers] == [_NGRAM_A, _NGRAM_B]
    output = capsys.readouterr().out
    assert f'opens with words already sent this run ("{_NGRAM_A_LEADING}")' in output
    assert "EXEMPT from the per-profile attempt budget" in output


def test_the_regeneration_hint_names_the_repeated_words_without_calling_the_draft_bad():
    """The retry_hint the extra draw is made with is read by the MODEL, so it must (a) name
    the repeated opening words concretely enough to steer away from them and (b) NOT tell the
    model its previous draft was rejected -- it was not, it was fine and merely familiar, and
    the "rejected and NOT sent" wording the real retry path uses would push the model to
    change the wrong things.

    It must also obey the owner's prompt rule: no em dash, no hyphen, plain ASCII. The
    colliding opener here contains hyphens on purpose; _leading_ngram drops punctuation when
    it normalizes, so nothing hyphenated can reach the hint through that substitution."""
    hyphenated = "state-of-the-art view from up there"
    c = _Client(opener_texts=[hyphenated, hyphenated, _NGRAM_B])
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    s.maybe_opener("r", "hinge", object())
    s.maybe_opener("r", "hinge", object())

    assert c.calls == 3
    hint = c.retry_hints[2]
    assert "state of the art" in hint                  # the normalized colliding n-gram
    assert "Your previous draft was fine" in hint
    assert "rejected" not in hint
    # Owner rule: no hyphen and no em dash anywhere in model-facing prompt text. The em dash
    # is written as an escape rather than the literal character on purpose -- the two are
    # near-indistinguishable in most editors, which is the very confusion the rule exists for.
    assert "-" not in hint and "\u2014" not in hint
    assert all(ord(ch) < 128 for ch in hint)           # owner rule: plain ASCII only
    # The extra draw still steers the cascade with whatever the attempt loop had accumulated
    # for this profile (nothing failed to parse here, so: empty).
    assert c.skip_models_seen[2] == frozenset()


def test_the_extra_draw_does_not_consume_the_per_profile_attempt_budget(capsys):
    """Property 2, stated as the accounting rule it actually is. max_attempts=2, and the
    profile spends BOTH attempts (attempt 1 unparseable, attempt 2 usable but colliding). The
    guard's redraw still happens -- a third call to the client -- and the run neither stops
    nor disables.

    That is the whole proof: if the redraw lived inside maybe_opener's `for attempt in
    range(...)` loop, this profile could not have had a third call at all, and a profile whose
    last attempt happened to open with familiar words would be pushed into the max_attempts
    exhaustion that stops the entire run. Doc 3.6 calls that out explicitly -- a hard
    rejection here "can stop a run over a stylistic near-miss, which is disproportionate"."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc_sequence=[None, parse_error], opener_texts=[_NGRAM_A, _NGRAM_A, _NGRAM_B])
    st = _Store()
    s = OpenerService(c, _Tracker(), st, "casual", max_attempts=2)

    assert s.maybe_opener("r", "hinge", object()).text == _NGRAM_A     # profile 1: seeds buffer
    after_profile_1 = c.calls
    assert after_profile_1 == 1
    out = s.maybe_opener("r", "hinge", object())                        # profile 2

    assert out.text == _NGRAM_B
    # Profile 2 spent BOTH of its two allowed attempts and still got a third call: 2 attempts
    # (parse failure, then a usable but colliding draft) + exactly ONE budget-exempt redraw.
    assert c.calls - after_profile_1 == 3
    # The redraw inherits the attempt loop's accumulated skip set -- the model that produced
    # the unparseable attempt 1 is still steered away from, even though the redraw itself is
    # not an attempt.
    assert c.skip_models_seen[-1] == frozenset({"gemini-x"})
    assert s.disabled is False and s.stop_requested is False
    assert s.exhausted_reason is None
    # The redraw is not an ATTEMPT, so it is neither numbered as one nor written to the
    # opener_rejections ledger (which exists to measure how often the deterministic SEND
    # guards reject an attempt -- a draw that never gated a send would inflate exactly that
    # statistic). Only attempt 1's genuine parse failure is recorded.
    assert len(st.rejections) == 1 and st.rejections[0][3] == 1
    assert "attempt 3/" not in capsys.readouterr().out


def test_a_second_colliding_draw_is_accepted_and_never_stops_the_run(capsys):
    """Property 3. The guard asks ONCE. If the redraw opens with the same words again, that
    opener is SENT anyway, loudly -- a near-repeat that goes out is a much smaller problem
    than a retry storm, a stalled profile, or an unbounded spend hole with no ceiling.

    Everything that could turn a cosmetic style check into a failure must stay untouched:
    the run is not stopped, the service is not disabled, neither latch counter moves, and no
    rejection is recorded."""
    c = _Client(opener_texts=[_NGRAM_A])       # every success repeats the SAME opener
    st = _Store()
    s = OpenerService(c, _Tracker(), st, "casual")

    s.maybe_opener("r", "hinge", object())                        # seeds the buffer with A
    capsys.readouterr()
    out = s.maybe_opener("r", "hinge", object())                  # A collides, redraw is A too

    assert out.text == _NGRAM_A                                    # accepted, not rejected
    assert c.calls == 3                                            # ONE redraw only, never a loop
    assert s.stop_requested is False
    assert s.disabled is False
    assert s.exhausted_reason is None
    assert s.last_skip_reason is None                              # this call SUCCEEDED
    assert s._consecutive_bad_requests == 0
    assert s._consecutive_transient_failures == 0
    assert st.rejections == []
    assert len(st.openers) == 2                                    # one row per profile, as always
    output = capsys.readouterr().out
    assert f'still opens with "{_NGRAM_A_LEADING}"' in output
    assert "SENDING it anyway" in output


def test_a_regeneration_that_fails_keeps_the_original_opener_and_moves_no_latch(capsys):
    """The guard must NEVER raise and never turn a perfectly good opener into a failure. If
    the extra draw blows up (a timeout, a fake client that cannot cope with a second call, a
    cascade aborting mid-flight), the original draft -- which was already good enough to send
    -- is what gets sent, and nothing anywhere is told a failure happened: no latch counter
    moves, no last_skip_reason is written, no exhaustion."""
    c = _Client(exc_sequence=[None, None, RuntimeError("connection reset")],
                opener_texts=[_NGRAM_A])
    st = _Store()
    s = OpenerService(c, _Tracker(), st, "casual")

    s.maybe_opener("r", "hinge", object())
    capsys.readouterr()
    out = s.maybe_opener("r", "hinge", object())

    assert out.text == _NGRAM_A                     # the original draft, kept
    assert c.calls == 3
    assert s.disabled is False and s.stop_requested is False
    assert s.exhausted_reason is None
    assert s.last_skip_reason is None
    assert s._consecutive_transient_failures == 0    # a failed REDRAW is not a failed call
    assert len(st.openers) == 2
    output = capsys.readouterr().out
    assert "entropy regeneration failed (RuntimeError: connection reset)" in output
    assert "keeping the original opener" in output


def test_a_regeneration_that_comes_back_empty_still_bills_the_draw_and_keeps_the_original(capsys):
    """The one hole an audit found in _apply_entropy_guard: if the SECOND client.generate()
    call succeeds (no exception, so the OpenerParseError branch never fires) but hands back a
    result whose opener text is empty/whitespace, that call was still real and billed -- and
    the code used to keep the original opener and return without ever recording it.
    _record_billed_draw's own docstring states the invariant this breaks: "A discarded draw is
    no less billed than a sent one, and this service's entire budget contract rests on every
    real call being recorded exactly once."

    A real GeminiOpener never reaches this branch (its _parse() raises
    REASON_EMPTY_AFTER_SANITIZE on an empty opener instead of returning success), so this test
    exercises it the only way it is reachable today: a fake client whose second draw succeeds
    with blank text."""
    c = _Client(opener_texts=[_NGRAM_A, _NGRAM_A, ""])
    t, st = _Tracker(), _Store()
    s = OpenerService(c, t, st, "casual")

    s.maybe_opener("r", "hinge", object())        # profile 1: seeds the buffer with A
    capsys.readouterr()
    out = s.maybe_opener("r", "hinge", object())   # profile 2: collides, redraw comes back empty

    assert out.text == _NGRAM_A                     # the original draft, kept and sent
    assert c.calls == 3
    assert s.disabled is False and s.stop_requested is False
    assert s.exhausted_reason is None
    assert s.last_skip_reason is None
    # The empty redraw is billed (this fix) AND the kept original is billed at the normal
    # post-guard site -- profile 1's single call makes 1, profile 2 makes 2, for 3 total.
    assert len(st.spend) == 3
    assert t.recorded == [("gemini-x", "usage")] * 3
    assert len(st.openers) == 2
    assert st.openers[1][3] == _NGRAM_A              # what was actually sent, not blank text
    output = capsys.readouterr().out
    assert "entropy regeneration returned no opener text" in output
    assert "keeping the original opener" in output


def test_both_draws_are_billed_but_only_the_sent_opener_is_recorded():
    """A discarded draft is no less billed than a sent one, and this service's whole budget
    contract rests on every real call being recorded exactly once. So a regenerated profile
    produces TWO spend records (the thrown-away draft, then the kept draw) and exactly ONE
    `openers` row, holding the text that actually goes out."""
    c = _Client(opener_texts=[_NGRAM_A, _NGRAM_A, _NGRAM_B])
    t, st = _Tracker(), _Store()
    s = OpenerService(c, t, st, "casual")

    s.maybe_opener("r", "hinge", object())        # profile 1: 1 call, 1 spend row
    s.maybe_opener("r", "hinge", object())        # profile 2: 2 calls, 2 spend rows

    assert len(st.spend) == 3
    assert t.recorded == [("gemini-x", "usage")] * 3
    assert len(st.openers) == 2
    assert st.openers[1][3] == _NGRAM_B            # the opener actually sent, not the draft


def test_the_ring_buffer_records_the_collision_and_the_regeneration():
    """recent_openers is the paper trail a bug report reads without a store round-trip (see
    service.py's recent_openers docstring). It must show that THIS opener repeated an earlier
    opening and that a second draw was taken, otherwise a guard that fires constantly -- or
    one that silently stops firing -- is invisible after the fact."""
    c = _Client(opener_texts=[_NGRAM_A, _NGRAM_A, _NGRAM_B])
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    s.maybe_opener("r", "hinge", object())
    s.maybe_opener("r", "hinge", object())

    first, second = s.recent_openers_snapshot()
    assert first["entropy_collision"] == "" and first["entropy_regenerated"] is False
    assert second["entropy_collision"] == _NGRAM_A_LEADING
    assert second["entropy_regenerated"] is True
    # The buffer holds what was SENT, so the next profile's collision check runs against the
    # regenerated text rather than the draft that was thrown away.
    assert second["opener"] == _NGRAM_B


def test_the_guard_skips_the_extra_draw_when_the_run_is_stopping(capsys):
    """A collision is not worth spending on once the run is already stopping. The original
    opener is perfectly sendable, so this degrades to "ship the repetitive one" -- not to a
    failure, and not to a stop of its own."""
    stopping = {"now": False}

    class _StopsAfterHandingBackADraft(_Client):
        """Flips the run's stop flag the instant it hands back the second profile's draft --
        the realistic race, where the operator clicks Stop while the request is in flight.
        Flipping it BEFORE the call instead would short-circuit maybe_opener's own loop-top
        should_stop check and return None before any opener existed, which is a different
        (already covered) branch."""
        def generate(self, *a, **k):
            res = super().generate(*a, **k)
            if self.calls >= 2:
                stopping["now"] = True
            return res

    c = _StopsAfterHandingBackADraft(opener_texts=[_NGRAM_A])
    s = OpenerService(c, _Tracker(), _Store(), "casual")
    should_stop = lambda: stopping["now"]   # noqa: E731

    s.maybe_opener("r", "hinge", object(), should_stop=should_stop)   # seeds the buffer with A
    capsys.readouterr()
    out = s.maybe_opener("r", "hinge", object(), should_stop=should_stop)

    assert out.text == _NGRAM_A                  # the colliding draft, sent as is
    assert c.calls == 2                          # no extra draw was paid for
    assert s.disabled is False and s.stop_requested is False
    output = capsys.readouterr().out
    assert "the run is stopping" in output
    assert "sending it as is rather than spending on another draw" in output


def test_the_guard_skips_the_extra_draw_when_the_budget_is_reached(capsys):
    """Same reasoning as the stopping case: a stylistic near-miss never justifies crossing the
    run budget. Note the guard costs one EXTRA tracker.budget_reached() call on a collision
    path (call 1 spends two answers, call 2's pre-call check a third), so this queue's FOURTH
    entry is the one the guard itself reads."""
    c = _Client(opener_texts=[_NGRAM_A])
    t = _Tracker([False, False, False, True])
    s = OpenerService(c, t, _Store(), "casual")

    s.maybe_opener("r", "hinge", object())
    capsys.readouterr()
    out = s.maybe_opener("r", "hinge", object())

    assert out.text == _NGRAM_A                  # the in-flight opener still returns
    assert c.calls == 2                          # no extra draw was paid for
    output = capsys.readouterr().out
    assert "the run budget is reached" in output
    assert "sending it as is rather than spending on another draw" in output


# ---------------------------------------------------------------------------------------
# THE REDUNDANCY MONITOR (opener.py's _redundant_description_markers, doc 3.7). It is the
# deterministic proxy for the over-description bug this whole redesign targets: if the model
# says in `referenced` that it is reacting to "an outdoor sauna at sunset" and the opener
# contains "sauna" and "sunset", the opener is reciting its own grounding note back to a woman
# who is looking at that exact photo while she reads it.
#
# IT SHIPS LOG ONLY AND MUST NEVER GATE. It is a documented LOWER BOUND (a terse `referenced`
# defeats it completely), it has never been calibrated against real data, and five consecutive
# rejections stop the whole run -- gating on an uncalibrated metric would make it a run-killer.
# The markers travel to the ring buffer so the threshold can eventually be derived offline.
# ---------------------------------------------------------------------------------------

class _RedundantClient:
    """Returns a usable opener that ALSO carries redundancy markers, exactly as opener.py's
    _parse attaches them to a successfully parsed OpenerResult."""
    markers = ['opener restates the referenced word "sauna"',
               'opener restates the referenced word "sunset"']

    def generate(self, profile, style, retry_hint="", *, anchor=None, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        return SimpleNamespace(
            model="gemini-x", usage="usage",
            opener="That view by the sauna during sunset looks relaxing, where is this from?",
            referenced="Photo of her in an outdoor sauna at sunset",
            item_index=1, angle="asking where the sauna is",
            redundancy_markers=list(self.markers))


def test_redundancy_markers_reach_the_ring_buffer_and_never_reject_the_opener(capsys):
    """The markers must be carried into recent_openers (the offline-calibration record doc 3.7
    needs) and must have NO effect on whether the opener is sent: it comes back normally, the
    service stays enabled, the run is not asked to stop, and nothing is written to the
    rejections ledger. The console line says so in as many words."""
    st = _Store()
    s = OpenerService(_RedundantClient(), _Tracker(), st, "casual")

    out = s.maybe_opener("r", "hinge", object())

    assert out.text.startswith("That view by the sauna")   # sent, despite two markers
    assert s.disabled is False and s.stop_requested is False
    assert s.exhausted_reason is None and s.last_skip_reason is None
    assert st.rejections == []                              # never a rejection, by design
    assert len(st.openers) == 1
    entry = s.recent_openers_snapshot()[0]
    assert entry["redundancy_markers"] == _RedundantClient.markers
    output = capsys.readouterr().out
    assert "restates 2 word(s) from its own `referenced` note" in output
    assert "Logged only, never a rejection." in output


def test_a_non_list_redundancy_markers_value_degrades_to_an_empty_list():
    """service.py isinstance-checks the field rather than trusting it, so an older/fake result
    carrying a bare string here degrades to [] instead of being iterated character by
    character into a dozen bogus single-letter markers in the bug report."""
    class _StringMarkersClient:
        def generate(self, profile, style, retry_hint="", *, anchor=None, items=None,
                 should_stop=None,
                     skip_models=frozenset()):
            return SimpleNamespace(model="gemini-x", usage="usage", opener="hey there",
                                   referenced="the lake", item_index=1, angle="",
                                   redundancy_markers="sauna")

    s = OpenerService(_StringMarkersClient(), _Tracker(), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object()).text == "hey there"
    assert s.recent_openers_snapshot()[0]["redundancy_markers"] == []
