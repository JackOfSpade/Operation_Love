"""OpenerService branch logic: disabled / budget / provider failures / stop-vs-continue.

The service is the GLOBAL, budget-aware gate for openers shared by every worker.
It's exercised indirectly elsewhere; this pins its own decision branches with
lightweight fakes (no provider SDK/network).
"""
import pytest

import operation_love.opener.service as service_mod
from operation_love.opener.opener import (
    GeminiAPIError,
    GeminiCapacityExhausted,
    OpenerAborted,
    OpenerError,
    OpenerParseError,
)
from operation_love.opener.service import OpenerService


class _Res:
    model = "gemini-x"
    usage = "usage"
    opener = "hey, that hiking photo is great"
    referenced = "hiking"
    referenced_index = 2


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
    """
    def __init__(self, exc=None, exc_sequence=None):
        self.exc = exc
        self.exc_sequence = list(exc_sequence) if exc_sequence is not None else None
        self.calls = 0
        self.retry_hints = []
        self.should_stops = []
        self.skip_models_seen = []

    def generate(self, profile, style, retry_hint="", *, should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        self.retry_hints.append(retry_hint)
        self.should_stops.append(should_stop)
        self.skip_models_seen.append(skip_models)
        if self.exc_sequence is not None:
            step = self.exc_sequence.pop(0) if self.exc_sequence else None
            if step is not None:
                raise step
            return _Res()
        if self.exc:
            raise self.exc
        return _Res()


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

    def record_spend(self, *a):
        self.spend.append(a)

    def record_opener(self, *a):
        self.openers.append(a)


def test_disabled_without_client():
    s = OpenerService(None, _Tracker(), _Store(), "casual")
    assert s.disabled is True
    assert s.maybe_opener("r", "bumble", object()) is None


def test_generates_records_and_stays_enabled():
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")
    out = s.maybe_opener("r", "bumble", object())
    assert out.text == _Res.opener and out.index == _Res.referenced_index and c.calls == 1
    assert len(st.spend) == 1 and len(st.openers) == 1     # both spend + opener persisted
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
# last_skip_reason: the per-call (last-writer-wins) companion to exhausted_reason (first-
# writer-wins, run-lifetime). It exists so a caller that must never send a like with no
# opener (worker.py's _auto_loop) can report the ACTUAL cause of a per-call failure that
# leaves the service otherwise healthy, instead of a generic line. See maybe_opener's and
# last_skip_reason's own docstrings in service.py for the exact contract: set on every
# early-return-None path that does NOT go through _exhaust() (unparseable response,
# OpenerError, a sub-latch 400, a sub-latch transient failure), and cleared on success.
# ---------------------------------------------------------------------------------------

def test_last_skip_reason_starts_unset():
    s = OpenerService(_Client(), _Tracker(), _Store(), "casual")
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

@pytest.mark.parametrize("bad_max_attempts", [0, -1, True, "5", 2.5], ids=[
    "zero", "negative", "bool_true", "string", "float"])
def test_max_attempts_must_be_a_positive_int(bad_max_attempts):
    """bool is rejected explicitly even though it IS an int subclass in Python (True == 1):
    silently coercing max_attempts=True into 1 would be a confusing accident, not a real
    configuration choice."""
    with pytest.raises(ValueError, match="max_attempts"):
        OpenerService(_Client(), _Tracker(), _Store(), "casual", max_attempts=bad_max_attempts)


# ---------------------------------------------------------------------------------------
# A: advisory=True (Hinge's observe-mode post-heart suggestion) -- a display-only failure
# must never end the observe session. maybe_opener(advisory=True) uses exactly ONE attempt
# (never self.max_attempts) and routes every exhaustion through _exhaust(request_stop=False):
# disabled/exhausted_reason are still set (spend stays protected), but stop_requested is left
# alone. The default (advisory=False, every existing test in this file) must be completely
# unaffected -- none of those tests pass advisory at all, so they pin that on their own.
# ---------------------------------------------------------------------------------------

def test_advisory_uses_exactly_one_attempt_not_max_attempts():
    """A parse failure that would normally retry up to max_attempts (5, the default) times
    must stop after exactly ONE attempt when advisory=True -- a human is sitting there
    waiting on this call, so a multi-attempt retry storm is a UX problem here, not a
    spend-protecting safeguard."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc=parse_error)                # would keep failing every attempt forever
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    assert s.maybe_opener("r", "hinge", object(), advisory=True) is None
    assert c.calls == 1                          # not 5
    assert s.disabled is True                    # spend still protected
    assert s.stop_requested is False              # but the run itself was never asked to stop


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
    with an empty skip set again, even though the FIRST profile accumulated a failing model."""
    parse_error = OpenerParseError("bad JSON", "usage", "gemini-x")
    c = _Client(exc_sequence=[parse_error])   # profile 1: attempt 1 fails, attempt 2 succeeds
    s = OpenerService(c, _Tracker(), _Store(), "casual")

    s.maybe_opener("r", "hinge", object())                    # profile 1
    assert c.skip_models_seen == [frozenset(), frozenset({"gemini-x"})]

    c.skip_models_seen = []                                   # isolate profile 2's own calls
    s.maybe_opener("r", "hinge", object())                    # profile 2 -- fresh call, no exc queued
    assert c.skip_models_seen == [frozenset()]                 # NOT frozenset({"gemini-x"})


def test_advisory_single_attempt_also_passes_an_empty_skip_models():
    """advisory's single attempt is still 'attempt 1' of its own call -- skip_models must be
    empty (nothing has failed yet within THIS call) exactly like a first AUTO attempt."""
    c, t, st = _Client(), _Tracker([False, False]), _Store()
    s = OpenerService(c, t, st, "casual")
    s.maybe_opener("r", "hinge", object(), advisory=True)
    assert c.skip_models_seen == [frozenset()]
