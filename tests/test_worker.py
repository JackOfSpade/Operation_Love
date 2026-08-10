"""Worker + OpenerService tests with fakes — no Playwright/emulator/SDK/network."""
import threading
import time

from operation_love.costing import CostTracker, ModelPricing, Usage
from operation_love.drivers.base import DatingAppDriver, DriverClosed
from operation_love.opener.opener import GeminiAPIError, OpenerError, OpenerParseError, OpenerResult
from operation_love.opener.service import OpenerPick, OpenerService
from operation_love.limits import RateLimiter
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import Decision
from operation_love.worker import Worker

PRICING = {"gemini-test-model": ModelPricing(input=5.0, output=25.0)}


# --- fakes ---------------------------------------------------------------
class FakeDriver(DatingAppDriver):
    def __init__(self, n):
        self.cards = [Profile(photos=[b"x"], bio=f"bio{i}") for i in range(n)]
        self.i = 0
        self.likes, self.dislikes, self.opened, self.closed = [], 0, False, False

    def open_session(self): self.opened = True
    def next_profile(self):
        if self.i >= len(self.cards):
            return None
        p = self.cards[self.i]; self.i += 1; return p
    def like(self, opener=None, item_index=0): self.likes.append(opener)
    def dislike(self): self.dislikes += 1
    def out_of_profiles(self): return self.i >= len(self.cards)
    def close(self): self.closed = True


class ClosedDriver(FakeDriver):
    def __init__(self):
        super().__init__(1)
    def next_profile(self):
        raise DriverClosed("browser closed")


class RaisingLikeDriver(FakeDriver):
    """Hinge-style: halts on unexpected, and its like() raises (e.g. an unsent like)."""
    halt_on_error = True

    def __init__(self, n):
        super().__init__(n)
        self.snapshotted = []

    def like(self, opener=None, item_index=0):
        raise RuntimeError("like did not land")

    def snapshot_failure(self, exc):
        self.snapshotted.append((self._adb_live(), exc))

    def _adb_live(self):
        return not self.closed          # snapshot must run while the transport is still open


class FakeDecider:
    def __init__(self, decision="dislike"):
        self.decision = decision
    def decide(self, profile):
        return Decision(decision=self.decision, score=0.9, embedding=[0.1, 0.2], source="ranker")


class StopAfterDecide(FakeDecider):
    def __init__(self, stop_event):
        super().__init__("dislike")
        self.stop_event = stop_event
    def decide(self, profile):
        self.stop_event.set()
        return super().decide(profile)


class FakeOpenerClient:
    def __init__(self, cost_tokens=400):
        self.calls = 0; self.cost_tokens = cost_tokens
        self.should_stops = []   # records should_stop from every call -- see BUG 1's tests
    def generate(self, profile, style, retry_hint="", *, should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        self.should_stops.append(should_stop)
        return OpenerResult(opener=f"hi {self.calls}", referenced="r",
                            usage=Usage(input_tokens=self.cost_tokens), model="gemini-test-model")


class SlowOpenerClient(FakeOpenerClient):
    def generate(self, profile, style, retry_hint="", *, should_stop=None,
                 skip_models=frozenset()):
        time.sleep(0.05)
        return super().generate(profile, style, retry_hint, should_stop=should_stop)


# --- opener clients that fail in the PER-CALL (not global-exhaustion) ways
# maybe_opener() can return None -- see service.py's maybe_opener docstring. OpenerError,
# a single sub-latch 400, and a single sub-latch transient failure each leave
# OpenerService.disabled False (a lone occurrence is below every latch threshold), which
# is exactly the case the worker's no-bare-like guard exists for. ParseErrorOpenerClient
# is different: OpenerParseError is now RETRIED internally by maybe_opener() (see its
# docstring), so a client that keeps failing every call, like this one, drives the retry
# loop all the way to exhaustion (disabled True) rather than leaving the service enabled.
class ParseErrorOpenerClient:
    """Model returned a billed response that didn't parse into a usable opener, every
    single attempt -- exercises maybe_opener()'s retry-until-exhausted path."""
    def __init__(self):
        self.calls = 0
        self.retry_hints = []
    def generate(self, profile, style, retry_hint="", *, should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        self.retry_hints.append(retry_hint)
        raise OpenerParseError("bad JSON in response body", Usage(input_tokens=10),
                               "gemini-test-model")


class OpenerErrorOpenerClient:
    """Per-profile content problem raised before/without a billed round trip (e.g. a
    corrupt captured photo that couldn't be decoded). NOT retried by maybe_opener()."""
    def __init__(self):
        self.calls = 0
    def generate(self, profile, style, retry_hint="", *, should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise OpenerError("Gemini opener: photo index 0 could not be decoded")


class BadRequestOpenerClient:
    """A single HTTP 400 -- below _BAD_REQUEST_LATCH_THRESHOLD, so this alone must not
    disable the service, only skip this one profile's opener."""
    def __init__(self):
        self.calls = 0
    def generate(self, profile, style, retry_hint="", *, should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise GeminiAPIError(400, "INVALID_ARGUMENT", "bad image or request")


class TransientOpenerClient:
    """An unclassified exception (timeout/connection blip) -- below
    _TRANSIENT_LATCH_THRESHOLD, so this alone must not disable the service."""
    def __init__(self):
        self.calls = 0
    def generate(self, profile, style, retry_hint="", *, should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise RuntimeError("connection reset")


class FakeStore:
    def __init__(self):
        self.decisions, self.labels, self.profiles, self.spend, self.openers = [], [], [], [], []
    def load_labels(self): return []
    def record_profile(self, run_id, app, profile_id, liked, source="manual", **k):
        self.profiles.append((app, profile_id, liked, k))
        return True
    def add_label(self, run_id, app, liked, embedding, source="manual", profile_id="", **k):
        self.labels.append((app, profile_id, liked, k))
    def record_decision(self, run_id, app, decision, score, source="auto"):
        self.decisions.append((app, decision, source))
    def record_opener(self, run_id, app, model, opener, referenced): self.openers.append(opener)
    def record_spend(self, run_id, model, usage, cost): self.spend.append(cost)
    def count_today(self, app):
        return sum(1 for row in self.decisions if row[0] == app and row[2] == "auto")
    def flush(self): pass
    def close(self): pass


class _Pacing:
    swipe_delay_s = 0.0


def _worker(driver, decider, service, store):
    return Worker("bumble", driver, decider, service, store, "run1", _Pacing(),
                  threading.Event(), mode="auto")


# --- OpenerService (global budget) --------------------------------------
def test_budget_caps_openers_globally():
    tracker = CostTracker(PRICING, run_budget_usd=0.001)   # 400 input tok = $0.002 > cap
    client = FakeOpenerClient(cost_tokens=400)
    svc = OpenerService(client, tracker, FakeStore(), "style")
    assert svc.maybe_opener("r", "bumble", Profile()).text == "hi 1"   # first allowed
    assert svc.maybe_opener("r", "hinge", Profile()) is None           # second over budget
    assert client.calls == 1 and svc.stop_requested is True


def test_budget_caps_openers_across_concurrent_workers():
    tracker = CostTracker(PRICING, run_budget_usd=0.001)   # first call spends past cap
    client = SlowOpenerClient(cost_tokens=400)
    store = FakeStore()
    svc = OpenerService(client, tracker, store, "style")
    results = []

    def call(app):
        pick = svc.maybe_opener("r", app, Profile())
        results.append(pick.text if pick else None)

    threads = [threading.Thread(target=call, args=(app,)) for app in ("bumble", "hinge")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)

    assert all(not t.is_alive() for t in threads)
    assert client.calls == 1
    assert sorted(results, key=lambda x: x or "") == [None, "hi 1"]
    assert len(store.spend) == 1 and len(store.openers) == 1
    assert svc.stop_requested is True


class _CalibratedDriver(FakeDriver):
    """A driver whose app has a real, measured think_time_s() calibration (Hinge)."""
    think_time_calibrated = True


def test_pace_scales_think_time_by_configured_anchor(monkeypatch):
    from operation_love import worker as worker_mod

    w = _worker(_CalibratedDriver(0), FakeDecider("like"), FakeOpenerClient(), FakeStore())
    monkeypatch.setattr(worker_mod, "think_time_s", lambda decision: 10.0)

    class _Scaled:
        swipe_delay_s = worker_mod._THINK_TIME_BASELINE_S * 0.5   # -> scale 0.5

    w.pacing = _Scaled()
    waited = []
    monkeypatch.setattr(w.stop_event, "wait", lambda s: waited.append(s))
    w._pace("like")
    assert waited == [5.0]                    # 10.0 * 0.5


def test_pace_uses_flat_anchor_for_a_driver_without_calibrated_pacing(monkeypatch):
    # e.g. Bumble: no measured think-time model for this app, so pacing stays the
    # original decision-agnostic anchor + log-normal spread, not a borrowed one.
    from operation_love import worker as worker_mod

    w = _worker(FakeDriver(0), FakeDecider("like"), FakeOpenerClient(), FakeStore())
    monkeypatch.setattr(worker_mod, "think_time_s",
                        lambda decision: (_ for _ in ()).throw(AssertionError("must not be called")))
    monkeypatch.setattr(worker_mod, "human_delay", lambda s: s * 2)
    waited = []
    monkeypatch.setattr(w.stop_event, "wait", lambda s: waited.append(s))
    w._pace("like")
    assert waited == [w.pacing.swipe_delay_s * 2]


def test_pace_maps_dislike_to_the_pass_think_time_bucket(monkeypatch):
    from operation_love import worker as worker_mod

    w = _worker(_CalibratedDriver(0), FakeDecider("like"), FakeOpenerClient(), FakeStore())
    seen = []
    monkeypatch.setattr(worker_mod, "think_time_s", lambda decision: seen.append(decision) or 10.0)

    class _Scaled:
        swipe_delay_s = worker_mod._THINK_TIME_BASELINE_S   # -> scale 1.0

    w.pacing = _Scaled()
    monkeypatch.setattr(w.stop_event, "wait", lambda s: None)
    w._pace("dislike")           # decider decisions are "like"/"dislike", never "pass"
    assert seen == ["pass"]


def test_pace_uses_session_context_when_the_auto_loop_supplies_profile():
    class _Policy:
        def __init__(self):
            self.calls = []

        def post_action_delay_s(self, decision, profile, score, *, scale):
            self.calls.append((decision, profile, score, scale))
            return 8.25

    class _Pacing2:
        swipe_delay_s = 7.0

    class _Stop:
        def __init__(self): self.waits = []
        def wait(self, seconds): self.waits.append(seconds)

    w = Worker.__new__(Worker)
    w.driver = _CalibratedDriver(0)
    w.pacing = _Pacing2()
    w.stop_event = _Stop()
    w._auto_policy = policy = _Policy()
    profile = Profile(photos=[b"x"], bio="hello")

    w._pace("like", profile=profile, score=0.73)

    assert policy.calls == [("like", profile, 0.73, 2.0)]
    assert w.stop_event.waits == [8.25]


def test_auto_session_policy_is_installed_and_counts_only_landed_final_action(monkeypatch):
    """Worker owns session-policy lifecycle; the driver only receives its optional hook."""
    from operation_love import worker as worker_mod

    made = []

    class _Adjusted:
        def __init__(self, decision): self.decision = decision

    class _Policy:
        def __init__(self, *, base_threshold):
            self.base_threshold = base_threshold
            self.applied, self.landed = [], []
            made.append(self)

        def apply_decision(self, decision, profile):
            self.applied.append((decision, profile))
            # Simulate the real policy's only permitted decision change: a marginal
            # model like becomes a pass before the action and rate-limit checks.
            return _Adjusted(Decision("dislike", decision.score, decision.embedding,
                                      "ranker_contextual"))

        def record_landed_action(self, decision):
            self.landed.append(decision)

    class _PolicyDriver(FakeDriver):
        def set_auto_session_policy(self, policy):
            self.installed_policy = policy

    monkeypatch.setattr(worker_mod, "AutoSessionPolicy", _Policy)
    driver = _PolicyDriver(1)
    decider = FakeDecider("like")
    decider.model = type("Model", (), {"threshold": 0.61})()
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")

    Worker("bumble", driver, decider, svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    policy = made[0]
    assert policy.base_threshold == 0.61
    assert driver.installed_policy is policy
    assert len(policy.applied) == 1
    assert policy.landed == ["dislike"]
    assert driver.likes == [] and driver.dislikes == 1
    assert store.decisions == [("bumble", "dislike", "auto")]


# --- Session micro-break: fatigue-hazard model --------------------------
class _NonZeroPacing:
    swipe_delay_s = 3.5


def _break_worker():
    """A worker with nonzero pacing (breaks are a no-op at swipe_delay_s == 0) and
    hand-set fatigue state, so each test controls fatigue directly rather than
    depending on how many swipes happened to run first."""
    w = _worker(FakeDriver(0), FakeDecider("like"), FakeOpenerClient(), FakeStore())
    w.pacing = _NonZeroPacing()
    w._actions_since_break = 0
    w._break_hazard = 0.08
    w._break_due_after = 20.0
    return w


def test_session_break_low_fatigue_effectively_never_fires(monkeypatch):
    """Right after a reset (fatigue ~ 1/20 = 0.05 -> p = 0.08*0.05 = 0.004), even a
    fairly low random() draw must not trigger a break -- a fixed 8%-per-swipe roll
    (the old behavior) would have fired here."""
    from operation_love import worker as worker_mod
    w = _break_worker()
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.01)
    waited = []
    monkeypatch.setattr(w.stop_event, "wait", lambda s: waited.append(s))
    w._maybe_session_break()
    assert waited == []                    # no break fired
    assert w._actions_since_break == 1      # fatigue counter still advances


def test_session_break_fires_once_fatigue_builds(monkeypatch):
    """The SAME random() draw that doesn't fire at low fatigue must fire once
    actions-since-break has run well past _break_due_after (fatigue capped at 2.5x
    -> p = 0.08*2.0 = 0.16 here) -- proving the hazard actually ramps with fatigue
    rather than staying at one constant rate."""
    from operation_love import worker as worker_mod
    w = _break_worker()
    w._actions_since_break = 39            # -> 40 after increment; fatigue = 40/20 = 2.0
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.1)   # between 0.004 and 0.16
    monkeypatch.setattr(worker_mod, "human_delay", lambda anchor, sigma=None: 42.0)
    waited = []
    monkeypatch.setattr(w.stop_event, "wait", lambda s: waited.append(s))
    w._maybe_session_break()
    assert waited == [42.0]
    assert w._actions_since_break == 0      # reset after firing


def test_session_break_hazard_is_capped_after_extreme_fatigue(monkeypatch):
    from operation_love import worker as worker_mod
    w = _break_worker()
    w._actions_since_break = 999
    # Capped p = .08 * 2.5 = .20, so .21 must not fire. Without the cap the raw
    # fatigue multiplier makes p > 1 and every sufficiently old stretch breaks.
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.21)
    waited = []
    monkeypatch.setattr(w.stop_event, "wait", lambda s: waited.append(s))
    w._maybe_session_break()
    assert waited == []
    assert w._actions_since_break == 1000
    # The ceiling is still a fatigue hazard, not the old flat 8% roll: a draw between
    # .08 and the capped .20 must fire once this stretch is extremely old.
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.15)
    monkeypatch.setattr(worker_mod, "human_delay", lambda anchor, sigma=None: 42.0)
    w._maybe_session_break()
    assert waited == [42.0]


def test_session_break_rerolls_hazard_and_interval_after_firing(monkeypatch):
    """After a break fires, both _break_hazard and _break_due_after must be redrawn
    (not left at the same value for the whole run) -- distinguishable sentinels from
    random.uniform prove the specific new attrs get the specific new draws, in the
    hazard-then-interval order the implementation calls them."""
    from operation_love import worker as worker_mod
    w = _break_worker()
    w._actions_since_break = 39
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.0)   # always fires
    monkeypatch.setattr(worker_mod, "human_delay", lambda anchor, sigma=None: 1.0)
    monkeypatch.setattr(w.stop_event, "wait", lambda s: None)
    draws = iter([0.1234, 27.5])           # first call -> hazard, second -> due_after
    monkeypatch.setattr(worker_mod.random, "uniform", lambda a, b: next(draws))
    w._maybe_session_break()
    assert w._break_hazard == 0.1234
    assert w._break_due_after == 27.5


def test_session_break_pause_is_lognormal_not_flat_uniform(monkeypatch):
    """The pause duration comes from human_delay(45.0, sigma=0.5) -- a log-normal
    draw -- not the old flat random.uniform(20, 90)."""
    from operation_love import worker as worker_mod
    w = _break_worker()
    w._actions_since_break = 39
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.0)   # always fires
    seen = {}

    def _fake_human_delay(anchor, sigma=None):
        seen["args"] = (anchor, sigma)
        return 7.0

    monkeypatch.setattr(worker_mod, "human_delay", _fake_human_delay)
    monkeypatch.setattr(worker_mod.random, "uniform", lambda a, b: 0.09)
    waited = []
    monkeypatch.setattr(w.stop_event, "wait", lambda s: waited.append(s))
    w._maybe_session_break()
    assert seen["args"] == (45.0, 0.5)
    assert waited == [7.0]


def test_auto_loop_initializes_session_break_fatigue_state(monkeypatch):
    """_auto_loop must set up the fatigue instance state (not just _maybe_session_break
    assuming it exists) so a fresh run's first micro-break check doesn't crash on a
    missing attribute."""
    driver = FakeDriver(2)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    stop_event = threading.Event()
    monkeypatch.setattr(stop_event, "wait", lambda s=None: None)   # keep the test instant
    w = Worker("bumble", driver, FakeDecider("dislike"), svc, store, "run1", _NonZeroPacing(),
               stop_event, mode="auto")
    w.run()
    assert w._actions_since_break >= 0
    assert 0.04 <= w._break_hazard <= 0.14
    assert 12 <= w._break_due_after <= 35
    assert driver.dislikes == 2
    assert stop_event.is_set() is False


def test_unlimited_limiter_does_not_query_irrelevant_daily_history():
    class _NoDailyReadStore(FakeStore):
        def count_today(self, app):
            raise AssertionError("count_today must not run without max_per_day")

    driver = FakeDriver(2)
    store = _NoDailyReadStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider("dislike"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=RateLimiter()).run()
    assert driver.dislikes == 2


# --- Worker loop ---------------------------------------------------------
def test_worker_dislikes_whole_deck():
    driver = FakeDriver(3)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")  # openers disabled
    _worker(driver, FakeDecider("dislike"), svc, store).run()
    assert driver.dislikes == 3 and driver.likes == []
    # AUTO mode is pure inference: decisions are logged, but NO training labels/profiles
    # are saved (training data comes only from manual/observe swipes).
    assert len(store.decisions) == 3
    assert all(source == "auto" for _, _, source in store.decisions)
    assert store.labels == [] and store.profiles == []
    assert driver.opened and driver.closed


def test_worker_likes_with_openers():
    driver = FakeDriver(3)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert driver.likes == ["hi 1", "hi 2", "hi 3"]
    assert len(store.openers) == 3 and len(store.spend) == 3


def test_worker_skips_openers_when_driver_declines():
    # Bumble-style driver: no swipe-time opener -> never call Gemini (no wasted spend).
    driver = FakeDriver(3)
    driver.accepts_opener = False
    store = FakeStore()
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert driver.likes == [None, None, None]      # liked, but with no opener
    assert client.calls == 0 and store.openers == [] and store.spend == []


def test_worker_stops_before_bare_like_when_opener_budget_exhausts():
    driver = FakeDriver(5)
    store = FakeStore()
    client = FakeOpenerClient(cost_tokens=400)  # $0.002/call
    svc = OpenerService(client, CostTracker(PRICING, run_budget_usd=0.001), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    # The call itself can push the global cap over budget. AUTO must stop before
    # sending a bare like for that same profile, rather than substituting a like
    # with no opener after the provider service requested a stop.
    assert client.calls == 1
    assert driver.likes == []
    assert store.decisions == []
    assert driver.closed


def test_auto_mode_stop_reason_is_visible_in_status_when_opener_budget_exhausts():
    """WS-opener-reason: the owner rule is that the hub must show WHY an auto run stopped,
    not just that it did. An opener-exhaustion stop used to leave AppStatus at the default
    state='stopped' with no reason anywhere -- rendering identically to an operator clicking
    Stop. This pins the whole path: OpenerService records the cause (see
    test_opener_service.py) and the worker publishes it into the shared status the hub
    reads, on the SAME budget-exhaustion trigger as
    test_worker_stops_before_bare_like_when_opener_budget_exhausts above."""
    from operation_love.status import RunStatus

    driver = FakeDriver(5)
    store = FakeStore()
    client = FakeOpenerClient(cost_tokens=400)  # $0.002/call
    svc = OpenerService(client, CostTracker(PRICING, run_budget_usd=0.001), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "stopped"
    assert app["stop_reason"] == "run budget reached"


# ---------------------------------------------------------------------------------------
# COMPLETED RULE: in AUTO mode on an opener-capable app, a like is either sent WITH its
# opener or not sent at all -- and there is no configuration that changes that (the old
# budget.on_exhausted="swipe_without_opener" mode was removed outright; see service.py's
# module docstring). The tests above cover the pre-existing GLOBAL-exhaustion guard
# (stop_requested); these cover every other way maybe_opener() can return None -- a
# per-call failure that leaves OpenerService still enabled (see service.py's maybe_opener
# docstring and last_skip_reason), plus the case where a bad AI response keeps failing
# through every retry attempt and the SERVICE itself ends up exhausted. Each must halt
# before driver.like(), record no decision for the abandoned profile, and publish a
# stop_reason naming the actual cause (not a generic line).
# ---------------------------------------------------------------------------------------

def test_worker_halts_before_bare_like_when_the_opener_ultimately_fails_every_retry():
    """THE OWNER'S RULE: a bad AI response is retried, never sent bare, and if it is STILL
    bad after max_attempts (5, the OpenerService default) that means something is wrong and
    the whole run stops. ParseErrorOpenerClient fails every single call, so the worker's one
    maybe_opener() call drives OpenerService's retry loop out to full exhaustion -- this is
    the "ultimately fails" case, distinct from the single-failure-but-still-enabled cases
    below (OpenerError / a single sub-latch 400 / a single sub-latch transient failure)."""
    driver = FakeDriver(3)
    store = FakeStore()
    client = ParseErrorOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert client.calls == svc.max_attempts == 5   # every retry attempt was actually made
    assert driver.likes == []                       # no bare like sent
    assert store.decisions == []                     # no decision recorded for the abandoned profile
    assert svc.disabled is True and svc.stop_requested is True   # the service gave up for the run
    assert svc.exhausted_reason and "bad JSON in response body" in svc.exhausted_reason
    assert driver.closed


def test_worker_stops_before_bare_like_on_opener_error():
    driver = FakeDriver(3)
    store = FakeStore()
    client = OpenerErrorOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert client.calls == 1
    assert driver.likes == []
    assert store.decisions == []
    assert svc.disabled is False and svc.stop_requested is False
    assert svc.last_skip_reason and "could not be decoded" in svc.last_skip_reason
    assert driver.closed


def test_worker_stops_before_bare_like_on_single_bad_request():
    driver = FakeDriver(3)
    store = FakeStore()
    client = BadRequestOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert client.calls == 1                       # below _BAD_REQUEST_LATCH_THRESHOLD -- didn't latch
    assert driver.likes == []
    assert store.decisions == []
    assert svc.disabled is False and svc.stop_requested is False
    assert svc.last_skip_reason and "HTTP 400" in svc.last_skip_reason
    assert driver.closed


def test_worker_stops_before_bare_like_on_single_transient_failure():
    driver = FakeDriver(3)
    store = FakeStore()
    client = TransientOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert client.calls == 1                       # below _TRANSIENT_LATCH_THRESHOLD -- didn't latch
    assert driver.likes == []
    assert store.decisions == []
    assert svc.disabled is False and svc.stop_requested is False
    assert svc.last_skip_reason and "connection reset" in svc.last_skip_reason
    assert driver.closed


def test_auto_mode_stop_reason_names_the_cause_when_every_retry_attempt_fails():
    """Companion to test_auto_mode_stop_reason_is_visible_in_status_when_opener_budget_exhausts
    above: when OpenerService's retry loop exhausts (see
    test_worker_halts_before_bare_like_when_the_opener_ultimately_fails_every_retry), the hub
    must show the actual cause -- the last attempt's real failure message -- not a generic
    'stopped'."""
    from operation_love.status import RunStatus

    driver = FakeDriver(3)
    store = FakeStore()
    client = ParseErrorOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "stopped"
    assert "bad JSON in response body" in app["stop_reason"]


def test_worker_skips_openers_when_driver_declines_even_through_opener_failures():
    """A non-opener-capable app (Bumble-style) must be COMPLETELY unaffected by the new
    per-call guard -- it never calls the opener service at all in auto mode, so a failing
    opener client (that would halt an opener-capable app) must not even be reached."""
    driver = FakeDriver(3)
    driver.accepts_opener = False
    store = FakeStore()
    client = ParseErrorOpenerClient()          # would halt an opener-capable app every time
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert driver.likes == [None, None, None]  # liked all 3, normally, with no opener
    assert client.calls == 0                   # opener service never even called
    assert len(store.decisions) == 3
    assert svc.last_skip_reason is None         # never touched


def test_worker_with_opener_disabled_by_config_still_likes_normally_in_auto_mode():
    """opener.enabled=false in config produces an OpenerService(client=None, ...), which is
    `disabled` from construction (see OpenerService.__init__). That is a DELIBERATE
    "openers don't exist this run" choice, not a per-call failure of an otherwise-live
    service -- the no-bare-like guard must not mistake it for one and halt the very first
    like of every run. `disabled` is exactly what tells the two apart (see the guard's
    comment in worker.py's _auto_loop)."""
    driver = FakeDriver(3)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    assert svc.disabled is True                # opener.enabled=false -- no client configured

    _worker(driver, FakeDecider("like"), svc, store).run()

    assert driver.likes == [None, None, None]   # liked all 3, normally, with no opener
    assert len(store.decisions) == 3
    assert svc.stop_requested is False           # never asked anyone to stop


def test_observe_mode_stop_reason_is_visible_in_status_when_opener_exhausts():
    """Mirrors the auto-mode pin above for observe mode: a budget/credit stop discovered
    only after a human decision is already recorded (see the "honour the stop only now"
    comment in _observe_loop) must still land in AppStatus.stop_reason, not just silently
    flip the shared stop_event with no visible trace."""
    from operation_love.status import RunStatus

    class ObserveDriver(FakeDriver):
        def __init__(self):
            super().__init__(1)
            self._served = False
        def out_of_profiles(self):
            return self._served
        def current_profile(self):
            self._served = True
            return self.cards[0]
        def wait_for_decision(self, timeout=None, should_stop=None):
            return True                        # user LIKEs
        def render_busy(self, message=None):
            pass

    class ObserveDecider(FakeDecider):
        def embed(self, profile):
            return [0.1, 0.2]
        def retrain(self, store):
            return True

    driver = ObserveDriver()
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    reason = ("all configured Gemini models exhausted their free-tier quota; "
              "no opener capacity remains")
    svc._exhaust(reason)                        # simulates exhaustion discovered mid-run
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    Worker("bumble", driver, ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "stopped"
    assert app["stop_reason"] == reason


def test_worker_stops_at_per_run_like_budget():
    driver = FakeDriver(5)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    limiter = RateLimiter(max_likes_per_run=2)
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=limiter).run()
    # Likes up to the budget, then stops the run before the over-budget like.
    assert driver.likes == ["hi 1", "hi 2"]
    # AUTO is inference-only: 2 decisions logged, no training labels saved.
    assert store.labels == [] and len(store.decisions) == 2
    assert driver.closed


def test_like_ratio_ceiling_demotes_not_halts():
    """Ratio ceiling demotes 'like' to 'pass' — the run continues, not halts."""
    # With target_like_ratio=0.5 and all "like" decisions on 4 cards:
    # card 1: acted=0 → allow (no history) → like.  liked=1, acted=1
    # card 2: 1/1=100% >= 50% → demote → dislike.   liked=1, acted=2
    # card 3: 1/2=50%  >= 50% → demote → dislike.   liked=1, acted=3
    # card 4: 1/3=33%  <  50% → allow → like.       liked=2, acted=4
    driver = FakeDriver(4)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    limiter = RateLimiter(target_like_ratio=0.5)
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=limiter).run()
    # All 4 cards processed — NOT halted
    assert len(store.decisions) == 4
    assert driver.closed
    likes = sum(1 for _, d, _ in store.decisions if d == "like")
    passes = sum(1 for _, d, _ in store.decisions if d == "dislike")
    assert likes == 2 and passes == 2


class _RatioNoZeroGuardLimiter:
    """HOLE 3 (mutation audit): a DELIBERATELY minimal RateLimiter double. The real
    RateLimiter.allow_like_ratio (limits.py) already returns True whenever acted == 0, so
    _auto_loop's OWN `acted > 0` clause (right before the ratio check) is redundant against
    today's real limiter -- nothing breaks if that clause is deleted, and no test built
    against the real RateLimiter would notice. The clause is deliberate defense in depth
    (see _auto_loop's comment above it): it only matters if RateLimiter's internal guard is
    ever removed, or -- exactly what this double is for -- a limiter implementation is used
    that never had the guard in the first place. This double's allow_like_ratio divides
    liked/acted with NO acted == 0 check of its own, so calling it with acted == 0 raises
    ZeroDivisionError; only the worker's own `acted > 0` short-circuit can prevent that
    call from ever happening on the very first 'like' decision of a run (acted is still 0
    at the moment of that check -- see _auto_loop, `acted` only increments AFTER it)."""
    max_per_day = None
    target_like_ratio = 0.5

    def allow(self, acted_this_run, acted_today):
        return True

    def allow_like(self, liked_this_run):
        return True

    def allow_like_ratio(self, liked, acted):
        return liked / acted < self.target_like_ratio   # no acted == 0 guard, unlike limits.py

    def describe(self):
        return "test double (no internal zero guard)"


def test_first_like_survives_a_limiter_double_with_no_internal_zero_guard():
    """Confirms _auto_loop's `acted > 0` clause is load-bearing against a limiter double
    that (unlike the real RateLimiter) does not itself guard acted == 0: the very first
    'like' decision of the run must land normally, with no ZeroDivisionError, because the
    worker's own guard keeps allow_like_ratio(liked=0, acted=0) from ever being called."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=_RatioNoZeroGuardLimiter()).run()

    assert driver.likes == ["hi 1"]                         # the like landed, not swallowed
    assert store.decisions == [("bumble", "like", "auto")]


def test_worker_daily_limit_ignores_manual_decisions():
    driver = FakeDriver(2)
    store = FakeStore()
    store.decisions = [("bumble", "like", "manual")] * 83
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider("dislike"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=RateLimiter(max_per_day=1)).run()

    assert driver.dislikes == 1
    assert store.count_today("bumble") == 1
    assert store.decisions[-1] == ("bumble", "dislike", "auto")
    assert all(source == "manual" for _, _, source in store.decisions[:-1])


def test_no_auto_decision_recorded_when_like_raises_and_snapshot_runs_live():
    driver = RaisingLikeDriver(2)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")   # openers disabled -> pick None
    _worker(driver, FakeDecider("like"), svc, store).run()
    # like() raised -> halt_on_error -> the phantom 'like' must NOT be recorded (record is post-action)
    assert store.decisions == []
    assert driver.closed
    # the failure snapshot must have run while the transport was still live (before close())
    assert driver.snapshotted and driver.snapshotted[0][0] is True


def test_worker_treats_browser_close_as_graceful_stop():
    driver = ClosedDriver()
    store = FakeStore()
    stop_event = threading.Event()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           stop_event, mode="auto").run()

    assert stop_event.is_set()
    assert driver.closed
    assert store.labels == []
    assert store.profiles == []


def test_worker_stops_before_writing_auto_decision_after_stop():
    driver = FakeDriver(1)
    store = FakeStore()
    stop_event = threading.Event()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, StopAfterDecide(stop_event), svc, store, "run1", _Pacing(),
           stop_event, mode="auto").run()

    assert store.decisions == []
    assert store.labels == [] and store.profiles == []
    assert driver.dislikes == 0 and driver.likes == []
    assert driver.closed


def test_auto_mode_halts_on_unexpected_even_when_a_driver_opts_out_of_halting():
    """AUTO mode must STOP on any unexpected error for EVERY app, EVEN one that explicitly sets
    halt_on_error=False, so an autonomous run never keeps swiping blindly. It must halt on the
    FIRST error (no restart retries) and set the stop so buffered data is saved.

    The opt-out is set explicitly here rather than relied on as a default: halt_on_error now
    defaults to True on DatingAppDriver, so "a driver that never mentions it" no longer means
    "a driver that restarts". Setting it False makes this the strongest version of the claim —
    auto mode overrides the opt-out."""
    class BoomDriver(FakeDriver):
        halt_on_error = False           # explicitly opted out — auto must override it anyway
        def __init__(self, n):
            super().__init__(n)
            self.calls = 0
        def next_profile(self):
            self.calls += 1
            raise RuntimeError("unexpected boom")

    driver = BoomDriver(3)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    stop = threading.Event()
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           stop, mode="auto", max_restarts=5).run()

    assert driver.calls == 1        # halted on the first error — did NOT restart-and-retry
    assert stop.is_set()            # stop set (halt path), so the supervisor saves buffered data
    assert driver.closed


def test_observe_mode_restarts_only_when_a_driver_explicitly_opts_out_of_halting(monkeypatch):
    """Restart resilience in observe mode is now OPT-IN (halt_on_error=False), not the default.

    It used to be what you got by saying nothing: worker.py read `getattr(driver,
    "halt_on_error", False)`, so PlaywrightDriver — which declared the attribute nowhere —
    received restart-with-backoff by omission rather than by decision. The capability still
    exists for a genuinely flaky, human-supervised observe session; it just has to be asked
    for now."""
    import operation_love.worker as wmod
    monkeypatch.setattr(wmod, "human_cooldown", lambda s: 0)   # no backoff sleep in the test

    class FlakyObserve(FakeDriver):
        halt_on_error = False           # explicit opt-in to restarts
        def __init__(self, n):
            super().__init__(n)
            self.attempts = 0
        def open_session(self):
            self.attempts += 1
            if self.attempts <= 2:
                raise RuntimeError("transient")
            self.opened = True                                  # 3rd try: empty deck -> clean finish

    driver = FlakyObserve(0)                                    # 0 cards -> out_of_profiles True
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    stop = threading.Event()
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           stop, mode="observe", max_restarts=5).run()

    assert driver.attempts == 3     # restarted twice, then succeeded — did NOT halt on first error
    assert not stop.is_set()        # finished cleanly, not a halt
    assert driver.closed


def test_observe_status_says_wait_during_capture_and_embed():
    """Hub banner = the only swipe/wait feedback both apps share in observe (Bumble's in-page
    overlay is off by default, Hinge has none). It reads the per-app STATE, so the worker must
    hold a non-'waiting' state through BOTH no-swipe phases: reading the card and embedding the
    swipe you just made. Otherwise the banner says SWIPE during the slow embed and the next
    swipe gets mis-attributed — identically wrong for both apps."""
    from operation_love.status import RunStatus
    from operation_love.worker import _OBSERVE_CAPTURE_BUSY, _OBSERVE_PROCESSING_BUSY

    seen = {}

    class ObserveDriver(FakeDriver):
        def __init__(self):
            super().__init__(1)
            self._served = False
            self.busy = []
        def out_of_profiles(self):            # serve exactly one card, then the deck is empty
            return self._served
        def current_profile(self):
            seen["capture_state"] = status.app_view("bumble")["app"]["state"]
            self._served = True
            return self.cards[0]
        def wait_for_decision(self, timeout=None, should_stop=None):
            seen["wait_state"] = status.app_view("bumble")["app"]["state"]
            return True                        # user LIKEs
        def render_busy(self, message=None):
            self.busy.append(message)

    class ObserveDecider(FakeDecider):
        def embed(self, profile):
            seen["embed_state"] = status.app_view("bumble")["app"]["state"]
            return [0.1, 0.2]
        def retrain(self, store):
            return True

    status = RunStatus("run1", ["bumble"], min_labels=40, mode="observe")
    driver = ObserveDriver()
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert seen["capture_state"] == "capturing"   # WAIT while reading the card
    assert seen["wait_state"] == "waiting"        # SWIPE: the one moment a swipe is wanted
    assert seen["embed_state"] == "acting"        # WAIT while embedding (NOT 'waiting'/SWIPE)
    # both no-swipe phases also drove the in-page busy channel (parity for overlay-capable apps)
    assert _OBSERVE_CAPTURE_BUSY in driver.busy and _OBSERVE_PROCESSING_BUSY in driver.busy
    assert driver.closed and store.labels and store.labels[0][0] == "bumble"


def test_observe_persists_label_when_stop_requested_during_embed():
    """WS-003: Stop pressed during the (slow) embed — after the manual swipe happened and
    its photos were already archived via record_profile — must not discard the completed
    label. Stop should end the loop AFTER committing the work in hand, not before."""
    class ObserveDriver(FakeDriver):
        def __init__(self):
            super().__init__(1)
            self._served = False
        def out_of_profiles(self):
            return self._served
        def current_profile(self):
            self._served = True
            return self.cards[0]
        def wait_for_decision(self, timeout=None, should_stop=None):
            return True                        # user LIKEs
        def render_busy(self, message=None):
            pass

    class StopDuringEmbedDecider(FakeDecider):
        def __init__(self, stop_event):
            super().__init__()
            self.stop_event = stop_event
        def embed(self, profile):
            self.stop_event.set()              # Stop pressed while the (slow) embed was running
            return [0.1, 0.2]
        def retrain(self, store):
            return True

    driver = ObserveDriver()
    store = FakeStore()
    stop_event = threading.Event()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, StopDuringEmbedDecider(stop_event), svc, store, "run1", _Pacing(),
           stop_event, mode="observe").run()

    # the swipe already happened and the profile was archived -> the label must survive
    assert store.profiles and store.profiles[0][2] is True
    assert store.labels and store.labels[0][2] is True
    assert store.decisions and store.decisions[0][1] == "like"
    assert driver.closed


def test_pace_scales_wait_by_configured_swipe_delay(monkeypatch):
    """Every existing pacing test uses _Pacing.swipe_delay_s == 0.0 (the early-return path)
    -- the scaling branch itself (`scale = swipe_delay_s / _THINK_TIME_BASELINE_S`) was
    entirely unexercised. Pin the formula deterministically: stub think_time_s() to a fixed
    base and assert stop_event.wait() is called with base * scale, not the raw base.

    Needs a driver declaring think_time_calibrated: _pace() only uses the measured
    like/pass dwell shape for apps it was actually calibrated on (Hinge), and falls back
    to a flat anchor otherwise -- so without one this exercises the wrong branch."""
    import operation_love.worker as wmod
    monkeypatch.setattr(wmod, "think_time_s", lambda key: 2.0)   # fixed, deterministic base

    class _Pacing2:
        swipe_delay_s = 7.0   # 2x the 3.5s default anchor -> scale == 2.0

    class _RecordingStopEvent:
        def __init__(self):
            self.waits = []
        def wait(self, t):
            self.waits.append(t)
        def is_set(self):
            return False

    class _CalibratedDriver:
        think_time_calibrated = True

    w = Worker.__new__(Worker)
    w.driver = _CalibratedDriver()
    w.pacing = _Pacing2()
    w.stop_event = _RecordingStopEvent()
    w._pace("like")
    assert w.stop_event.waits == [4.0]   # 2.0 * (7.0 / 3.5)


def test_auto_defer_cold_start_stops_without_action_or_record():
    """AUTO cold-start: a not-ready ranker returns 'defer' -> the loop STOPS after pulling the
    first card, takes no swipe, records no decision/label, and still closes the driver."""
    class DeferDecider(FakeDecider):
        def decide(self, profile):
            return Decision(decision="defer", score=0.0, embedding=[0.1], source="cold_start")

    driver = FakeDriver(3)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    _worker(driver, DeferDecider(), svc, store).run()

    # (a) no action: nothing liked, nothing disliked
    assert driver.likes == [] and driver.dislikes == 0
    # (b) auto 'defer' records neither a decision nor a label/profile
    assert store.decisions == [] and store.labels == [] and store.profiles == []
    # (c) the finally ran -> driver closed
    assert driver.closed
    # (d) exactly one profile pulled, then the loop broke immediately
    assert driver.i == 1


def test_observe_mode_halts_by_default_when_a_driver_says_nothing_about_halting():
    """The fail-CLOSED default: a driver that never mentions halt_on_error must HALT, not
    restart. This is the regression guard for the bug this replaced — worker.py read
    `getattr(driver, "halt_on_error", False)`, so the riskier behaviour was what a driver
    got by FORGETTING, and PlaywrightDriver (which declared it nowhere) silently had it.

    Restarting is not free even in observe mode, where the bot only reads: the worker
    re-attaches to whatever is on screen, and a driver that was confused about which card
    it was looking at then mis-attributes the manual swipes it records afterwards --
    corrupting the taste model permanently, long after the session that caused it. Ending
    a seeding session early is cheap by comparison."""
    class SilentFlaky(FakeDriver):
        # deliberately declares NO halt_on_error -- inherits DatingAppDriver's True
        def __init__(self, n):
            super().__init__(n)
            self.attempts = 0
        def open_session(self):
            self.attempts += 1
            raise RuntimeError("transient")

    assert SilentFlaky(0).halt_on_error is True, "the ABC must supply the safe default"
    driver = SilentFlaky(0)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    stop = threading.Event()
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           stop, mode="observe", max_restarts=5).run()

    assert driver.attempts == 1     # halted on the FIRST error -- no restart-and-retry
    assert stop.is_set()            # and stopped, so buffered data still gets saved


# ---------------------------------------------------------------------------------------
# should_stop -- BUG 1 (adversarial audit): OpenerService.maybe_opener() (and, through it,
# GeminiOpener.generate()) can now abort an in-flight retry/cascade sequence as soon as a
# Stop click is observed, instead of running the full sequence to completion (up to ~52
# minutes against the shipped config -- see opener.py's and service.py's should_stop
# docstrings). That only helps if worker.py actually PASSES its stop signal through at
# every maybe_opener() call site. There are exactly two: the AUTO-loop like path, and the
# OBSERVE-mode on_like_intent suggestion path (Hinge's post-heart comment sheet).
# ---------------------------------------------------------------------------------------

def test_auto_loop_passes_stop_event_is_set_as_should_stop_to_maybe_opener():
    """Pins the AUTO-loop call site: worker.py must forward self.stop_event.is_set (not some
    other callable, and not omit it) so a Stop click can abort an in-flight opener call."""
    driver = FakeDriver(1)
    store = FakeStore()
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    w = _worker(driver, FakeDecider("like"), svc, store)
    w.run()

    assert client.calls == 1
    assert client.should_stops == [w.stop_event.is_set]


class _ObserveLikeIntentDriver(FakeDriver):
    """A Hinge-style observe driver: exposes the post-heart comment-sheet suggestion hook
    (supports_observe_like_intent) and calls it exactly like the real driver does, so
    _wait_for_observed_decision's on_like_intent callback (the one that calls
    maybe_opener()) actually runs."""
    supports_observe_like_intent = True
    accepts_opener = True

    def __init__(self):
        super().__init__(1)
        self._served = False

    def out_of_profiles(self):
        return self._served

    def current_profile(self):
        self._served = True
        return self.cards[0]

    def render_busy(self, message=None):
        pass

    def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
        if on_like_intent is not None:
            on_like_intent(True)     # simulate opening Hinge's like/comment sheet
        return True                  # then LIKE


def test_observe_mode_on_like_intent_passes_stop_event_is_set_to_maybe_opener():
    """Pins the OTHER call site: the observe-mode opener suggestion surfaced after the
    operator opens Hinge's like sheet must ALSO thread should_stop=self.stop_event.is_set,
    not just the auto-loop path pinned above."""
    class ObserveDecider(FakeDecider):
        def embed(self, profile):
            return [0.1, 0.2]
        def retrain(self, store):
            return True

    driver = _ObserveLikeIntentDriver()
    store = FakeStore()
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    w = Worker("bumble", driver, ObserveDecider(), svc, store, "run1", _Pacing(),
              threading.Event(), mode="observe")
    w.run()

    assert client.calls == 1
    assert client.should_stops == [w.stop_event.is_set]


# ---------------------------------------------------------------------------------------
# A: an advisory suggestion failure must NEVER end the observe session; advisory=True must
# be the kwarg the observe on_like_intent call site actually passes; the AUTO-loop like path
# must be completely unaffected (advisory stays False there, exactly today's behavior).
#
# B: the hub must show something while the blocking suggestion call is in flight -- an
# interim "suggesting" status published before the call, cleared unconditionally right
# after (success or exception).
# ---------------------------------------------------------------------------------------

class _ObserveDecider(FakeDecider):
    def __init__(self):
        super().__init__("dislike")   # decision value is irrelevant -- observe never uses it
    def embed(self, profile):
        return [0.1, 0.2]
    def retrain(self, store):
        return True


class _RecordingOpenerService:
    """Records every maybe_opener() call's kwargs (advisory, should_stop) and, if `status`
    is supplied, the app's live state AT THE MOMENT of the call -- this is what change B's
    tests use to prove the interim 'suggesting' state is actually up while the call is
    happening, not just before/after it. `raise_exc`, if set, makes the call raise instead
    of returning a pick -- exercising the exception path of the try/except in
    _wait_for_observed_decision's on_like_intent."""
    stop_requested = False
    disabled = False

    def __init__(self, *, status=None, app=None, raise_exc=None, suggestion="hi"):
        self.status = status
        self.app = app
        self.raise_exc = raise_exc
        self.suggestion = suggestion
        self.calls = []
        self.state_during_call = None

    def maybe_opener(self, run_id, app, profile, *, should_stop=None, advisory=False):
        self.calls.append({"run_id": run_id, "app": app, "should_stop": should_stop,
                           "advisory": advisory})
        if self.status is not None:
            self.state_during_call = self.status.app_view(self.app)["app"]["state"]
        if self.raise_exc is not None:
            raise self.raise_exc
        return OpenerPick(self.suggestion, index=0)


def test_observe_on_like_intent_passes_advisory_true():
    """The observe suggestion call site must pass advisory=True -- this is what makes
    maybe_opener() use exactly one attempt and route any exhaustion through
    request_stop=False instead of ending the session."""
    driver = _ObserveLikeIntentDriver()
    store = FakeStore()
    svc = _RecordingOpenerService()
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe").run()

    assert svc.calls and svc.calls[0]["advisory"] is True


def test_auto_loop_like_call_does_not_pass_advisory():
    """Companion to the pin above: the AUTO-loop like path must be completely unchanged --
    it must NOT pass advisory=True (the default, False, keeps today's max_attempts-retries-
    then-halt behavior)."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = _RecordingOpenerService()
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert svc.calls and svc.calls[0]["advisory"] is False


def test_observe_suggestion_publishes_interim_suggesting_state_during_the_call():
    """The hub must show SOMETHING while the blocking suggestion call is in flight, instead
    of the stale 'click pass X or heart' banner from before the heart tap. Captured from
    INSIDE the fake service's maybe_opener() -- i.e. at the exact moment the real, slow call
    would be blocking -- so this actually proves the state was live during the call, not
    merely bracketing it."""
    from operation_love.status import RunStatus

    driver = _ObserveLikeIntentDriver()
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    svc = _RecordingOpenerService(status=status, app="bumble")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert svc.state_during_call == "suggesting"


def _record_state_transitions(status):
    """Wrap status.set_app to record every (fields) call while still applying it normally --
    lets a test see the STATE SEQUENCE over time, not just the observe loop's FINAL state.
    The final state is the wrong thing to assert on for an interim marker like 'suggesting':
    the loop's LATER stages (e.g. _block_observe_processing's 'acting') overwrite it again
    well before the run ends, which would make a 'cleared' assertion pass for the wrong
    reason (loop progress, not the specific transition this call site is responsible for)."""
    calls = []
    original = status.set_app
    def recording(app, **fields):
        calls.append(dict(fields))
        original(app, **fields)
    status.set_app = recording
    return calls


def test_observe_suggestion_clears_interim_state_after_success():
    from operation_love.status import RunStatus

    driver = _ObserveLikeIntentDriver()
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(status=status, app="bumble", suggestion="loved your trail photo")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    states = [c["state"] for c in calls if "state" in c]
    suggesting_idx = states.index("suggesting")
    waiting_for_send_idx = states.index("waiting_for_send")
    assert waiting_for_send_idx == suggesting_idx + 1   # published, then cleared by the VERY NEXT
                                                          # state transition -- nothing else runs
                                                          # between them
    waiting_for_send_call = next(c for c in calls if c.get("state") == "waiting_for_send")
    assert waiting_for_send_call.get("opener_suggestion") == "loved your trail photo"


def test_observe_suggestion_clears_interim_state_after_an_exception():
    """The interim marker must be cleared even when the call raises -- worker.py's on_like_
    intent already wraps the call in try/except (a suggestion must never block a human
    send), and the unconditional state transition right after it is what clears
    'suggesting' on every path, not only the happy one."""
    from operation_love.status import RunStatus

    driver = _ObserveLikeIntentDriver()
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(status=status, app="bumble",
                                  raise_exc=RuntimeError("boom"))
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert svc.state_during_call == "suggesting"    # it WAS published before the call
    states = [c["state"] for c in calls if "state" in c]
    suggesting_idx = states.index("suggesting")
    waiting_for_send_idx = states.index("waiting_for_send")
    assert waiting_for_send_idx == suggesting_idx + 1   # cleared right after, even on a raise
    waiting_for_send_call = next(c for c in calls if c.get("state") == "waiting_for_send")
    assert waiting_for_send_call.get("opener_suggestion") is None   # the raise produced no suggestion


class _TwoCardObserveLikeIntentDriver(FakeDriver):
    """Like _ObserveLikeIntentDriver, but serves TWO cards, each with a LIKE outcome that
    opens (and closes) Hinge's comment sheet -- lets a test drive maybe_opener() twice, once
    per profile, to prove a failure on card 1 doesn't poison card 2."""
    supports_observe_like_intent = True
    accepts_opener = True

    def __init__(self):
        super().__init__(2)
        self.served = 0

    def out_of_profiles(self):
        return self.served >= 2

    def current_profile(self):
        card = self.cards[self.served]
        self.served += 1
        return card

    def render_busy(self, message=None):
        pass

    def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
        if on_like_intent is not None:
            on_like_intent(True)
            on_like_intent(False)
        return True   # LIKE both cards


def test_advisory_suggestion_failure_never_stops_the_observe_run_end_to_end():
    """The headline contract for change A, driven through the REAL OpenerService (not a
    fake) with a client that fails to parse on EVERY call: today's max_attempts (5) would
    burn 5 real calls and, on exhaustion, set stop_requested -- which _observe_loop honours,
    ENDING the whole labelling session over a display-only failure. advisory=True must
    instead use exactly ONE attempt per card and leave stop_requested False, so BOTH cards'
    human decisions get processed and persisted -- the entire point of observe mode."""
    driver = _TwoCardObserveLikeIntentDriver()
    store = FakeStore()
    client = ParseErrorOpenerClient()          # every attempt fails to parse, forever
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    w = Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
              threading.Event(), mode="observe")
    w.run()

    # Card 1's single advisory attempt already exhausts the service (disabled=True), so
    # card 2's call short-circuits on `disabled` at the very top of maybe_opener() without
    # ever reaching the client again -- exactly one real call total, not two, and nowhere
    # near the 5 a full AUTO-style retry storm would have burned on card 1 alone.
    assert client.calls == 1
    assert svc.disabled is True                 # spend still protected
    assert svc.stop_requested is False           # but the run itself was never asked to stop
    assert not w.stop_event.is_set()
    assert len(store.labels) == 2                # BOTH human decisions persisted
    assert [row[2] for row in store.labels] == [True, True]
    assert driver.closed


# ---------------------------------------------------------------------------------------
# C: opener_service=None must never crash the AUTO loop. Not reachable via supervisor.run()
# today (it always constructs a real OpenerService, even with openers disabled), but Worker
# is a public class any other caller can construct directly, and an audit proved BOTH the
# maybe_opener() call on a like AND the post-action stop_requested check (which runs after
# EVERY action, so even a dislike-only run hit it) raised a bare
# `AttributeError: 'NoneType' object has no attribute ...` instead of this codebase's usual
# clear, actionable failure.
# ---------------------------------------------------------------------------------------

def test_auto_like_with_opener_service_none_does_not_crash():
    from operation_love.status import RunStatus

    driver = FakeDriver(1)
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), None, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] != "error"          # not the HALT-on-unexpected path (an AttributeError)
    assert app.get("error") is None
    assert driver.likes == [None]           # no opener service -> liked, but with no opener
    assert driver.closed


def test_auto_dislike_with_opener_service_none_does_not_crash():
    """The post-action stop_requested check runs after EVERY action -- a dislike-only run
    (which never calls maybe_opener() at all) must not crash on it either."""
    from operation_love.status import RunStatus

    driver = FakeDriver(2)
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("dislike"), None, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] != "error"
    assert app.get("error") is None
    assert driver.dislikes == 2
    assert driver.closed
