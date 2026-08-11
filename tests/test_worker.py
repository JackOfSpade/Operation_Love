"""Worker + OpenerService tests with fakes — no Playwright/emulator/SDK/network."""
import threading
import time

import pytest

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
        # Recorded on every like() call, alongside self.likes -- see like()'s own comment
        # for why these need their own history rather than folding into self.likes (which
        # stays a bare list of opener texts so every pre-existing `driver.likes == [...]`
        # assertion in this file keeps working unchanged).
        self.like_item_indexes = []
        self.like_anchored_openers = []

    def open_session(self): self.opened = True
    def next_profile(self):
        if self.i >= len(self.cards):
            return None
        p = self.cards[self.i]
        self.i += 1
        return p
    # anchored_opener: the driver's repair-hatch callback for when item_index turns out wrong
    # AT SWIPE TIME (see worker.py's _anchored_opener docstring). FakeDriver never invokes it
    # itself -- it only RECORDS what it received, exactly like self.likes/self.like_item_indexes
    # -- so a test can both assert whether one was handed over and call it directly to exercise
    # its behavior (see the ANCHORED_OPENER tests near the end of this file).
    def like(self, opener=None, item_index=0, *, anchored_opener=None):
        self.likes.append(opener)
        self.like_item_indexes.append(item_index)
        self.like_anchored_openers.append(anchored_opener)
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

    def like(self, opener=None, item_index=0, *, anchored_opener=None):
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
        self.calls = 0
        self.cost_tokens = cost_tokens
        self.should_stops = []   # records should_stop from every call -- see BUG 1's tests
        self.anchors = []        # records anchor from every call -- see the ANCHORED_OPENER tests
    def generate(self, profile, style, retry_hint="", *, anchor=None, should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        self.should_stops.append(should_stop)
        self.anchors.append(anchor)
        return OpenerResult(opener=f"hi {self.calls}", referenced="r",
                            usage=Usage(input_tokens=self.cost_tokens), model="gemini-test-model")


class SlowOpenerClient(FakeOpenerClient):
    def generate(self, profile, style, retry_hint="", *, anchor=None, should_stop=None,
                 skip_models=frozenset()):
        time.sleep(0.05)
        return super().generate(profile, style, retry_hint, anchor=anchor, should_stop=should_stop)


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
    def generate(self, profile, style, retry_hint="", *, anchor=None, should_stop=None,
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
    def generate(self, profile, style, retry_hint="", *, anchor=None, should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise OpenerError("Gemini opener: photo index 0 could not be decoded")


class BadRequestOpenerClient:
    """A single HTTP 400 -- below _BAD_REQUEST_LATCH_THRESHOLD, so this alone must not
    disable the service, only skip this one profile's opener."""
    def __init__(self):
        self.calls = 0
    def generate(self, profile, style, retry_hint="", *, anchor=None, should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise GeminiAPIError(400, "INVALID_ARGUMENT", "bad image or request")


class TransientOpenerClient:
    """An unclassified exception (timeout/connection blip) -- below
    _TRANSIENT_LATCH_THRESHOLD, so this alone must not disable the service."""
    def __init__(self):
        self.calls = 0
    def generate(self, profile, style, retry_hint="", *, anchor=None, should_stop=None,
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


def test_pace_rejects_negative_swipe_delay_for_both_calibrated_and_flat_branches(monkeypatch):
    """config.validate() already rejects a negative pacing.swipe_delay_s on every path
    that goes through it, but Worker is a public class any caller can construct
    directly with a pacing object that skipped validation. Event.wait() on a negative
    timeout returns immediately, so an unguarded negative anchor would silently
    produce machine-speed swiping -- fail loudly instead, the same way the calibrated
    branch already does today via post_action_delay_s's own `scale < 0` check. Pin
    BOTH branches (flat/non-calibrated and calibrated) since the guard is meant to
    cover _pace as a whole, before either branch-specific code path runs."""
    class _NegativePacing:
        swipe_delay_s = -1.0

    flat_worker = _worker(FakeDriver(0), FakeDecider("like"), FakeOpenerClient(), FakeStore())
    flat_worker.pacing = _NegativePacing()
    with pytest.raises(ValueError, match="non-negative"):
        flat_worker._pace("like")

    calibrated_worker = _worker(_CalibratedDriver(0), FakeDecider("like"),
                                FakeOpenerClient(), FakeStore())
    calibrated_worker.pacing = _NegativePacing()
    with pytest.raises(ValueError, match="non-negative"):
        calibrated_worker._pace("like")


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


class ScriptedScoreDecider:
    """Always returns 'like', with a scripted per-call score in order -- lets a test
    drive the ratio-ceiling shaper (worker.py's _auto_loop, via RateLimiter.
    allow_like_ratio in limits.py) through a specific marginal/strong score sequence,
    which the old constant-score FakeDecider can't do."""
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def decide(self, profile):
        score = self.scores[self.calls]
        self.calls += 1
        return Decision(decision="like", score=score, embedding=[0.1, 0.2], source="ranker")


def test_like_ratio_ceiling_demotes_marginal_likes_but_lets_strong_ones_through():
    """The ratio ceiling is now SCORE-AWARE (see limits.py's allow_like_ratio
    docstring): once the running rate is at/above target_like_ratio, only a MARGINAL
    like (within 0.15 of the ranker's own threshold) is demoted to a pass -- a
    clearly strong like (>= threshold + 0.15) is let through even though that briefly
    overshoots the ratio. This replaces the earlier score-blind version: an audit ran
    the OLD shaper against real scores [0.80, 0.99, 0.81, 0.82] at
    target_like_ratio=0.5 and found it demoted the model's SINGLE MOST CONFIDENT call
    (0.99) purely because of arrival order, while keeping a weaker 0.80 liked -- the
    most anti-"purely model-driven" outcome the shaper could produce. It still
    demotes rather than halts (a soft shaper, not a hard cap -- max_likes_per_run is
    the separate hard cap, see test_worker_stops_at_per_run_like_budget above, which
    this change does not touch).

    With target_like_ratio=0.5 and threshold=0.5 (FakeDecider/ScriptedScoreDecider's
    implicit default, since neither exposes .model.threshold):
    card 1: score=0.55 (marginal), acted=0 -> no history yet    -> like.  liked=1 acted=1
    card 2: score=0.55 (marginal), 1/1=100% >= 50% -> MARGINAL  -> demoted to pass.
    card 3: score=0.95 (strong),   1/2=50%  >= 50% -> STRONG    -> survives, stays like.
    card 4: score=0.55 (marginal), 2/3=67%  >= 50% -> MARGINAL  -> demoted to pass.

    0.55 (not e.g. 0.51) is deliberately chosen so this test exercises ONLY the
    ratio-ceiling logic under test, not AutoSessionPolicy's own separate, independent
    contextual-threshold demotion that runs earlier in _auto_loop (d =
    self._auto_policy.apply_decision(d, profile).decision, BEFORE the ratio check):
    that policy can itself lift the effective bar up to base_threshold +
    max_threshold_lift = 0.5 + 0.045 = 0.545 (interaction.py's _THRESHOLD_MAX_LIFT),
    so a score of 0.51 could occasionally get demoted right there by pure chance
    (Worker doesn't inject a seeded rng into AutoSessionPolicy), producing a flaky
    test that has nothing to do with the ratio ceiling. 0.55 is always >= 0.545, so
    it always survives AutoSessionPolicy unconditionally, while still being < 0.65
    (threshold + _STRONG_LIKE_MARGIN) so the ratio ceiling itself still calls it
    marginal.
    """
    driver = FakeDriver(4)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    limiter = RateLimiter(target_like_ratio=0.5)
    decider = ScriptedScoreDecider([0.55, 0.55, 0.95, 0.55])
    Worker("bumble", driver, decider, svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=limiter).run()

    # All 4 cards processed — NOT halted.
    assert len(store.decisions) == 4
    assert driver.closed
    assert [d for _, d, _ in store.decisions] == ["like", "dislike", "like", "dislike"]
    # The strong 0.95 actually landed as a driver.like() call (with its opener), not
    # silently swallowed by the ceiling the way a marginal one is.
    assert driver.likes == ["hi 1", "hi 2"]


def test_ratio_ceiling_demotion_tags_a_distinguishable_source(monkeypatch):
    """Consistency with interaction.py's own contextual demotion (which tags
    f"{source}_contextual", see AutoSessionPolicy.apply_decision): the ratio-ceiling
    demotion must also retag .source instead of leaving it as the raw ranker source,
    so a future consumer of Decision.source can tell a shaped decision apart from a
    raw model one. No functional effect today -- Decision.source isn't read anywhere
    downstream and the stored row hardcodes source="auto" (see FakeStore.
    record_decision / the real store.record_decision call in worker.py) -- so this
    spies on the Decision objects the loop actually constructs rather than on the
    store, which cannot see .source at all."""
    from operation_love import worker as worker_mod
    from operation_love.ranker.decider import Decision as RealDecision

    created = []

    def spying_decision(decision, score, embedding, source):
        d = RealDecision(decision, score, embedding, source)
        created.append(d)
        return d

    monkeypatch.setattr(worker_mod, "Decision", spying_decision)

    driver = FakeDriver(2)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    limiter = RateLimiter(target_like_ratio=0.5)
    # 0.55, not 0.51: stays clear of AutoSessionPolicy's own independent demotion --
    # see the worked-numbers comment in
    # test_like_ratio_ceiling_demotes_marginal_likes_but_lets_strong_ones_through above.
    decider = ScriptedScoreDecider([0.55, 0.55])   # card 2: marginal, ratio at ceiling -> demoted
    Worker("bumble", driver, decider, svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=limiter).run()

    demoted = [d for d in created if d.decision == "dislike"]
    assert len(demoted) == 1
    assert demoted[0].source == "ranker_ratio_ceiling"


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


def test_observe_resync_is_visible_on_hub_and_console(capsys):
    """FINDING 11 regression: wait_for_decision returning None for a resync (the card
    changed but nothing corroborated a human decision) must not be invisible. Before the
    fix, worker.py:226-227 was a bare `continue` -- the only "nothing recorded" branch in
    _observe_loop that skipped both the console print AND the _stat(last_decision=...)
    call the no_photos/no_face/archive_failed siblings all make -- so the hub/console
    just cycled READY -> capturing -> READY with no explanation, indistinguishable from
    normal operation even under a systemic corroboration failure."""
    from operation_love.status import RunStatus

    class ResyncDriver(FakeDriver):
        def __init__(self):
            super().__init__(1)
            self._served = False
        def out_of_profiles(self):             # deck empties right after the one resync
            return self._served
        def current_profile(self):
            self._served = True
            return self.cards[0]
        def wait_for_decision(self, timeout=None, should_stop=None):
            return None                         # resync: card changed, nothing corroborated
        def render_busy(self, message=None):
            pass

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = ResyncDriver()
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert status.app_view("bumble")["app"]["last_decision"] == "resync"
    out = capsys.readouterr().out
    assert "resync" in out
    # a resync is "recapture, record nothing" -- the control flow itself must be untouched
    assert store.profiles == [] and store.labels == [] and store.decisions == []
    assert driver.closed


def test_observe_stop_during_wait_is_not_reported_as_a_resync(capsys):
    """Second defect surfaced while root-causing the resync bug above: wait_for_decision's
    None return covers SEVERAL different situations (see base.py's docstring), not just a
    resync -- and one of them is the operator's OWN Stop firing mid-wait. `should_stop`
    passed into wait_for_decision is self.stop_event.is_set, so self.stop_event being set
    when None comes back means THIS None is the Stop click, not anything the driver
    observed on screen. Before the fix, that case was reported exactly like a genuine
    resync: a real bug report's run ended on a plain manual Stop, and the final console
    line still read "Card changed without a corroborated decision (resync)" -- a developer
    reading that log reasonably went looking for a perception bug that had never happened
    at that moment. A Stop is the operator's own action, not evidence about the screen, so
    it must produce neither the resync print nor last_decision="resync"."""
    from operation_love.status import RunStatus

    stop_event = threading.Event()

    class StopDuringWaitDriver(FakeDriver):
        def __init__(self):
            super().__init__(1)
        def current_profile(self):
            return self.cards[0]
        def wait_for_decision(self, timeout=None, should_stop=None):
            stop_event.set()                    # the operator's own Stop click, mid-wait
            return None
        def render_busy(self, message=None):
            pass

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = StopDuringWaitDriver()
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider(), svc, store, "run1", _Pacing(),
           stop_event, mode="observe", status=status).run()

    out = capsys.readouterr().out
    assert "resync" not in out
    assert status.app_view("bumble")["app"]["last_decision"] is None
    # a Stop is "recapture, record nothing" too -- same as a genuine resync
    assert store.profiles == [] and store.labels == [] and store.decisions == []
    assert driver.closed


def test_observe_stop_during_capture_is_silent_and_records_nothing(capsys):
    """The reported bug: "when I hit stop, it doesn't stop while it's reading a profile, it
    completes the read (by scrolling a bunch) then stops."

    A capture is now handed the stop signal and can return None the moment Stop lands (see
    DatingAppDriver.supports_interruptible_capture). That None arrives on a DIFFERENT branch
    from the Stop-during-wait case above -- `if profile is None: continue` -- and this pins
    what that branch must do: print nothing, stamp no last_decision (an abandoned read is not
    a resync and not a decision), write nothing to the store, and end the run in the terminal
    'stopped' state rather than leaving the hub showing 'capturing' forever."""
    from operation_love.status import RunStatus

    stop_event = threading.Event()

    class StopDuringCaptureDriver(FakeDriver):
        supports_interruptible_capture = True

        def __init__(self):
            super().__init__(1)
            self.stop_seen = None

        def current_profile(self, *, should_stop=None):
            stop_event.set()                     # Stop lands mid-read...
            self.stop_seen = should_stop() if should_stop else None
            return None                          # ...so the driver abandons the read

        def render_busy(self, message=None):
            pass

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = StopDuringCaptureDriver()
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider(), svc, store, "run1", _Pacing(),
           stop_event, mode="observe", status=status).run()

    assert driver.stop_seen is True               # the driver really was given a live stop signal
    out = capsys.readouterr().out
    assert "resync" not in out
    assert status.app_view("bumble")["app"]["last_decision"] is None
    assert status.app_view("bumble")["app"]["state"] == "stopped"
    assert store.profiles == [] and store.labels == [] and store.decisions == []
    assert driver.closed


def test_capture_stop_signal_is_withheld_from_drivers_that_do_not_declare_support(capsys):
    """Gating, not unconditional passing: a driver (or the many lightweight test doubles, or
    tools/hinge_inspect.py) that never declares the capability must keep receiving a
    zero-argument capture call. Passing should_stop to those would be an immediate TypeError
    on a live run -- the failure mode this flag exists to prevent."""
    stop_event = threading.Event()

    class LegacyObserveDriver(FakeDriver):
        def __init__(self):
            super().__init__(1)
            self.calls = 0

        def current_profile(self):                # deliberately zero-arg, as most doubles are
            self.calls += 1
            stop_event.set()
            return None

        def render_busy(self, message=None):
            pass

    driver = LegacyObserveDriver()
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider(), svc, store, "run1", _Pacing(),
           stop_event, mode="observe").run()

    assert driver.calls == 1                      # called, and called without arguments
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
    (supports_observe_like_intent) and calls it exactly like the real driver does -- TWO
    positional args, active then anchor -- so _wait_for_observed_decision's on_like_intent
    callback (the one that calls maybe_opener()) actually runs with a real-shaped anchor,
    not the implicit anchor=None an old-style single-arg call would fall back to.

    `anchor` defaults to a nonempty sentinel, matching the common case where the driver DID
    manage to capture the live like-sheet frame; pass anchor=None to model the rarer case
    where it could not (see the ANCHORED_OPENER/ANCHOR test sections for both)."""
    supports_observe_like_intent = True
    accepts_opener = True

    def __init__(self, anchor=b"the-actual-like-sheet-frame"):
        super().__init__(1)
        self._served = False
        self.anchor = anchor

    def out_of_profiles(self):
        return self._served

    def current_profile(self):
        self._served = True
        return self.cards[0]

    def render_busy(self, message=None):
        pass

    def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
        if on_like_intent is not None:
            on_like_intent(True, self.anchor)   # simulate opening Hinge's like/comment sheet
        return True                              # then LIKE


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
    """Records every maybe_opener() call's kwargs (anchor, advisory, should_stop) and, if
    `status` is supplied, the app's live state AT THE MOMENT of the call -- this is what
    change B's tests use to prove the interim 'suggesting' state is actually up while the
    call is happening, not just before/after it. `raise_exc`, if set, makes the call raise
    instead of returning a pick -- exercising the exception path of the try/except in
    _wait_for_observed_decision's on_like_intent. `referenced`, if set, is echoed onto the
    returned OpenerPick -- see the ANCHOR tests, which check it lands in AppStatus verbatim."""
    stop_requested = False
    disabled = False

    def __init__(self, *, status=None, app=None, raise_exc=None, suggestion="hi", referenced=""):
        self.status = status
        self.app = app
        self.raise_exc = raise_exc
        self.suggestion = suggestion
        self.referenced = referenced
        self.calls = []
        self.state_during_call = None

    def maybe_opener(self, run_id, app, profile, *, anchor=None, should_stop=None,
                     advisory=False):
        self.calls.append({"run_id": run_id, "app": app, "anchor": anchor,
                           "should_stop": should_stop, "advisory": advisory})
        if self.status is not None:
            self.state_during_call = self.status.app_view(self.app)["app"]["state"]
        if self.raise_exc is not None:
            raise self.raise_exc
        return OpenerPick(self.suggestion, index=0, referenced=self.referenced)


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


# ---------------------------------------------------------------------------------------
# ANCHOR: the fix for openers arriving out of place -- e.g. a suggestion about a lake photo
# shown while the comment is actually going to land under a dining-table photo, because the
# model was never told which item the human's tap opened a comment sheet for (see
# _wait_for_observed_decision's own ANCHOR docstring paragraph in worker.py). The driver hands
# the live like-sheet frame to on_like_intent(active=True, anchor=...); it must be forwarded
# VERBATIM into maybe_opener(anchor=...), the resulting opener_referenced/opener_anchored must
# reach AppStatus alongside the suggestion, anchor=None must still produce a suggestion (never
# crash or stall), and the dismiss path (active=False) must reset all three published fields.
# ---------------------------------------------------------------------------------------

def test_observe_forwards_the_live_like_sheet_anchor_verbatim_into_maybe_opener():
    """THE regression test for the reported bug: an opener about a lake photo got suggested
    while the human's tap actually opened a comment sheet anchored to a dining-table photo,
    because the suggestion call never told the model which item Hinge's sheet was actually
    showing. The driver's on_like_intent(active=True, anchor=<frame>) must forward that EXACT
    frame into maybe_opener(anchor=...) -- not None, not some other bytes -- so the opener is
    written about what the human is actually looking at right now, not whatever the model
    happens to like best from the whole profile scroll."""
    driver = _ObserveLikeIntentDriver(anchor=b"the-actual-like-sheet-frame")
    store = FakeStore()
    svc = _RecordingOpenerService()
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe").run()

    assert svc.calls and svc.calls[0]["anchor"] == b"the-actual-like-sheet-frame"


def test_observe_suggestion_publishes_referenced_and_anchored_true_alongside_the_text():
    """The hub must show not just the suggested text but WHAT it claims to be about
    (opener_referenced) and WHETHER it was actually grounded in the live like-sheet frame
    (opener_anchored) -- this is what lets the operator catch a mismatch at a glance instead
    of trusting every suggestion blindly (see AppStatus.opener_anchored's docstring for the
    exact out-of-place-caption bug this exists to make visible)."""
    from operation_love.status import RunStatus

    driver = _ObserveLikeIntentDriver(anchor=b"frame-bytes")
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="loved your trail photo",
                                  referenced="the trail photo")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    waiting_for_send_call = next(c for c in calls if c.get("state") == "waiting_for_send")
    assert waiting_for_send_call["opener_suggestion"] == "loved your trail photo"
    assert waiting_for_send_call["opener_referenced"] == "the trail photo"
    assert waiting_for_send_call["opener_anchored"] is True


def test_observe_suggestion_with_no_anchor_still_suggests_and_marks_it_unanchored():
    """A driver can't always capture the live like-sheet frame (e.g. a screencap glitch) --
    anchor=None must stay a valid, non-crashing, non-stalling path: the suggestion is still
    generated (written blind, exactly as it always behaved before the anchor fix), but
    opener_anchored must publish False so the operator can tell a blind suggestion apart from
    a grounded one, rather than trusting every suggestion equally regardless of whether the
    model actually knew what it was writing about."""
    from operation_love.status import RunStatus

    driver = _ObserveLikeIntentDriver(anchor=None)
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="hey there")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert svc.calls and svc.calls[0]["anchor"] is None       # driver truly had nothing to hand over
    waiting_for_send_call = next(c for c in calls if c.get("state") == "waiting_for_send")
    assert waiting_for_send_call["opener_suggestion"] == "hey there"   # still suggested -- no stall
    assert waiting_for_send_call["opener_anchored"] is False


def test_observe_dismissing_the_like_sheet_publishes_cleared_opener_fields():
    """The sheet can close WITHOUT a send (dismissed) just as it can after one; on_like_intent
    (active=False, anchor=None) must explicitly clear opener_suggestion/opener_referenced/
    opener_anchored in that SAME status.set_app call, not rely on some later, unrelated state
    transition to do it for it -- see status.py's own "safe default" comment: a stale "about
    her trail photo" caption (or a stale anchored=True badge) surviving onto whatever gets
    shown next is worse than showing nothing at all."""
    from operation_love.status import RunStatus

    class DismissThenLikeDriver(_ObserveLikeIntentDriver):
        def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
            if on_like_intent is not None:
                on_like_intent(True, self.anchor)    # sheet opens, suggestion generated
                on_like_intent(False, None)           # sheet closes (dismissed, not sent)
            return True

    driver = DismissThenLikeDriver()
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="loved your trail photo", referenced="the trail")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    # Only the on_like_intent(active=False) call publishes state="waiting" together with
    # opener_referenced -- the loop's own pre-wait _stat(state="waiting", opener_suggestion=
    # None) call (published before wait_for_decision is even called) never mentions
    # opener_referenced, so this filter isolates exactly the call under test.
    dismiss_call = next(c for c in calls if c.get("state") == "waiting" and "opener_referenced" in c)
    assert dismiss_call["opener_suggestion"] is None
    assert dismiss_call["opener_referenced"] is None
    assert dismiss_call["opener_anchored"] is False


# ---------------------------------------------------------------------------------------
# CROSS-CARD LEAK: an adversarial review found two worker.py call sites (the per-card reset
# below, and _observe_loop's own `finally`) that pass opener_suggestion=None explicitly --
# which, per status.py's RunStatus.set_app, opts them OUT of the field's own auto-clear
# safety net (that net only fires when opener_suggestion is ABSENT from the update dict).
# Both sites were fixed to also name opener_referenced/opener_anchored, but nothing pinned
# the invariant, which is why the gap survived. These two tests check the RESULTING
# AppStatus snapshot right after the specific call under test (not just the kwargs worker.py
# happened to pass in) -- checking only the passed-in dict would not actually catch a
# regression back to the buggy two-field call: dict.get("opener_referenced") on a MISSING key
# also returns None, silently matching the correct expectation for the wrong reason.
# ---------------------------------------------------------------------------------------

def _record_calls_and_snapshots(status, app):
    """Like _record_state_transitions, but also captures a live app_view() snapshot
    immediately after each set_app call actually applies. A test can then assert on the
    RESULTING AppStatus fields for one specific call, which is what the hub would actually
    render at that moment -- not merely restate the kwargs worker.py passed in (see the
    CROSS-CARD LEAK section comment above for why that weaker check would not catch the
    regression these tests exist to pin)."""
    calls = []
    original = status.set_app
    def recording(app_, **fields):
        original(app_, **fields)
        calls.append({"fields": dict(fields), "after": status.app_view(app)["app"]})
    status.set_app = recording
    return calls


class _ResyncThenSilentSecondCardDriver(FakeDriver):
    """Card 1 opens Hinge's comment sheet (on_like_intent(active=True, ...) -- a suggestion is
    generated, publishing opener_referenced/opener_anchored=True), but wait_for_decision then
    resolves to None -- a RESYNC. This mirrors the real driver: hinge.py's wait_for_decision
    can return None right after notifying on_like_intent(active=True, ...) (e.g. `if sent is
    None: return None`, reached before any matching close notice), and _observe_loop's own
    resync branch (`continue`, see its "Card changed without a corroborated decision" comment)
    does not touch the opener fields either -- so nothing clears the stale suggestion until
    the per-card reset at the top of the loop's NEXT iteration, which is exactly what these
    tests target. Card 2 never opens a sheet at all (on_like_intent is never called for it) --
    the "next card's suggestion fails or is absent" half of the reported failure mode."""
    supports_observe_like_intent = True
    accepts_opener = True

    def __init__(self):
        super().__init__(2)
        self.served = 0
        self.wait_calls = 0

    def out_of_profiles(self):
        return self.served >= 2

    def current_profile(self):
        card = self.cards[self.served]
        self.served += 1
        return card

    def render_busy(self, message=None):
        pass

    def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
        self.wait_calls += 1
        if self.wait_calls == 1:
            if on_like_intent is not None:
                on_like_intent(True, b"card-one-like-sheet-frame")   # sheet opens, suggestion made
            return None                                              # resync -- no close notice
        return True                                                  # card 2: plain LIKE, no sheet


def test_observe_per_card_reset_clears_a_prior_cards_referenced_and_anchored_fields():
    """THE regression test named in the task: after a card produces an anchored suggestion
    (opener_referenced set, opener_anchored=True) that is never explicitly closed (a resync,
    not a dismiss or a send), the per-card reset for the NEXT card must still clear all three
    opener fields -- not just opener_suggestion. Because that reset call explicitly names
    opener_suggestion, it opts itself out of RunStatus.set_app's own auto-clear (see
    status.py's comment on the safety net), so it must name opener_referenced/opener_anchored
    too, or the hub would keep showing card 1's "about: her dog" caption and anchored badge
    while card 2 has no suggestion of its own at all."""
    from operation_love.status import RunStatus

    driver = _ResyncThenSilentSecondCardDriver()
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    calls = _record_calls_and_snapshots(status, "bumble")
    svc = _RecordingOpenerService(suggestion="about her dog", referenced="her dog")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    reset_calls = [c for c in calls
                   if c["fields"].get("state") == "waiting" and "opener_suggestion" in c["fields"]]
    assert len(reset_calls) >= 2      # the per-card reset fires once per card, including card 2's
    card_two_reset = reset_calls[1]["after"]
    assert card_two_reset["opener_suggestion"] is None
    assert card_two_reset["opener_referenced"] is None    # NOT left over from card 1's "her dog"
    assert card_two_reset["opener_anchored"] is False      # NOT left over from card 1's True


def test_observe_loop_finally_clears_a_still_live_suggestion_before_the_terminal_state():
    """THE regression test for _observe_loop's `finally`: if the run ends (Stop clicked, in
    this test right after the suggestion for the only card is published) while a suggestion
    is still live in AppStatus, the finally block's own clearing call -- which, like the
    per-card reset, explicitly names opener_suggestion and therefore also opts out of
    RunStatus.set_app's auto-clear -- must name opener_referenced/opener_anchored too.
    Snapshotted right after THAT specific call (identified by opener_suggestion being named
    with no accompanying `state` key -- the unique signature of this one call site in
    worker.py) rather than the final post-run() snapshot, because _finish_session's own
    state=stopped/out_of_profiles transition runs immediately afterward and would
    coincidentally re-clear the fields through the ORDINARY auto-clear path regardless of
    whether this call did its job -- masking the exact bug an adversarial review found here."""
    from operation_love.status import RunStatus

    stop_event = threading.Event()

    class _StopRightAfterSuggestionDriver(_ObserveLikeIntentDriver):
        def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
            result = super().wait_for_decision(timeout=timeout, should_stop=should_stop,
                                               on_like_intent=on_like_intent)
            stop_event.set()          # Stop lands the instant the sheet's suggestion is up
            return result

    driver = _StopRightAfterSuggestionDriver(anchor=b"the-actual-like-sheet-frame")
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    calls = _record_calls_and_snapshots(status, "bumble")
    svc = _RecordingOpenerService(suggestion="about her dog", referenced="her dog")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           stop_event, mode="observe", status=status).run()

    finally_clear = next(c for c in calls
                         if "opener_suggestion" in c["fields"] and "state" not in c["fields"])
    assert finally_clear["after"]["opener_suggestion"] is None
    assert finally_clear["after"]["opener_referenced"] is None
    assert finally_clear["after"]["opener_anchored"] is False


class _TwoCardObserveLikeIntentDriver(FakeDriver):
    """Like _ObserveLikeIntentDriver, but serves TWO cards, each with a LIKE outcome that
    opens (and closes) Hinge's comment sheet -- lets a test drive maybe_opener() twice, once
    per profile, to prove a failure on card 1 doesn't poison card 2. Uses the real driver's
    two-arg on_like_intent contract (active, anchor) on both the open and the close call,
    matching what the real Hinge driver actually does."""
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
            on_like_intent(True, b"like-sheet-frame")
            on_like_intent(False, None)
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


# ---------------------------------------------------------------------------------------
# ANCHORED_OPENER: the driver's repair hatch for when item_index turns out wrong AT SWIPE
# TIME -- the deck can scroll/reorder between when the opener was written and when the driver
# actually taps the heart, so the item the driver lands on may not be the one the opener text
# is about (see worker.py's _anchored_opener docstring). The AUTO-loop must hand the driver a
# callable that re-asks maybe_opener() with the live like-screen frame and returns fresh text
# grounded in it -- or refuses (returns None) rather than ship a mismatched comment -- and must
# withhold the callable entirely whenever there is no opener for it to repair in the first
# place (app doesn't accept openers / no opener_service / no opener was generated this call).
# ---------------------------------------------------------------------------------------

class _SequencedOpenerService:
    """Records every maybe_opener() call's kwargs and returns a SCRIPTED pick per call, in
    call order -- lets a test drive the AUTO-loop's initial opener call and the driver's later
    anchored_opener repair re-ask through two DIFFERENT results, so it can tell which call
    produced which text. A scripted entry of None models a repair re-ask that itself fails
    (e.g. the service exhausted between the first call and the repair attempt) -- the driver
    must then refuse to send rather than fall back to the stale, possibly-mismatched text."""
    stop_requested = False
    disabled = False

    def __init__(self, picks):
        self.picks = list(picks)
        self.calls = []

    def maybe_opener(self, run_id, app, profile, *, anchor=None, should_stop=None,
                     advisory=False):
        self.calls.append({"run_id": run_id, "app": app, "anchor": anchor,
                           "should_stop": should_stop, "advisory": advisory})
        return self.picks[len(self.calls) - 1]


def test_auto_like_receives_an_anchored_opener_callback_when_an_opener_was_generated():
    """The AUTO-loop must hand driver.like() a callable anchored_opener whenever it generated
    an opener for this like -- this is the repair hatch the driver can invoke later if it
    can't land the comment on the item the opener was written about. Also pins item_index:
    it must be the SAME index OpenerPick reported, not always 0."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = _SequencedOpenerService([OpenerPick("hi 1", index=2, referenced="r")])
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.likes == ["hi 1"]
    assert driver.like_item_indexes == [2]
    assert len(driver.like_anchored_openers) == 1
    assert callable(driver.like_anchored_openers[0])


def test_anchored_opener_callback_reasks_maybe_opener_with_the_given_anchor_and_returns_new_text():
    """Invoking the callback the driver received with a frame of bytes must call maybe_opener()
    AGAIN, this time with anchor=<those exact bytes>, and return whatever new text comes back
    -- this is how the driver repairs an opener that turned out to be written about the wrong
    item once the like screen actually opened somewhere else than expected."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = _SequencedOpenerService([
        OpenerPick("hi 1", index=0, referenced="r"),
        OpenerPick("hi 2 anchored", index=1, referenced="r2"),
    ])
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    anchored_opener = driver.like_anchored_openers[0]
    result = anchored_opener(b"live-like-screen-frame")

    assert result == "hi 2 anchored"
    assert len(svc.calls) == 2                                    # the original call + the repair re-ask
    assert svc.calls[1]["anchor"] == b"live-like-screen-frame"     # forwarded verbatim
    assert svc.calls[1]["advisory"] is False                       # unchanged AUTO-mode behavior


def test_anchored_opener_callback_returns_none_when_the_reask_fails():
    """THE OWNER'S RULE applies here too: a comment written about the wrong item is not a
    comment that can be sent, so a failed repair re-ask must make the driver refuse to send
    (return None) rather than silently falling back to the original, mismatched opener text."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = _SequencedOpenerService([
        OpenerPick("hi 1", index=0, referenced="r"),
        None,                                          # the repair re-ask itself fails
    ])
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    anchored_opener = driver.like_anchored_openers[0]
    assert anchored_opener(b"live-like-screen-frame") is None


def test_anchored_opener_callback_is_none_when_the_app_does_not_accept_openers():
    """Bumble-style driver: no swipe-time opener at all, so there is nothing for a repair
    callback to ever repair -- the opener service must not even be called, and the driver
    must receive anchored_opener=None, not a callable that would immediately be irrelevant."""
    driver = FakeDriver(1)
    driver.accepts_opener = False
    store = FakeStore()
    svc = _SequencedOpenerService([OpenerPick("hi 1")])   # never actually reached
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.likes == [None]
    assert driver.like_anchored_openers == [None]
    assert svc.calls == []                                # opener service never even called


def test_anchored_opener_callback_is_none_when_opener_service_is_none():
    """Worker is a public class any caller can construct with opener_service=None (see the
    C section above) -- the anchored_opener wiring must degrade the same way as the plain
    opener wiring does: no callable handed to the driver, no crash."""
    driver = FakeDriver(1)
    store = FakeStore()
    Worker("bumble", driver, FakeDecider("like"), None, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.likes == [None]
    assert driver.like_anchored_openers == [None]


def test_anchored_opener_callback_is_none_when_no_opener_was_generated():
    """opener.enabled=false in config produces an OpenerService(client=None, ...) -- `disabled`
    from construction (see OpenerService.__init__), so maybe_opener() always returns None and
    there is no opener text a repair callback could ever improve on. The driver must receive
    anchored_opener=None here too, not a callable wrapping a pick that doesn't exist."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert driver.likes == [None]
    assert driver.like_anchored_openers == [None]
