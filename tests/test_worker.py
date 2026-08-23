"""Worker + OpenerService tests with fakes — no Playwright/emulator/SDK/network."""
import math
import threading
import time

import pytest

from operation_love.costing import CostTracker, ModelPricing, Usage
from operation_love.drivers.base import (ActionCancelled, DatingAppDriver, DeckBlockedError,
                                        DriverClosed, ItemTargetingError,
                                        OBSERVE_ITEM_INCONCLUSIVE, OBSERVE_ITEM_MATCH,
                                        OBSERVE_ITEM_MISMATCH, ObserveItemCheck)
from operation_love.opener.opener import (
    FIRST_ITEM_INDEX,
    INDEX_SPACE_MODEL_ITEMS,
    INDEX_SPACE_PROFILE_PHOTOS,
    ITEM_INDEX_ABSENT,
    GeminiAPIError,
    OpenerError,
    OpenerParseError,
    OpenerResult,
    REASON_PROMPT_BLOCKED,
)
from operation_love.opener.service import OpenerPick, OpenerService
from operation_love.limits import RateLimiter
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import Decision
from operation_love.worker import Worker

PRICING = {"gemini-test-model": ModelPricing(input=5.0, output=25.0)}

# Liveness bound, not a performance bound: it exists only so a genuine hang fails these tests
# instead of hanging the whole suite forever. Widened 2026-08-22 when `python -m pytest` moved
# to one worker per core (pyproject.toml addopts `-n auto --dist loadgroup`), which measured a
# ~15x slowdown (0.33s idle vs 5.06s under load) on tests/test_concurrency.py's positive
# liveness waits of the same shape. Nothing about the property under test (did the background
# thread reach the expected state / finish?) depends on the exact number, so widening it loses
# nothing.
_LIVENESS_TIMEOUT_S = 15.0


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
        self.like_model_item_indexes = []

    def open_session(self): self.opened = True
    def next_profile(self):
        if self.i >= len(self.cards):
            return None
        p = self.cards[self.i]
        self.i += 1
        return p
    # No `anchored_opener` keyword, deliberately: the driver's repair hatch -- a callback that
    # re-asked the model for text about whatever item the tap had actually landed on -- was
    # removed on 2026-08-12 (ops/OPENER-REDESIGN.md 5.6: rewriting the opener to match whatever
    # we hit is a substitution, and a targeting miss is now a stop). This signature tracks
    # base.Driver.like exactly, so a worker that started passing one again would fail here.
    def like(self, opener=None, item_index=None, *, model_item_index=None):
        self.likes.append(opener)
        self.like_item_indexes.append(item_index)
        # The OTHER index space, recorded separately for the same reason `like_item_indexes` is
        # separate from `likes`: since doc 5.5's counting navigation was wired the worker hands
        # a model item number straight through as `model_item_index` rather than converting it,
        # and a test that cannot see which of the two arguments was used cannot tell a navigated
        # like from a capture-order one.
        self.like_model_item_indexes.append(model_item_index)
    def dislike(self): self.dislikes += 1
    def out_of_profiles(self): return self.i >= len(self.cards)
    def close(self): self.closed = True


class InterruptibleLikeDriver(FakeDriver):
    supports_interruptible_like_navigation = True

    def __init__(self, n):
        super().__init__(n)
        self.like_stop_callbacks = []

    def like(self, opener=None, item_index=None, *, model_item_index=None, should_stop=None):
        self.like_stop_callbacks.append(should_stop)
        return super().like(opener, item_index, model_item_index=model_item_index)


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

    def like(self, opener=None, item_index=None, *, model_item_index=None):
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
        self.items = []          # records items from every call -- doc 5.2's crop request shape
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        self.should_stops.append(should_stop)
        self.items.append(items)
        # item_index/index_space: a REALISTIC pick (the legacy frame shape's first item, in
        # the translatable profile-photos space), not the OpenerResult dataclass's own
        # ITEM_INDEX_ABSENT/INDEX_SPACE_MODEL_ITEMS defaults. Those defaults exist so a
        # careless construction fails safe (see OpenerResult.index_space's own docstring),
        # but "fails safe" now means the AUTO-loop refuses to call driver.like() at all (see
        # the INDEX SPACES section below) -- which would silently turn every test in this
        # file that just wants "a like happens, with an opener" into an untargeted-item stop
        # unrelated to what it's actually testing. Tests that DO want an absent/untranslatable
        # index construct their own OpenerPick directly (see _SequencedOpenerService) instead
        # of going through this fake.
        return OpenerResult(opener=f"hi {self.calls}", referenced="r",
                            usage=Usage(input_tokens=self.cost_tokens), model="gemini-test-model",
                            item_index=FIRST_ITEM_INDEX, index_space=INDEX_SPACE_PROFILE_PHOTOS)


class SlowOpenerClient(FakeOpenerClient):
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        time.sleep(0.05)
        return super().generate(profile, style, retry_hint, items=items,
                                should_stop=should_stop)


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
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
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
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise OpenerError("Gemini opener: photo index 0 could not be decoded")


class SafetyBlockedOpenerClient:
    """Gemini accepted the call but blocked this profile's content before generation."""
    def __init__(self):
        self.calls = 0
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise OpenerParseError(
            "Gemini blocked the opener prompt before generating content "
            "(promptFeedback.blockReason=PROHIBITED_CONTENT)",
            Usage(input_tokens=10), "gemini-test-model",
            reason_code=REASON_PROMPT_BLOCKED)


class BadRequestOpenerClient:
    """A single HTTP 400 -- below _BAD_REQUEST_LATCH_THRESHOLD, so this alone must not
    disable the service, only skip this one profile's opener."""
    def __init__(self):
        self.calls = 0
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise GeminiAPIError(400, "INVALID_ARGUMENT", "bad image or request")


class TransientOpenerClient:
    """An unclassified exception (timeout/connection blip) -- below
    _TRANSIENT_LATCH_THRESHOLD, so this alone must not disable the service."""
    def __init__(self):
        self.calls = 0
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise RuntimeError("connection reset")


class FakeStore:
    def __init__(self):
        self.decisions, self.labels, self.profiles, self.spend, self.openers = [], [], [], [], []
        self.rejections = []
    def load_labels(self): return []
    def record_profile(self, run_id, app, profile_id, liked, source="manual", **k):
        self.profiles.append((app, profile_id, liked, k))
        return True
    def add_label(self, run_id, app, liked, embedding, source="manual", profile_id="", **k):
        self.labels.append((app, profile_id, liked, k))
    def record_decision(self, run_id, app, decision, score, source="auto", **_):
        self.decisions.append((app, decision, source))
    # Signature mirrors ranker/store.py's real record_opener EXACTLY, both trailing
    # parameters included: angle="" (doc 3.4/3.5: the model's free-text account of what its
    # opener is doing) and item_description="" (doc 5.7: the model's description of the ITEM
    # it picked to write about and to like). Deliberately spelled out rather than *a:
    # the service persists inside a non-fatal try/except, so a fake that drifts from the
    # real signature does not raise here -- it prints "Warning: failed to persist opener
    # spend record" and leaves self.openers empty, which surfaces as an unrelated-looking
    # count assertion failing further down. Keeping the parameters explicit means the
    # mismatch is at least visible in the traceback when it happens again.
    def record_opener(self, run_id, app, model, opener, referenced, angle="",
                      item_description="", **_):
        self.openers.append(opener)
    def record_opener_rejection(self, run_id, app, model, attempt, reason_code, reason, raw_opener):
        self.rejections.append((app, model, attempt, reason_code, reason, raw_opener))
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


def test_worker_binds_opt_in_driver_debug_to_its_exact_run_id_before_opening():
    class BindingDriver(FakeDriver):
        def __init__(self):
            super().__init__(0)
            self.bound_run_id = None

        def bind_debug_run(self, run_id):
            self.bound_run_id = run_id

    driver = BindingDriver()
    _worker(driver, FakeDecider(), None, FakeStore())._bind_debug_run()
    assert driver.bound_run_id == "run1"


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
        t.join(timeout=_LIVENESS_TIMEOUT_S)

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


@pytest.mark.parametrize("value", [-1.0, 0.5, math.nan, math.inf, 1e300, True, "3.5"])
def test_pace_rejects_invalid_direct_pacing_for_both_driver_branches(value):
    """Direct Worker construction must enforce the same finite 0-or-[1,3600] contract."""
    class _InvalidPacing:
        swipe_delay_s = value

    flat_worker = _worker(FakeDriver(0), FakeDecider("like"), FakeOpenerClient(), FakeStore())
    flat_worker.pacing = _InvalidPacing()
    with pytest.raises(ValueError, match="swipe_delay_s"):
        flat_worker._pace("like")

    calibrated_worker = _worker(_CalibratedDriver(0), FakeDecider("like"),
                                FakeOpenerClient(), FakeStore())
    calibrated_worker.pacing = _InvalidPacing()
    with pytest.raises(ValueError, match="swipe_delay_s"):
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


def test_auto_stages_then_commits_an_opener_only_after_like_returns():
    driver = FakeDriver(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert driver.likes == ["hi 1"]
    assert len(store.openers) == 1 and len(svc.recent_openers_snapshot()) == 1


def test_auto_persists_landed_decision_before_committing_its_staged_opener():
    """A durable opener must always have a preceding durable Like decision to join to.

    This ordering is the cleanup/reporting invariant: generation is only a draft, a returned
    driver.like is the physical boundary, and decision storage establishes the acted-on fact
    before the draft becomes an opener row.
    """
    events = []

    class OrderedStore(FakeStore):
        def record_decision(self, *args, **kwargs):
            events.append("decision")
            kwargs.pop("profile_id", None)
            kwargs.pop("created_at", None)
            return super().record_decision(*args, **kwargs)

        def record_opener(self, *args, **kwargs):
            events.append("opener")
            for key in ("profile_id", "decision", "decision_source", "decision_created_at",
                        "model_item_index"):
                kwargs.pop(key, None)
            return super().record_opener(*args, **kwargs)

    driver = FakeDriver(1)
    store = OrderedStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert events == ["decision", "opener"]
    assert store.decisions == [("bumble", "like", "auto")]
    assert store.openers == ["hi 1"]


def test_auto_keeps_legacy_opener_service_without_stage_keyword_compatible():
    """The stage/commit protocol is additive; an old injected service must not crash AUTO."""
    class LegacyService:
        disabled = False
        stop_requested = False

        def maybe_opener(self, run_id, app, profile, *, items=None, should_stop=None):
            return OpenerPick("legacy opener", index=1,
                              index_space=INDEX_SPACE_PROFILE_PHOTOS)

        def commit_opener(self, pick, **_):
            raise AssertionError("legacy service must not receive a staged commit")

    driver = FakeDriver(1)
    store = FakeStore()
    Worker("bumble", driver, FakeDecider("like"), LegacyService(), store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.likes == ["legacy opener"]
    assert store.decisions == [("bumble", "like", "auto")]


def test_auto_targeting_failure_leaves_staged_opener_uncommitted():
    driver = _TargetingMissDriver(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert store.openers == [] and svc.recent_openers_snapshot() == []


def test_auto_post_send_paywall_leaves_staged_opener_uncommitted():
    class _PaywallWithOpener(_BlockedAfterLikeDriver):
        accepts_opener = True
    driver = _PaywallWithOpener()
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert store.openers == [] and svc.recent_openers_snapshot() == []


def test_auto_generic_like_failure_leaves_staged_opener_uncommitted():
    driver = RaisingLikeDriver(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert store.openers == [] and svc.recent_openers_snapshot() == []


def test_auto_commits_staged_opener_when_stop_arrives_after_like_lands():
    stop = threading.Event()
    class _LandedThenStop(FakeDriver):
        def like(self, *args, **kwargs):
            super().like(*args, **kwargs)
            stop.set()
    driver = _LandedThenStop(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(), stop,
           mode="auto").run()
    assert driver.likes == ["hi 1"]
    assert len(store.openers) == 1 and len(svc.recent_openers_snapshot()) == 1


def test_worker_passes_stop_callback_only_to_interruptible_like_navigation():
    driver = InterruptibleLikeDriver(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, 5), store, "style")
    worker = _worker(driver, FakeDecider("like"), svc, store)
    worker.run()
    assert len(driver.like_stop_callbacks) == 1
    assert driver.like_stop_callbacks[0].__self__ is worker.stop_event
    assert driver.like_stop_callbacks[0]() is False


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
    # stop_kind disambiguates stop_reason's SOURCE now that the deck-blocked check (added
    # 2026-08-11, see the DECK_BLOCKED section below) also populates stop_reason -- an
    # OpenerService-triggered stop must publish "opener", not the new "deck_blocked".
    assert app["stop_kind"] == "opener"


# ---------------------------------------------------------------------------------------
# COMPLETED RULE: in AUTO mode on an opener-capable app, a like is normally sent WITH its
# opener or not sent at all. A provider safety block is the narrow exception: AUTO already
# chose Like, so it proceeds without a comment. There is no configuration that broadens that
# exception (the old
# budget.on_exhausted="swipe_without_opener" mode was removed outright; see service.py's
# module docstring). The tests above cover the pre-existing GLOBAL-exhaustion guard
# (stop_requested); these cover every other way maybe_opener() can return None -- a
# per-call failure that leaves OpenerService still enabled (see service.py's maybe_opener
# docstring and last_skip_reason), plus the case where a bad AI response keeps failing
# through every retry attempt and the SERVICE itself ends up exhausted. Each must halt
# before driver.like(), record no decision for the abandoned profile, and publish a
# stop_reason naming the actual cause (not a generic line). The dedicated safety-block test
# pins the opposite result for that one structured outcome.
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


def test_worker_auto_likes_without_comment_when_profile_content_is_safety_blocked():
    """The ranker already chose Like; an uncontrollable profile-content block removes only
    the optional comment and must not discard that decision or poison later profiles."""
    driver = FakeDriver(1)
    store = FakeStore()
    client = SafetyBlockedOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")

    Worker("hinge", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert client.calls == 1
    assert driver.likes == [None]
    assert driver.like_item_indexes == [None]
    assert driver.like_model_item_indexes == [None]
    assert store.decisions == [("hinge", "like", "auto")]
    assert store.openers == []
    assert svc.disabled is False and svc.stop_requested is False
    assert svc.last_skip_allows_commentless_like is True
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
    # See the matching stop_kind comment on
    # test_auto_mode_stop_reason_is_visible_in_status_when_opener_budget_exhausts above --
    # this is the OTHER opener-triggered stop path (retry exhaustion, not budget), and it
    # must publish the same "opener" stop_kind, not the new "deck_blocked".
    assert app["stop_kind"] == "opener"


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
    # Same stop_kind disambiguation as the auto-mode pins above -- an OpenerService stop
    # discovered mid-observe-session must publish "opener", not the new "deck_blocked".
    assert app["stop_kind"] == "opener"


# ---------------------------------------------------------------------------------------
# DECK_BLOCKED: worker.py's blocked-deck check covers Hinge's own "you're out of free likes for
# today" Hinge+ paywall. An unrecognised paywall can leave observe mode polling inside
# wait_for_decision(timeout=None) for a decision that cannot arrive. Both loops
# now call driver.blocked_reason() every iteration, BEFORE out_of_profiles() (the more
# specific, more actionable answer -- see worker.py's comment at each call site), and stop
# the run as state="blocked"/stop_kind="deck_blocked" the moment it returns a string. This
# is a GRACEFUL stop, not the HALT-on-unexpected error path: the phone is in a perfectly
# normal state, nothing is broken, and observe mode's passivity rule means the screen must
# be left exactly as found -- never tapped, swiped, or typed into to clear it.
# ---------------------------------------------------------------------------------------

_PAYWALL_REASON = "Hinge is out of free likes for today — the Hinge+ upgrade screen is up"


class _BlockedDriver(FakeDriver):
    """Reports a blocked deck (Hinge's out-of-likes paywall stand-in) from the very first
    loop iteration. `current_profile`/`wait_for_decision` raise if called at all -- the
    blocked check must short-circuit BEFORE any capture or decision-wait is even attempted,
    since the paywall is a purchase screen and observe mode must never touch it."""
    def __init__(self, n, reason=_PAYWALL_REASON):
        super().__init__(n)
        self.reason = reason
        self.blocked_calls = 0

    def blocked_reason(self):
        self.blocked_calls += 1
        return self.reason

    def current_profile(self, *, should_stop=None):
        raise AssertionError("must not capture a profile once the deck is reported blocked")

    def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
        raise AssertionError("must not wait for a decision once the deck is reported blocked")

    def render_busy(self, message=None):
        pass


def test_observe_mode_stops_as_blocked_when_driver_reports_a_blocked_deck():
    """THE regression test for THE INCIDENT's D2 fix: a driver whose blocked_reason()
    reports something on screen must stop observe mode immediately -- state="blocked",
    stop_reason=<the driver's own operator-facing sentence, verbatim>, stop_kind=
    "deck_blocked" -- and record NO decision/label, since nothing was ever captured or
    decided on. Before this check existed there was no bail-out at all; the paywall just
    hung the run until the owner pressed Stop by hand."""
    from operation_love.status import RunStatus

    driver = _BlockedDriver(1)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    Worker("bumble", driver, FakeDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "blocked"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert app["stop_kind"] == "deck_blocked"
    assert store.decisions == [] and store.labels == [] and store.profiles == []
    assert driver.closed


def test_auto_mode_stops_as_blocked_and_never_acts_when_driver_reports_a_blocked_deck():
    """Same incident, AUTO mode: a driver reporting a blocked deck must stop the same way
    (state="blocked", stop_kind="deck_blocked") AND -- the extra AUTO-specific requirement
    -- must never issue a single like/dislike. An autonomous loop that kept swiping past an
    unrecognized paywall would be the worse failure mode; this pins that it doesn't."""
    from operation_love.status import RunStatus

    driver = _BlockedDriver(3)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "blocked"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert app["stop_kind"] == "deck_blocked"
    assert driver.likes == [] and driver.dislikes == 0    # no autonomous action into a paywall
    assert store.decisions == []
    assert driver.closed


class _BlockedAfterLikeDriver(FakeDriver):
    """The deck is healthy until Hinge refuses Send Like and replaces it with a paywall."""
    accepts_opener = False

    def __init__(self):
        super().__init__(1)
        self.like_calls = 0

    def blocked_reason(self):
        return None

    def like(self, opener=None, item_index=None, *, model_item_index=None):
        self.like_calls += 1
        raise DeckBlockedError(_PAYWALL_REASON)


def test_auto_mode_post_send_blocked_error_records_no_rejected_like():
    """The worker's between-profile probe cannot see a screen that only appears *after*
    Send Like.  A driver therefore raises DeckBlockedError at that exact point; it must produce
    the same graceful status as an already-blocked deck, without increasing counters or saving a
    phantom decision."""
    from operation_love.status import RunStatus

    driver = _BlockedAfterLikeDriver()
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert driver.like_calls == 1
    assert driver.likes == []
    assert store.decisions == [] and store.labels == []
    assert app["state"] == "blocked"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert app["stop_kind"] == "deck_blocked"
    assert driver.closed


def test_worker_handles_concrete_hinge_blocked_error_without_count_store_or_failure_snapshot():
    """The concrete Hinge error must take Worker's graceful DeckBlockedError branch.

    Its HingeActionError base is intentionally retained for direct callers, so this protects the
    multiple-inheritance order from quietly turning a known paywall into the generic error path.
    """
    from operation_love.drivers.hinge import HingeActionError, HingeDeckBlockedError
    from operation_love.status import RunStatus

    class ConcreteHingePaywallDriver(FakeDriver):
        accepts_opener = False

        def __init__(self):
            super().__init__(1)
            self.failure_snapshots = []

        def blocked_reason(self):
            return None

        def like(self, opener=None, item_index=None, *, model_item_index=None):
            raise HingeDeckBlockedError(_PAYWALL_REASON)

        def snapshot_failure(self, exc):
            self.failure_snapshots.append(exc)

    assert issubclass(HingeDeckBlockedError, DeckBlockedError)
    assert issubclass(HingeDeckBlockedError, HingeActionError)
    assert HingeDeckBlockedError.__mro__[:3] == (
        HingeDeckBlockedError, HingeActionError, DeckBlockedError)

    driver = ConcreteHingePaywallDriver()
    store = FakeStore()
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="auto")
    Worker("hinge", driver, FakeDecider("like"), None, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "blocked" and app["stop_kind"] == "deck_blocked"
    assert app["swipes_run"] == 0
    assert store.decisions == [] and driver.failure_snapshots == []
    assert driver.likes == [] and driver.closed


def test_auto_mode_handles_concrete_hinge_blocked_error_from_dislike_boundary():
    """A foreground race on Pass is the same blocked state as one on Send Like.

    The dislike call historically sat outside the Like branch's DeckBlockedError handler, so
    the identical Hinge error became a red unexpected failure depending only on the decision.
    """
    from operation_love.drivers.hinge import HingeDeckBlockedError
    from operation_love.status import RunStatus

    class ForegroundLostOnDislike(FakeDriver):
        def __init__(self):
            super().__init__(1)
            self.failure_snapshots = []

        def blocked_reason(self):
            return None

        def dislike(self):
            raise HingeDeckBlockedError(_PAYWALL_REASON)

        def snapshot_failure(self, exc):
            self.failure_snapshots.append(exc)

    driver = ForegroundLostOnDislike()
    store = FakeStore()
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="auto")
    Worker("hinge", driver, FakeDecider("dislike"), None, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "blocked" and app["stop_kind"] == "deck_blocked"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert app["swipes_run"] == 0
    assert store.decisions == [] and driver.failure_snapshots == []
    assert driver.dislikes == 0 and driver.closed


def test_auto_mode_surfaces_block_latched_during_capture_that_returns_no_profile():
    """A package switch between the loop probe and capture must not look like a clean stop."""
    from operation_love.status import RunStatus

    class ForegroundLostDuringCapture(FakeDriver):
        def __init__(self):
            super().__init__(1)
            self.blocked_calls = 0

        def blocked_reason(self):
            self.blocked_calls += 1
            return None if self.blocked_calls == 1 else _PAYWALL_REASON

        def out_of_profiles(self):
            return False

        def next_profile(self):
            return None

    driver = ForegroundLostDuringCapture()
    store = FakeStore()
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="auto")
    Worker("hinge", driver, FakeDecider("dislike"), None, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("hinge")["app"]
    assert driver.blocked_calls == 2
    assert app["state"] == "blocked" and app["stop_kind"] == "deck_blocked"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert store.decisions == [] and driver.likes == [] and driver.dislikes == 0
    assert driver.closed


def test_observe_mode_handles_concrete_hinge_blocked_error_at_reviewed_input_boundary():
    """The reviewed Observe bridge shares the guarded input path and blocked semantics."""
    from operation_love.drivers.hinge import HingeDeckBlockedError
    from operation_love.status import RunStatus

    class ForegroundLostDuringObserve(FakeDriver):
        def __init__(self):
            super().__init__(1)
            self.failure_snapshots = []

        def blocked_reason(self):
            return None

        def out_of_profiles(self):
            return False

        def current_profile(self):
            return self.cards[0]

        def wait_for_decision(self, timeout=None, should_stop=None):
            raise HingeDeckBlockedError(_PAYWALL_REASON)

        def render_busy(self, message=None):
            pass

        def snapshot_failure(self, exc):
            self.failure_snapshots.append(exc)

    driver = ForegroundLostDuringObserve()
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="observe")
    Worker("hinge", driver, FakeDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "blocked" and app["stop_kind"] == "deck_blocked"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert store.decisions == [] and store.labels == []
    assert driver.failure_snapshots == [] and driver.closed


class _BlockedAndEmptyDriver(FakeDriver):
    """Reports BOTH a blocked deck AND an empty one -- the realistic case, since Hinge's
    paywall covers the deck entirely, so out_of_profiles' own perception can't find any
    cards either. blocked_reason() must win: it is the more specific, more actionable
    answer (worker.py's own comment at both call sites), not whichever check happens to be
    written first in the loop."""
    def __init__(self):
        super().__init__(0)             # genuinely no cards -- out_of_profiles is also True
        self.reason = _PAYWALL_REASON

    def blocked_reason(self):
        return self.reason


def test_observe_mode_prefers_blocked_over_out_of_profiles_when_driver_reports_both():
    from operation_love.status import RunStatus

    driver = _BlockedAndEmptyDriver()
    assert driver.out_of_profiles() is True         # sanity: the empty-deck condition really holds
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    Worker("bumble", driver, FakeDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "blocked"                 # NOT "out_of_profiles"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert app["stop_kind"] == "deck_blocked"
    assert driver.closed


def test_auto_mode_prefers_blocked_over_out_of_profiles_when_driver_reports_both():
    from operation_love.status import RunStatus

    driver = _BlockedAndEmptyDriver()
    assert driver.out_of_profiles() is True          # sanity: the empty-deck condition really holds
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "blocked"                  # NOT "out_of_profiles"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert app["stop_kind"] == "deck_blocked"
    assert driver.likes == [] and driver.dislikes == 0
    assert driver.closed


def test_auto_mode_unaffected_when_driver_explicitly_reports_no_blocked_deck():
    """Guards against the new check accidentally halting a normal run: a driver whose
    blocked_reason() is explicitly present (not merely inherited-and-never-called) but
    always answers None -- "nothing standing between us and the deck" -- must swipe the
    whole deck exactly as before, with blocked_reason() polled once per loop iteration
    (proving the check genuinely runs every time) but never once tripping it."""
    class ExplicitlyUnblockedDriver(FakeDriver):
        def __init__(self, n):
            super().__init__(n)
            self.blocked_calls = 0

        def blocked_reason(self):
            self.blocked_calls += 1
            return None

    driver = ExplicitlyUnblockedDriver(3)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("dislike"), svc, store).run()

    assert driver.dislikes == 3 and driver.likes == []
    # Polled every iteration: 3 cards processed + 1 final iteration where out_of_profiles()
    # finally reads True -- 4 calls, none of which ever returned anything but None.
    assert driver.blocked_calls == 4
    assert driver.closed


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


def test_observe_restart_renews_reviewed_action_bridge_registration(monkeypatch):
    """Every failed session unregisters in _finish_session; the replacement must rebind."""
    import operation_love.worker as wmod

    monkeypatch.setattr(wmod, "human_cooldown", lambda _seconds: 0)

    class Bridge:
        def __init__(self):
            self.registered = None
            self.register_calls = 0

        def register(self, worker):
            self.registered = worker
            self.register_calls += 1

        def unregister(self, worker):
            if self.registered is worker:
                self.registered = None

    class RestartingObserve(FakeDriver):
        halt_on_error = False

        def __init__(self):
            super().__init__(0)
            self.attempts = 0

        def open_session(self):
            self.attempts += 1
            self.opened = True

        def out_of_profiles(self):
            if self.attempts <= 2:
                raise RuntimeError("transient after session opened")
            assert bridge.registered is worker
            return True

    bridge = Bridge()
    driver = RestartingObserve()
    stop = threading.Event()
    worker = Worker(
        "bumble", driver, FakeDecider("like"), None, FakeStore(), "run1", _Pacing(),
        stop, mode="observe", max_restarts=5, observe_action_bridge=bridge)

    worker.run()

    assert driver.attempts == 3
    assert bridge.register_calls == 3
    assert bridge.registered is None


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


def test_action_cancelled_ends_auto_normally_without_a_phantom_like_or_snapshot():
    """A Stop at the driver's action boundary is neither a targeting miss nor a crash."""
    class CancelledLike(InterruptibleLikeDriver):
        def __init__(self):
            super().__init__(1)
            self.snapshots = 0

        def like(self, opener=None, item_index=None, *, model_item_index=None, should_stop=None):
            self.like_stop_callbacks.append(should_stop)
            raise ActionCancelled("stop before heart tap")

        def snapshot_failure(self, exc):
            self.snapshots += 1

    driver = CancelledLike()
    store = FakeStore()
    client = FakeOpenerClient()
    service = OpenerService(client, CostTracker(PRICING, None), store, "s")
    worker = _worker(driver, FakeDecider("like"), service, store)
    worker.run()

    assert worker.stop_event.is_set()
    assert driver.likes == [] and driver.snapshots == 0
    assert store.decisions == []


# Sentinel for "this test did not override the pick", distinct from None (which is a real
# maybe_opener answer: "no opener for this profile").
_UNSET = object()

# The numbered item crops an enumerating capture hands over (ops/OPENER-REDESIGN.md 5.2/5.7).
# Doc 5.9's inversion means OBSERVE sends these too, so every observe fake below has to carry
# them -- a Profile with no items is now a Profile no suggestion can be made for, on purpose.
_OBSERVE_ITEMS = (b"item-1-crop", b"item-2-crop", b"item-3-crop")


def _observe_card(index=0):
    return Profile(photos=[b"x"], bio=f"bio{index}", name="Ada", items=_OBSERVE_ITEMS,
                   item_context=(b"vitals-crop",))


class _ObserveLikeIntentDriver(FakeDriver):
    """A Hinge-shaped observe driver for doc 5.9's INVERTED flow.

    It exposes the comment-sheet hook (supports_observe_like_intent) and calls it exactly like
    the real driver does -- TWO positional args, active then anchor -- and it enumerates, i.e.
    its Profile carries numbered item crops, because since the inversion that is what a
    suggestion is made FROM. It also implements `observe_item_mismatch`, the driver-side check
    the worker runs the human's tap through; `mismatch` is what that returns ("" = the human
    opened the item the suggestion names).

    `gate`, when set, is called at the top of wait_for_decision -- the fake equivalent of a human
    who waits for the hub before tapping. Without it the test would race the suggestion thread.
    `anchor=None` models the driver failing to capture the live sheet frame, which under the
    inversion is no longer "generate blind" but "cannot check what you opened"."""
    supports_observe_like_intent = True
    accepts_opener = True

    def __init__(self, anchor=b"the-actual-like-sheet-frame", *, mismatch="", gate=None):
        super().__init__(1)
        self.cards = [_observe_card()]
        self._served = False
        self.anchor = anchor
        self.mismatch = mismatch
        self.gate = gate
        self.checked = []          # every (sheet, model_item_index) the worker asked about

    def out_of_profiles(self):
        return self._served

    def current_profile(self):
        self._served = True
        return self.cards[0]

    def render_busy(self, message=None):
        pass

    def observe_item_mismatch(self, sheet, model_item_index):
        self.checked.append((sheet, model_item_index))
        return self.mismatch

    def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
        if self.gate is not None:
            self.gate()
        if on_like_intent is not None:
            on_like_intent(True, self.anchor)   # simulate opening Hinge's like/comment sheet
        return True                              # then LIKE


def _settled(status, app="bumble", timeout=10.0):
    """A `gate` that blocks until the card's suggestion has finished publishing.

    Doc 5.9 puts generation on its own thread so READY can be published immediately, which means
    a test driver that taps the instant wait_for_decision is entered would be racing it. This is
    the human who looks at the hub first: it waits for `opener_pending` to go False, which the
    worker publishes exactly once per card when the call settles (or immediately, when there was
    no call to make). Bounded, so a test that never produces a suggestion fails on its own
    assertion rather than hanging."""
    def gate():
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            view = status.app_view(app)["app"]
            if view is not None and not view.get("opener_pending"):
                return
            time.sleep(0.002)
    return gate


def test_observe_suggestion_passes_a_run_and_card_cancellation_predicate_to_maybe_opener():
    """Pins the OTHER call site: the observe-mode suggestion -- which doc 5.9 moved from the
    post-heart callback to a thread started right after READY -- must still thread
    a predicate including the run's stop event, not just the auto-loop path pinned above. It
    matters MORE here than before the inversion: the call now happens on every card rather than
    only on the ones the human hearts, so a Stop click (or leaving one card) has more in-flight
    calls to abort."""
    from operation_love.status import RunStatus

    class ObserveDecider(FakeDecider):
        def embed(self, profile):
            return [0.1, 0.2]
        def retrain(self, store):
            return True

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    store = FakeStore()
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    w = Worker("bumble", driver, ObserveDecider(), svc, store, "run1", _Pacing(),
              threading.Event(), mode="observe", status=status)
    w.run()

    assert client.calls == 1
    # The predicate includes card-local cancellation as well as the run-level event.  This
    # normal completion did not stop the run, but its wait-finally did cancel its card, so the
    # callable retained by the client must now report a stop rather than leave a stale request
    # eligible to start later.
    assert not w.stop_event.is_set()
    assert len(client.should_stops) == 1 and client.should_stops[0]()


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
    """Records every maybe_opener() call's kwargs (anchor, items, advisory, should_stop) and, if
    `status` is supplied, the app's live state AT THE MOMENT of the call. `raise_exc`, if set,
    makes the call raise instead of returning a pick -- exercising the exception path in
    _ObserveSuggestion._generate. `referenced`/`item_description` are echoed onto the returned
    OpenerPick, which is what the hub renders beside the text.

    `index` defaults to a REAL item number rather than the dataclass's ITEM_INDEX_ABSENT: since
    doc 5.9's inversion an opener with no item number is one observe refuses to show (there is
    nothing to tell the human to like and nothing to check their tap against), so a careless
    default would turn every test here into a warning-path test."""
    stop_requested = False
    disabled = False
    last_skip_reason = None

    def __init__(self, *, status=None, app=None, raise_exc=None, suggestion="hi", referenced="",
                 index=2, item_description="", pick=_UNSET):
        self.status = status
        self.app = app
        self.raise_exc = raise_exc
        self.suggestion = suggestion
        self.referenced = referenced
        self.index = index
        self.item_description = item_description
        self.pick = pick
        self.calls = []
        self.committed = []
        self.state_during_call = None

    def maybe_opener(self, run_id, app, profile, *, anchor=None, items=None,
                     should_stop=None, advisory=False):
        self.calls.append({"run_id": run_id, "app": app, "anchor": anchor, "items": items,
                           "should_stop": should_stop, "advisory": advisory})
        if self.status is not None:
            self.state_during_call = self.status.app_view(self.app)["app"]["state"]
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.pick is not _UNSET:
            return self.pick
        return OpenerPick(self.suggestion, index=self.index, referenced=self.referenced,
                          item_description=self.item_description)

    def commit_advisory_opener(self, pick, **_):
        self.committed.append(pick)
        return True


def test_observe_suggestion_passes_advisory_true():
    """The observe suggestion call site must pass advisory=True -- this is what makes
    maybe_opener() use the short, deadline-bounded retry policy and route any exhaustion
    through request_stop=False instead of ending the session.

    THE ONE DELIBERATE DIVERGENCE FROM AUTO left by doc 5.9's inversion, and it is a divergence
    in RETRY POLICY, never in the request: the crops, the schema and the prompt are identical
    (pinned below), so any opener observe DOES show is byte-identical to what auto would send.
    What differs is only the smaller attempt/time budget, because an advisory failure must
    never end a labelling session and observe now calls on every card rather than only likes."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    store = FakeStore()
    svc = _RecordingOpenerService()
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert svc.calls and svc.calls[0]["advisory"] is True
    assert len(svc.committed) == 1


def test_confirmed_observe_like_commits_before_a_concurrent_stop():
    """Stop before a decision drops the advisory draft; Stop *after* a confirmed Like may not.

    The driver/bridge has already returned ``True`` at this boundary, so the account action is
    complete even if the hub's Stop request wins the next scheduler timeslice.  The worker must
    commit the staged opener and persist the preference, then end without capturing another
    card.  Before the guard in _observe_loop was removed this exact timing lost both records.
    """
    class StopAfterConfirmedDecisionWorker(Worker):
        def _wait_for_observed_decision(self, suggestion, profile_token=None):
            deadline = time.monotonic() + _LIVENESS_TIMEOUT_S
            while suggestion._pick is None and time.monotonic() < deadline:
                time.sleep(0.002)
            assert suggestion._pick is not None, "the staged advisory opener never arrived"
            # Model a Stop click immediately after the driver has structurally verified and
            # returned a Like, before _observe_loop starts its persistence section.
            self.stop_event.set()
            return True

    driver = _ObserveLikeIntentDriver()
    store = FakeStore()
    svc = _RecordingOpenerService()
    stop = threading.Event()
    StopAfterConfirmedDecisionWorker(
        "bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(), stop,
        mode="observe").run()

    assert stop.is_set()
    assert len(svc.committed) == 1
    assert store.decisions == [("bumble", "like", "manual")]
    assert len(store.profiles) == len(store.labels) == 1
    assert driver.i == 0  # current_profile served exactly one card; Stop started no new read


@pytest.mark.parametrize(
    ("embedding", "archive_ok"),
    [(None, True), ([0.1, 0.2], False)],
    ids=["no_face", "archive_failed"],
)
def test_observe_landed_like_has_decision_before_opener_when_optional_label_work_fails(
        embedding, archive_ok):
    """A real reviewed Like is an action even when it cannot become a training label.

    The old order committed its opener first and only wrote a decision after archive/embed/
    label work.  Both early-return paths consequently left a true action looking like a phantom
    opener.  Decision persistence is now the common action boundary, with the staged opener
    immediately after it; only the optional training record may be absent.
    """
    from operation_love.status import RunStatus

    events = []

    class OrderedStore(FakeStore):
        def record_decision(self, *args, **kwargs):
            events.append("decision")
            kwargs.pop("profile_id", None)
            kwargs.pop("created_at", None)
            return super().record_decision(*args, **kwargs)

        def record_profile(self, *args, **kwargs):
            events.append("profile")
            super().record_profile(*args, **kwargs)
            return archive_ok

    class OrderedSuggestionService(_RecordingOpenerService):
        def commit_advisory_opener(self, pick, **_):
            events.append("opener")
            return super().commit_advisory_opener(pick)

    class OptionalLabelDecider(_ObserveDecider):
        def embed(self, profile):
            return embedding

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    store = OrderedStore()
    svc = OrderedSuggestionService()
    Worker("bumble", driver, OptionalLabelDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert events[:2] == ["decision", "opener"]
    assert store.decisions == [("bumble", "like", "manual")]
    assert len(svc.committed) == 1
    assert store.labels == []


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


def test_observe_publishes_ready_before_the_suggestion_is_even_requested():
    """DOC 5.9's TIMING RULE: "publish READY immediately and let the suggestion fill in behind
    it". Captured from INSIDE the fake service's maybe_opener() -- i.e. at the exact moment the
    real, slow call would be blocking -- so this proves the operator was already free to act
    WHILE the call was in flight, not merely before or after it.

    This is stronger than a courtesy and that is why it is pinned: generating before entering
    wait_for_decision would mean a human who acts during the (up to request_timeout_s) call is
    never observed at all, because the driver's first frame would already be the next card --
    the decision lost, and the one after it attributed to the wrong profile."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    store = FakeStore()
    svc = _RecordingOpenerService(status=status, app="bumble")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    # "waiting" is the GO cue -- the operator is being told to use the app's controls -- and it
    # is live while the model is being asked. The old flow published a blocking "suggesting"
    # state here instead, which was correct then (the call sat between the heart tap and the
    # suggestion) and would be a lie now.
    assert svc.state_during_call == "waiting"
    assert svc.calls, "the suggestion must actually have been requested"


def test_observe_publishes_an_optional_opener_without_consulting_the_ranker():
    """Observe suggestions are conditional writing help, not ranker verdicts.

    A cold or negative ranker must neither suppress the advisory request nor decide the card.
    The human's pass below is deliberately recorded as the outcome, while the suggestion is
    still requested and published before that manual action.
    """
    from operation_love.status import RunStatus

    class RankerMustNotDecide:
        def __init__(self):
            self.decide_calls = 0

        def decide(self, profile):
            self.decide_calls += 1
            raise AssertionError("observe must not ask the ranker to decide")

        def embed(self, profile):
            return [0.1, 0.2]

        def retrain(self, store):
            return True

    class PassingDriver(_ObserveLikeIntentDriver):
        def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
            if self.gate is not None:
                self.gate()
            return False

    status = RunStatus("run1", ["bumble"], min_labels=99, mode="observe")
    calls = _record_state_transitions(status)
    driver = PassingDriver(gate=_settled(status))
    decider = RankerMustNotDecide()
    store = FakeStore()
    svc = _RecordingOpenerService(suggestion="optional hello", index=2)

    Worker("bumble", driver, decider, svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert decider.decide_calls == 0
    assert len(svc.calls) == 1
    assert any(call.get("opener_suggestion") == "optional hello" for call in calls)
    assert store.decisions == [("bumble", "dislike", "manual")]


def _record_state_transitions(status):
    """Wrap status.set_app to record every (fields) call while still applying it normally --
    lets a test see the STATE SEQUENCE over time, not just the observe loop's FINAL state.
    The final state is the wrong thing to assert on for a per-card transition: the loop's LATER
    stages (e.g. _block_observe_processing's 'acting') overwrite it again well before the run
    ends, which would make a 'cleared' assertion pass for the wrong reason (loop progress, not
    the specific transition this call site is responsible for)."""
    calls = []
    original = status.set_app
    def recording(app, **fields):
        calls.append(dict(fields))
        original(app, **fields)
    status.set_app = recording
    return calls


def test_observe_marks_the_suggestion_pending_and_then_clears_it():
    """`opener_pending` is what replaced the old blocking "suggesting" STATE: the hub says a
    suggestion is on its way beside a live GO cue, rather than telling the operator to wait.
    It must be published before the call and cleared by the publish that carries the answer --
    a pending flag that outlives its call is a spinner that never stops."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="loved your trail photo")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    pendings = [c for c in calls if c.get("opener_pending") is True]
    assert pendings, "the pending marker must be published before the call"
    assert all(c.get("opener_suggestion") is None for c in pendings), \
        "pending means there is nothing to type yet"
    answered = next(c for c in calls if c.get("opener_suggestion") == "loved your trail photo")
    assert answered["opener_pending"] is False
    assert calls.index(answered) > calls.index(pendings[0])
    # And nothing publishes the old blocking state any more.
    assert "suggesting" not in [c.get("state") for c in calls]


def test_observe_suggestion_failure_publishes_a_warning_rather_than_going_quiet():
    """A raise inside the suggestion call must never break a labelling run -- and must never be
    silent either. There is no text, so the hub gets `opener_warning` naming the failure: doc
    5.9's rule is that the one thing observe may not do is say nothing."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(raise_exc=RuntimeError("boom"))
    w = Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
               threading.Event(), mode="observe", status=status)
    w.run()

    warned = next(c for c in calls if c.get("opener_warning"))
    assert "boom" in warned["opener_warning"]
    assert warned["opener_suggestion"] is None
    assert warned["opener_pending"] is False
    assert len(store.labels) == 1 and not w.stop_event.is_set()   # the run carried on regardless


# ---------------------------------------------------------------------------------------
# DOC 5.9's INVERSION. Observe used to generate AFTER the human tapped, handing the live
# like-sheet frame to the model as an ANCHOR so the opener was right by construction. It now
# generates BEFORE the tap, from the same numbered crops auto sends, and the hub tells the human
# WHICH ITEM to like. Three properties fall out and all three are pinned below:
#   * the request is the crop shape, and the anchor is not sent at all;
#   * "like item N" plus the model's own description of item N reach the hub with the text;
#   * the human may open a DIFFERENT item, which must be detected and surfaced -- warning, and
#     NO text to type. Never silent, never a stop (that is AUTO's answer to the same rule).
# ---------------------------------------------------------------------------------------

def test_observe_sends_the_item_crops_and_no_anchor():
    """The canary property, at the request layer. Observe must issue the SAME request shape as
    auto -- `ItemRequest.from_profile(profile)`, no anchor -- because a mode-dependent payload is
    exactly the divergence that made testing observe say nothing about auto. The anchor is not
    merely unnecessary here, it is mutually exclusive with the item list (the client refuses a
    request carrying both), so this also pins that the inversion removed a live instruction
    rather than leaving two in the same call."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    store = FakeStore()
    svc = _RecordingOpenerService()
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert len(svc.calls) == 1
    assert svc.calls[0]["anchor"] is None
    items = svc.calls[0]["items"]
    assert items is not None and items.items == _OBSERVE_ITEMS
    assert items.context == (b"vitals-crop",) and items.name == "Ada"


def test_observe_publishes_the_item_number_and_description_with_the_text():
    """The instruction the inversion produces. `opener_item` is what the operator acts on FIRST
    -- there is no point typing a message under the wrong card -- and `opener_item_description`
    is the model's own words for that item so it can be found without counting hearts. Doc 5.7's
    "observe displays it" for `item_description`, which until now was returned, carried and
    persisted but rendered nowhere."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="loved your trail photo", referenced="the trail photo",
                                  index=3, item_description="the ridgeline photo")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    shown = next(c for c in calls if c.get("opener_suggestion") == "loved your trail photo")
    assert shown["opener_item"] == 3
    assert shown["opener_item_description"] == "the ridgeline photo"
    assert shown["opener_referenced"] == "the trail photo"
    assert shown["opener_warning"] is None


def test_observe_publishes_a_proved_media_ordinal_for_hinge_display():
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["hinge"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    driver.model_item_media_ordinal = lambda model_item: 4 if model_item == 3 else None
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="loved your trail photo", index=3,
                                  item_description="the ridgeline photo")
    Worker("hinge", driver, _ObserveDecider(), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    shown = next(c for c in calls if c.get("opener_suggestion") == "loved your trail photo")
    assert shown["opener_item"] == 3
    assert shown["opener_media_ordinal"] == 4


def test_observe_rejects_a_pick_in_any_non_model_item_index_space():
    """Observe's numbered crop request can only be interpreted in model-item space.

    A stale/custom producer can still hand back a legacy ``profile_photos`` pick.  AUTO has an
    explicit capture-order branch for that legacy response, but Observe's sheet checker always
    treats its number as a numbered crop.  Showing the text would therefore let the hub certify
    one item while the model wrote about another, so this is a warning with no text and no sheet
    check rather than an attempted conversion.
    """
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(pick=OpenerPick(
        "must never be offered", index=1, referenced="r",
        index_space=INDEX_SPACE_PROFILE_PHOTOS,
    ))
    w = Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
               threading.Event(), mode="observe", status=status)
    w.run()

    assert driver.checked == []
    assert all(call.get("opener_suggestion") != "must never be offered" for call in calls)
    warned = next(call for call in calls if call.get("opener_warning"))
    assert "unsupported index space" in warned["opener_warning"]
    assert "profile_photos" in warned["opener_warning"]
    assert len(store.labels) == 1 and not w.stop_event.is_set()


def test_observe_cancelled_before_publish_never_announces_a_stale_ready_suggestion(
        monkeypatch, capsys):
    """Cancellation between generation and publish must suppress status *and* console advice.

    The delayed wrapper creates the exact narrow interleaving: the model result has already
    been stored under the suggestion lock, but the final publish has not acquired it yet.  The
    worker's normal wait-finally calls ``cancel`` in that gap.  A stale ready line would survive
    in a bug report after the card has moved on, even though the hub correctly suppresses it.
    """
    from operation_love.status import RunStatus
    from operation_love.worker import _ObserveSuggestion

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    worker = Worker("bumble", _ObserveLikeIntentDriver(), _ObserveDecider(),
                    _RecordingOpenerService(suggestion="stale advice", index=2), FakeStore(),
                    "run1", _Pacing(), threading.Event(), mode="observe", status=status)
    suggestion = _ObserveSuggestion(worker, _observe_card())
    entered_publish = threading.Event()
    release_publish = threading.Event()
    real_publish = suggestion._publish

    def pause_before_final_publish(*, announce_pick=None):
        if announce_pick is not None:
            entered_publish.set()
            assert release_publish.wait(timeout=_LIVENESS_TIMEOUT_S)
        return real_publish(announce_pick=announce_pick)

    monkeypatch.setattr(suggestion, "_publish", pause_before_final_publish)
    suggestion.start()
    assert entered_publish.wait(timeout=_LIVENESS_TIMEOUT_S)
    suggestion.cancel()
    release_publish.set()
    assert suggestion._thread is not None
    suggestion._thread.join(timeout=_LIVENESS_TIMEOUT_S)
    assert not suggestion._thread.is_alive()

    assert "suggestion ready" not in capsys.readouterr().out
    assert status.app_view("bumble")["app"]["opener_suggestion"] is None


def test_observe_cancelled_queued_suggestion_never_reaches_provider_and_does_not_block_newest():
    """Leaving card two while card one owns the shared opener lock must cost nothing for two.

    The real service serializes all provider attempts under one lock.  Before cancellation was
    composed into its ``should_stop`` predicate, a fast swiper could build a queue of daemon
    threads: each old card waited, then made a billable request after its card was gone, before
    discarding its answer.  This fake has the same two-stage structure (queue lock, then provider
    boundary) and makes that interleaving deterministic.  Card three proves that the canceled
    card does not occupy a provider turn ahead of the current card once card one releases.
    """
    from operation_love.worker import _ObserveSuggestion

    class Driver:
        accepts_opener = True

        def observe_item_mismatch(self, sheet, model_item_index):
            return ""

    class SerializedService:
        disabled = False
        stop_requested = False
        last_skip_reason = None

        def __init__(self):
            self._lock = threading.Lock()
            self.first_at_provider = threading.Event()
            self.second_queued = threading.Event()
            self.release_first = threading.Event()
            self.provider_calls = []

        def maybe_opener(self, run_id, app, profile, *, items, should_stop, advisory):
            if profile.bio == "second":
                self.second_queued.set()
            with self._lock:
                # This is OpenerService.maybe_opener's before-attempt cancellation boundary.
                if should_stop():
                    return None
                self.provider_calls.append(profile.bio)
                if profile.bio == "first":
                    self.first_at_provider.set()
                    assert self.release_first.wait(timeout=_LIVENESS_TIMEOUT_S)
                return OpenerPick("hello", index=1)

    service = SerializedService()
    worker = Worker("hinge", Driver(), None, service, FakeStore(), "run1", _Pacing(),
                    threading.Event(), mode="observe")
    first = _ObserveSuggestion(worker, _observe_card())
    first._profile.bio = "first"
    second = _ObserveSuggestion(worker, _observe_card())
    second._profile.bio = "second"
    newest = _ObserveSuggestion(worker, _observe_card())
    newest._profile.bio = "newest"

    first.start()
    assert service.first_at_provider.wait(timeout=_LIVENESS_TIMEOUT_S)
    second.start()
    assert service.second_queued.wait(timeout=_LIVENESS_TIMEOUT_S)
    second.cancel()
    service.release_first.set()
    assert first._thread is not None and second._thread is not None
    first._thread.join(timeout=_LIVENESS_TIMEOUT_S)
    second._thread.join(timeout=_LIVENESS_TIMEOUT_S)
    assert not first._thread.is_alive() and not second._thread.is_alive()
    assert service.provider_calls == ["first"]

    newest.start()
    assert newest._thread is not None
    newest._thread.join(timeout=_LIVENESS_TIMEOUT_S)
    assert not newest._thread.is_alive()
    assert service.provider_calls == ["first", "newest"]


def test_observe_checks_the_opened_sheet_against_the_item_the_opener_names():
    """The mismatch guard's HAPPY path, and the pin that it actually runs. The frame the driver
    hands over on the heart tap is passed to the driver's own deterministic check together with
    the item number the model chose; the text survives only because that check confirmed it."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(anchor=b"the-actual-like-sheet-frame",
                                      gate=_settled(status))
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="loved your trail photo", index=3)
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert driver.checked == [(b"the-actual-like-sheet-frame", 3)]
    sent = next(c for c in calls if c.get("state") == "waiting_for_send")
    assert sent["opener_suggestion"] == "loved your trail photo"
    assert sent["opener_warning"] is None


def test_observe_replaces_the_opener_with_a_warning_when_the_human_opens_another_item():
    """THE headline requirement of doc 5.9, and the regression the inversion would otherwise
    introduce on the one path where a real message reaches a real person.

    The suggestion is about item 3; the human hearts something else. The hub must REPLACE the
    opener with the warning and offer nothing to type -- not annotate it, not show both. Text
    left on screen beside a caveat is text that gets typed anyway, which is precisely the
    out-of-place opener this whole redesign exists to stop."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status),
                                      mismatch="you opened item 5, not item 3")
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="loved your trail photo", index=3)
    w = Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
               threading.Event(), mode="observe", status=status)
    w.run()

    sent = next(c for c in calls if c.get("state") == "waiting_for_send")
    assert sent["opener_warning"] == "you opened item 5, not item 3"
    assert sent["opener_suggestion"] is None
    assert sent["opener_item"] == 3        # still says which item it WAS written for
    # Observe refuses to show text; it does NOT stop the run, and the human's own decision is
    # still recorded. That asymmetry with AUTO is doc 5.9's, stated: same rule, different
    # enforcement, because here the bot is not the one sending anything.
    assert not w.stop_event.is_set()
    assert len(store.labels) == 1


def test_a_mismatch_reaches_the_console_as_well_as_the_hub(capsys):
    """"Silent is the one thing it must not be" is a claim about both surfaces. The hub is what
    doc 5.9 names, but stdout is what a bug report keeps and what the hub's own log panel tees,
    and an operator who is looking at the phone rather than the browser has to be able to find
    out afterwards why there was nothing to type."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status),
                                      mismatch="you opened item 5, not item 3")
    store = FakeStore()
    svc = _RecordingOpenerService(suggestion="loved your trail photo", index=3)
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    printed = capsys.readouterr().out
    assert "you opened item 5, not item 3" in printed
    assert printed.count("you opened item 5, not item 3") == 1   # once per distinct warning
    assert "loved your trail photo" not in printed               # the TEXT stays hub-only


def test_observe_keeps_advice_when_a_sheet_frame_is_temporarily_unavailable():
    """A missing sheet frame is inconclusive, not proof the human chose a different item.

    Hinge can publish a usable frame on a later callback, so the optional generated opener stays
    visible meanwhile. An affirmative mismatch remains the only condition that hides it.
    """
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(anchor=None, gate=_settled(status))
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="hey there", index=2)
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert driver.checked == [], "nothing to check means the check must not be faked"
    sent = next(c for c in calls if c.get("state") == "waiting_for_send")
    assert sent["opener_suggestion"] == "hey there"
    assert sent["opener_warning"] is None


def test_observe_warns_and_never_asks_when_the_capture_could_not_be_enumerated():
    """An enumeration refusal STOPS an auto run (doc 5.2: raw scroll frames cannot carry an item
    number, so there is no honest request to make). In observe the human is the one acting, so it
    must not stop -- but it must not silently fall back to the old frame-shape request either,
    which would put observe and auto back on different payloads. No call is made at all, and the
    driver's own sentence is what the operator is shown."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    driver.cards = [Profile(photos=[b"x"], items_unavailable="the scroll top was not confirmed")]
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService()
    w = Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
               threading.Event(), mode="observe", status=status)
    w.run()

    assert svc.calls == [], "no numbered items means no request, never a frame-shape fallback"
    warned = next(c for c in calls if c.get("opener_warning"))
    assert warned["opener_warning"] == "the scroll top was not confirmed"
    assert not w.stop_event.is_set() and len(store.labels) == 1


def test_observe_prefers_the_drivers_derived_reason_when_nothing_was_numbered():
    """found+fixed 2026-08-22: a capture that enumerated fine but numbered nothing (every card
    demoted by the still-photo gate, say) is `items_unnumbered`, not `items_unavailable` -- and
    unlike the sibling test above, this must never stop the run (the auto loop's own hard-stop
    check reads `items_unavailable` alone, untouched here). The hub warning should still prefer
    the driver's own derived sentence over the generic "no numbered items" fallback."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    driver.cards = [Profile(
        photos=[b"x"],
        items_unnumbered="15 selectable card(s) were considered; 15 could not be judged because "
                         "the one dwell burst this capture takes never covered them.")]
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService()
    w = Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
               threading.Event(), mode="observe", status=status)
    w.run()

    assert svc.calls == [], "no numbered items means no request"
    warned = next(c for c in calls if c.get("opener_warning"))
    assert warned["opener_warning"].startswith("15 selectable card(s) were considered")
    assert not w.stop_event.is_set() and len(store.labels) == 1


def test_observe_falls_back_to_the_generic_message_when_the_driver_recorded_no_reason():
    """A driver that leaves BOTH `items_unavailable` and `items_unnumbered` empty -- every
    non-Hinge driver today, or a Hinge capture that never attempted enumeration at all -- still
    gets a sentence on the hub, never a blank warning."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    driver.cards = [Profile(photos=[b"x"])]
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService()
    w = Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
               threading.Event(), mode="observe", status=status)
    w.run()

    assert svc.calls == []
    warned = next(c for c in calls if c.get("opener_warning"))
    assert warned["opener_warning"] == (
        "this capture produced no numbered items, so there is nothing for the model "
        "to choose from")
    assert not w.stop_event.is_set() and len(store.labels) == 1


def test_observe_summarizes_an_item_index_contradiction_for_the_hub():
    """Strip votes are debug evidence, not instructions a person can act on."""
    from operation_love.worker import _operator_items_unavailable_warning

    raw = ("the item index this capture produced contradicts itself, so its numbering cannot be "
           "trusted: frames 3 and 4 could not be put in one coordinate space: no_consensus")
    warning = _operator_items_unavailable_warning(raw)

    assert "could not be reliably counted" in warning
    assert "changed while it was being read" not in warning
    assert "pass or like manually" in warning
    assert "frames 3 and 4" not in warning and "no_consensus" not in warning


def test_observe_targeting_readiness_gate_withholds_text_before_the_provider_call(capsys):
    """A per-device calibration gate must run before generation, not only after sheet-open.

    Otherwise unchecked opener text can sit on the hub while the human decides, then disappear
    only after their tap.  Missing calibration is advisory in Observe: labels continue, but the
    provider sees no request and no text is ever offered.
    """
    from operation_love.status import RunStatus

    class _UncalibratedObserveDriver(_ObserveLikeIntentDriver):
        def targeted_suggestion_blocker(self):
            return "targeting_calibration is unavailable; no opener text is offered"

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _UncalibratedObserveDriver(gate=_settled(status))
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="must never appear")
    w = Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
               threading.Event(), mode="observe", status=status)
    w.run()

    assert svc.calls == []
    assert all(call.get("opener_suggestion") != "must never appear" for call in calls)
    warned = next(call for call in calls if call.get("opener_warning"))
    assert "targeting_calibration is unavailable" in warned["opener_warning"]
    assert not w.stop_event.is_set() and len(store.labels) == 1
    startup = capsys.readouterr().out
    assert startup.count("targeted opener suggestions need setup") == 1
    assert "Manual pass/like labels still work" in startup
    assert "ops/RUNBOOK.md" in startup


def test_observe_says_nothing_at_all_on_an_app_that_has_no_opener_feature():
    """A warning on EVERY card of a run that was never going to have a suggestion is a standing
    red box that teaches the operator to stop reading warnings. "This app cannot attach a comment
    to a like" (Bumble) and "openers are off for this run" are per-RUN facts and are not news;
    only a per-CARD refusal is."""
    from operation_love.status import RunStatus

    class _NoOpenerObserveDriver(_ObserveLikeIntentDriver):
        accepts_opener = False

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _NoOpenerObserveDriver(gate=_settled(status))
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService()
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert svc.calls == []
    assert all(not c.get("opener_warning") for c in calls)
    assert all(not c.get("opener_pending") for c in calls)


def test_observe_tells_the_driver_whether_openers_exist_before_the_session_opens():
    """`set_opener_enabled` is the driver's only gate on enumeration since doc 5.9 removed the
    auto-session one, so BOTH loops must call it -- an observe run with openers off would
    otherwise pay a ~40-frame enumeration read per card for a numbered list nobody would look at.
    Called before open_session(), like the auto loop's, so the very first capture already knows."""
    class _RecordingHookDriver(_ObserveLikeIntentDriver):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.opener_flags = []
            self.hook_order = []

        def set_opener_enabled(self, enabled):
            self.opener_flags.append(enabled)
            self.hook_order.append("set_opener_enabled")

        def open_session(self):
            self.hook_order.append("open_session")
            super().open_session()

    class _DisabledService(_RecordingOpenerService):
        disabled = True

    live = _RecordingHookDriver()
    Worker("bumble", live, _ObserveDecider(), _RecordingOpenerService(), FakeStore(), "run1",
           _Pacing(), threading.Event(), mode="observe").run()
    assert live.opener_flags == [True]
    assert live.hook_order == ["set_opener_enabled", "open_session"]

    off = _RecordingHookDriver()
    Worker("bumble", off, _ObserveDecider(), _DisabledService(), FakeStore(), "run1",
           _Pacing(), threading.Event(), mode="observe").run()
    assert off.opener_flags == [False]


def test_observe_dismissing_the_like_sheet_goes_back_to_the_instruction_and_drops_the_warning():
    """The sheet can close WITHOUT a send (dismissed) just as it can after one.

    Under the inversion this is NOT "clear everything": the card has not changed, so "like item
    3" is still the right advice and the human may well go and open item 3 next. What must be
    dropped is the EVIDENCE about what they opened -- a mismatch warning that outlived the sheet
    it was about would tell the operator their next tap was wrong before they made it. So a
    dismiss returns the hub to the pre-tap instruction, with the text back."""
    from operation_love.status import RunStatus

    class DismissThenLikeDriver(_ObserveLikeIntentDriver):
        def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
            self.gate()
            if on_like_intent is not None:
                on_like_intent(True, self.anchor)    # sheet opens on the WRONG item -> warning
                on_like_intent(False, None)           # sheet closes (dismissed, not sent)
            return True

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = DismissThenLikeDriver(gate=_settled(status), mismatch="you opened item 5, not item 3")
    store = FakeStore()
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="loved your trail photo", referenced="the trail",
                                  index=3)
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    warned = next(c for c in calls if c.get("opener_warning"))
    dismissed = calls[calls.index(warned) + 1]
    assert dismissed["state"] == "waiting"
    assert dismissed["opener_warning"] is None
    assert dismissed["opener_suggestion"] == "loved your trail photo"
    assert dismissed["opener_item"] == 3


def test_observe_keeps_the_advice_visible_while_a_settling_sheet_is_rechecked():
    """An unmeasurable first preview is not evidence the human opened a different item.

    The Hinge keyboard can reflow the selected-card preview above the composer. The temporary
    layout failure must not remove an already-generated opener; a later settled frame still gets
    the normal positive verification and only an affirmative wrong-item verdict may hide text.
    """
    from operation_love.status import RunStatus

    class SettlingSheetDriver(_ObserveLikeIntentDriver):
        def observe_item_check(self, sheet, model_item_index):
            self.checked.append((sheet, model_item_index))
            if sheet == b"settling-sheet":
                return ObserveItemCheck(
                    OBSERVE_ITEM_INCONCLUSIVE,
                    "the selected image could not yet be confirmed as model item 3")
            return ObserveItemCheck(OBSERVE_ITEM_MATCH)

        def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
            self.gate()
            if on_like_intent is not None:
                on_like_intent(True, b"settling-sheet")
                on_like_intent(True, b"settled-item-3-sheet")
            return True

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = SettlingSheetDriver(gate=_settled(status))
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="loved your trail photo", index=3)
    Worker("bumble", driver, _ObserveDecider(), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert driver.checked == [(b"settling-sheet", 3), (b"settled-item-3-sheet", 3)]
    open_sheet = [c for c in calls if c.get("state") == "waiting_for_send"]
    assert len(open_sheet) == 2
    assert all(c.get("opener_suggestion") == "loved your trail photo" for c in open_sheet)
    assert all(c.get("opener_warning") is None for c in open_sheet)


def test_observe_keeps_a_verified_suggestion_through_inconclusive_typing_refreshes():
    """Exact 2026-08-16 Julia regression: once item 3 is positively verified, keyboard text,
    cursor handles and selection overlays in later frames of the SAME continuously open composer
    may be inconclusive but must never alternate the hub back to "no suggestion to type"."""
    from operation_love.status import RunStatus

    class TypingRefreshDriver(_ObserveLikeIntentDriver):
        def observe_item_check(self, sheet, model_item_index):
            self.checked.append((sheet, model_item_index))
            if sheet in {b"typing-overlay", b"selection-popup"}:
                return ObserveItemCheck(
                    OBSERVE_ITEM_INCONCLUSIVE,
                    "the selected-card preview is not immediately above the inline composer")
            return ObserveItemCheck(OBSERVE_ITEM_MATCH)

        def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
            self.gate()
            if on_like_intent is not None:
                on_like_intent(True, b"verified-item-3-sheet")
                on_like_intent(True, b"typing-overlay")
                on_like_intent(True, b"selection-popup")
            return True

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = TypingRefreshDriver(gate=_settled(status))
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="loved your FlowRider photo", index=3)
    Worker("bumble", driver, _ObserveDecider(), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    assert driver.checked == [
        (b"verified-item-3-sheet", 3),
        (b"typing-overlay", 3),
        (b"selection-popup", 3),
    ]
    open_sheet = [c for c in calls if c.get("state") == "waiting_for_send"]
    assert len(open_sheet) == 3
    assert all(c.get("opener_suggestion") == "loved your FlowRider photo" for c in open_sheet)
    assert all(c.get("opener_warning") is None for c in open_sheet)


def test_observe_affirmative_mismatch_revokes_a_prior_match_until_sheet_closes():
    """The anti-flicker latch is not a permission to ignore real contrary evidence.

    A positively identified wrong item revokes a prior match and stays refused for that open
    composer. Closing it resets the epoch, so a newly opened correct sheet can verify afresh.
    """
    from operation_love.status import RunStatus

    class MismatchThenReopenDriver(_ObserveLikeIntentDriver):
        def observe_item_check(self, sheet, model_item_index):
            self.checked.append((sheet, model_item_index))
            if sheet == b"wrong-item-5-sheet":
                return ObserveItemCheck(
                    OBSERVE_ITEM_MISMATCH,
                    "you opened item 5, but this suggestion was written about item 3")
            return ObserveItemCheck(OBSERVE_ITEM_MATCH)

        def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
            self.gate()
            if on_like_intent is not None:
                on_like_intent(True, b"initial-item-3-sheet")
                on_like_intent(True, b"wrong-item-5-sheet")
                on_like_intent(True, b"later-item-3-frame")
                on_like_intent(False, None)
                on_like_intent(True, b"reopened-item-3-sheet")
            return True

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = MismatchThenReopenDriver(gate=_settled(status))
    calls = _record_state_transitions(status)
    svc = _RecordingOpenerService(suggestion="loved your FlowRider photo", index=3)
    Worker("bumble", driver, _ObserveDecider(), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    open_sheet = [c for c in calls if c.get("state") == "waiting_for_send"]
    assert open_sheet[0]["opener_suggestion"] == "loved your FlowRider photo"
    assert open_sheet[1]["opener_warning"]
    assert open_sheet[2]["opener_warning"]       # later match cannot erase real mismatch
    assert open_sheet[3]["opener_suggestion"] == "loved your FlowRider photo"  # reopen reset
    assert open_sheet[3]["opener_warning"] is None


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
    """Card 1 gets a suggestion and opens Hinge's comment sheet, but wait_for_decision then
    resolves to None -- a RESYNC. This mirrors the real driver: hinge.py's wait_for_decision can
    return None right after notifying on_like_intent(active=True, ...) (e.g. `if sent is None:
    return None`, reached before any matching close notice), and _observe_loop's own resync
    branch (`continue`, see its "Card changed without a corroborated decision" comment) does not
    touch the opener fields either -- so nothing clears the stale suggestion until the per-card
    reset at the top of the loop's NEXT iteration, which is exactly what these tests target.
    Card 2 has no numbered items at all, so no suggestion is made for it -- the "next card's
    suggestion fails or is absent" half of the reported failure mode."""
    supports_observe_like_intent = True
    accepts_opener = True

    def __init__(self, gate=None):
        super().__init__(2)
        self.cards = [_observe_card(0), Profile(photos=[b"x"], bio="bio1")]
        self.served = 0
        self.wait_calls = 0
        self.gate = gate

    def out_of_profiles(self):
        return self.served >= 2

    def current_profile(self):
        card = self.cards[self.served]
        self.served += 1
        return card

    def render_busy(self, message=None):
        pass

    def observe_item_mismatch(self, sheet, model_item_index):
        return ""

    def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
        self.wait_calls += 1
        if self.gate is not None:
            self.gate()
        if self.wait_calls == 1:
            if on_like_intent is not None:
                on_like_intent(True, b"card-one-like-sheet-frame")   # sheet opens, suggestion up
            return None                                              # resync -- no close notice
        return True                                                  # card 2: plain LIKE, no sheet


def test_observe_per_card_reset_clears_every_opener_field_from_the_previous_card():
    """THE regression test named in an earlier task, widened by doc 5.9's inversion from three
    fields to seven. After a card produces a suggestion that is never explicitly closed (a
    resync, not a dismiss or a send), the per-card reset for the NEXT card must clear ALL of
    them. Because that reset call explicitly names opener_suggestion, it opts itself out of
    RunStatus.set_app's own auto-clear (see status.py's comment on the safety net), so it must
    name the rest too -- or the hub would keep showing card 1's "like item 3 — the ridgeline
    photo" instruction while card 2 has no suggestion of its own at all, which under the
    inversion is worse than a stale caption: it is an instruction to like a specific item on
    somebody else's profile."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ResyncThenSilentSecondCardDriver(gate=_settled(status))
    store = FakeStore()
    calls = _record_calls_and_snapshots(status, "bumble")
    svc = _RecordingOpenerService(suggestion="about her dog", referenced="her dog", index=3,
                                  item_description="the ridgeline photo")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe", status=status).run()

    # The per-card reset is the FIRST "waiting" publish after that card's "capturing" one. It
    # cannot be picked out by its keys any more: the suggestion's own publishes are also
    # state="waiting" carrying the whole opener set, which is deliberate (every publisher states
    # all seven fields) and is exactly why this test locates the call by its POSITION in the
    # loop instead.
    capturing = [i for i, c in enumerate(calls) if c["fields"].get("state") == "capturing"]
    assert len(capturing) == 2, "one capture per card"
    card_two_reset = next(c for c in calls[capturing[1]:]
                          if c["fields"].get("state") == "waiting")["after"]
    assert card_two_reset["opener_suggestion"] is None
    assert card_two_reset["opener_referenced"] is None    # NOT left over from card 1's "her dog"
    assert card_two_reset["opener_item"] is None          # NOT still saying "like item 3"
    assert card_two_reset["opener_item_description"] is None
    assert card_two_reset["opener_warning"] is None
    assert card_two_reset["opener_pending"] is False


def test_observe_loop_finally_clears_a_still_live_suggestion_before_the_terminal_state():
    """THE regression test for _observe_loop's `finally`: if the run ends (Stop clicked, in
    this test right after the suggestion for the only card is published) while a suggestion
    is still live in AppStatus, the finally block's own clearing call -- which, like the
    per-card reset, explicitly names opener_suggestion and therefore also opts out of
    RunStatus.set_app's auto-clear -- must name every other opener field too.
    Snapshotted right after THAT specific call (identified by opener_suggestion being named
    with no accompanying `state` key -- the unique signature of this one call site in
    worker.py) rather than the final post-run() snapshot, because _finish_session's own
    state=stopped/out_of_profiles transition runs immediately afterward and would
    coincidentally re-clear the fields through the ORDINARY auto-clear path regardless of
    whether this call did its job -- masking the exact bug an adversarial review found here."""
    from operation_love.status import RunStatus

    stop_event = threading.Event()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")

    class _StopRightAfterSuggestionDriver(_ObserveLikeIntentDriver):
        def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
            result = super().wait_for_decision(timeout=timeout, should_stop=should_stop,
                                               on_like_intent=on_like_intent)
            stop_event.set()          # Stop lands the instant the sheet's suggestion is up
            return result

    driver = _StopRightAfterSuggestionDriver(anchor=b"the-actual-like-sheet-frame",
                                             gate=_settled(status))
    store = FakeStore()
    calls = _record_calls_and_snapshots(status, "bumble")
    svc = _RecordingOpenerService(suggestion="about her dog", referenced="her dog", index=3,
                                  item_description="the ridgeline photo")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           stop_event, mode="observe", status=status).run()

    finally_clear = next(c for c in calls
                         if "opener_suggestion" in c["fields"] and "state" not in c["fields"])
    after = finally_clear["after"]
    assert after["opener_suggestion"] is None
    assert after["opener_referenced"] is None
    assert after["opener_item"] is None
    assert after["opener_item_description"] is None
    assert after["opener_warning"] is None
    assert after["opener_pending"] is False


class _RecordingResultOpenerClient(FakeOpenerClient):
    """Same canned-success shape as FakeOpenerClient, but returns a FIXED opener string (not
    the f"hi {calls}" counter) and keeps every OpenerResult generate() actually produced --
    so a test can assert byte-for-byte against THAT object's .opener rather than a re-typed
    literal that might only match by coincidence."""
    def __init__(self, opener_text, cost_tokens=400):
        super().__init__(cost_tokens=cost_tokens)
        self.opener_text = opener_text
        self.results = []

    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        self.should_stops.append(should_stop)
        self.items.append(items)
        # Same reasoning as FakeOpenerClient.generate() above: a realistic, translatable pick,
        # not the dataclass's own untranslatable-by-default item_index/index_space.
        result = OpenerResult(opener=self.opener_text, referenced="r",
                              usage=Usage(input_tokens=self.cost_tokens),
                              model="gemini-test-model",
                              item_index=FIRST_ITEM_INDEX,
                              index_space=(INDEX_SPACE_MODEL_ITEMS if items is not None
                                           else INDEX_SPACE_PROFILE_PHOTOS))
        self.results.append(result)
        return result


def test_observe_suggestion_and_auto_like_type_the_identical_opener_the_client_returned():
    """OWNER REQUIREMENT: the observe banner's suggested text is a CANARY for auto mode --
    catching scaffolding words a model wrapped around the real opener only works if what the
    operator is shown to type is BYTE-FOR-BYTE what auto mode would actually send. This pins
    that fidelity end to end through the REAL OpenerService/OpenerPick pipeline (unlike
    _RecordingOpenerService above, a hand-written test double that never round-trips through
    OpenerPick.text at all) on both sinks named in the requirement:
      - AUTO mode: driver.like() receives OpenerResult.opener as its `opener` arg, unmodified.
      - OBSERVE mode: status.set_app(..., opener_suggestion=...) receives the identical
        OpenerResult.opener, unmodified.
    The fixture opener is deliberately not innocuous ASCII: it carries the kind of scaffolding
    a model might wrap a real line in ("Sure! Here's a great one:"), a curly apostrophe, and
    leading/trailing whitespace -- so a silent .strip()/fold/quote-normalization introduced by
    EITHER call site (and not the other) breaks the cross-check below instead of hiding behind
    two independently-"clean" literals that happen to already match.

    DOC 5.9 MADE THIS TEST MEAN SOMETHING IT DID NOT MEAN BEFORE. Byte-identical TEXT out of two
    modes that issued different REQUESTS was never much of a canary: auto sent numbered crops and
    asked the model to choose, observe sent scroll frames plus an anchor and asked it to describe
    one. So this now also asserts that both modes reached the client with the SAME request shape
    -- the numbered crops -- which is the property the text-fidelity assertion rests on."""
    from operation_love.status import RunStatus

    weird_opener = '  Sure! Here’s a great one: "Nice antlers."  '

    # --- AUTO: driver.like()'s opener kwarg must be exactly what generate() returned. ---
    auto_driver = FakeDriver(1)
    auto_driver.cards = [_observe_card()]
    auto_store = FakeStore()
    auto_client = _RecordingResultOpenerClient(weird_opener)
    auto_svc = OpenerService(auto_client, CostTracker(PRICING, None), auto_store, "s")
    _worker(auto_driver, FakeDecider("like"), auto_svc, auto_store).run()

    assert auto_client.results, "the fake client must have been called at least once"
    assert auto_driver.likes == [auto_client.results[0].opener]
    assert auto_driver.likes == [weird_opener]

    # --- OBSERVE: status.opener_suggestion must be exactly what generate() returned. ---
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    observe_driver = _ObserveLikeIntentDriver(gate=_settled(status))
    observe_store = FakeStore()
    observe_client = _RecordingResultOpenerClient(weird_opener)
    observe_svc = OpenerService(observe_client, CostTracker(PRICING, None), observe_store, "s")
    calls = _record_state_transitions(status)
    Worker("bumble", observe_driver, _ObserveDecider(), observe_svc, observe_store, "run1",
           _Pacing(), threading.Event(), mode="observe", status=status).run()

    assert observe_client.results, "the fake client must have been called at least once"
    waiting_for_send_call = next(c for c in calls if c.get("state") == "waiting_for_send")
    assert waiting_for_send_call["opener_suggestion"] == observe_client.results[0].opener
    assert waiting_for_send_call["opener_suggestion"] == weird_opener

    # The canary property itself: what auto mode types and what observe mode shows the
    # operator to type must be the IDENTICAL string, not merely each independently correct.
    assert auto_driver.likes[0] == waiting_for_send_call["opener_suggestion"]

    # ...and it rests on both modes having ASKED the same question. Same numbered crops, same
    # unnumbered context and same name -- one ItemRequest each.
    assert [i.items for i in auto_client.items] == [_OBSERVE_ITEMS]
    assert [i.items for i in observe_client.items] == [_OBSERVE_ITEMS]
    assert [i.context for i in observe_client.items] == [i.context for i in auto_client.items]
    assert [i.name for i in observe_client.items] == [i.name for i in auto_client.items]


class _TwoCardObserveLikeIntentDriver(FakeDriver):
    """Like _ObserveLikeIntentDriver, but serves TWO cards, each with a LIKE outcome that
    opens (and closes) Hinge's comment sheet -- lets a test drive maybe_opener() twice, once
    per profile, to prove a failure on card 1 doesn't poison card 2. Uses the real driver's
    two-arg on_like_intent contract (active, anchor) on both the open and the close call,
    matching what the real Hinge driver actually does."""
    supports_observe_like_intent = True
    accepts_opener = True

    def __init__(self, gate=None):
        super().__init__(2)
        self.cards = [_observe_card(0), _observe_card(1)]
        self.served = 0
        self.gate = gate

    def out_of_profiles(self):
        return self.served >= 2

    def current_profile(self):
        card = self.cards[self.served]
        self.served += 1
        return card

    def render_busy(self, message=None):
        pass

    def observe_item_mismatch(self, sheet, model_item_index):
        return ""

    def wait_for_decision(self, timeout=None, should_stop=None, on_like_intent=None):
        if self.gate is not None:
            self.gate()
        if on_like_intent is not None:
            on_like_intent(True, b"like-sheet-frame")
            on_like_intent(False, None)
        return True   # LIKE both cards


def test_advisory_suggestion_failure_never_stops_the_observe_run_end_to_end():
    """The headline contract for change A, driven through the REAL OpenerService (not a
    fake) with a client that fails to parse on EVERY call: max_attempts (5) would
    burn 5 real calls and, on exhaustion, set stop_requested -- which _observe_loop honours,
    ENDING the whole labelling session over a display-only failure. advisory=True must
    instead use the shorter advisory budget and leave stop_requested False, so BOTH cards'
    human decisions get processed and persisted -- the entire point of observe mode.

    Doc 5.9's inversion did not change any of that; it only moved WHEN the failing call
    happens (before the human acts, on its own thread) and made it happen on every card rather
    than only on hearted ones -- which makes the contract matter more, not less."""
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _TwoCardObserveLikeIntentDriver(gate=_settled(status))
    store = FakeStore()
    client = ParseErrorOpenerClient()          # every attempt fails to parse, forever
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    w = Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
              threading.Event(), mode="observe", status=status)
    w.run()

    # Card 1's bounded advisory retries exhaust the service (disabled=True), so
    # card 2's call short-circuits on `disabled` at the very top of maybe_opener() without
    # ever reaching the client again -- three real calls total, not the full AUTO budget.
    assert client.calls == svc.advisory_max_attempts == 3
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
# WHAT THE AUTO LOOP HANDS THE DRIVER, AND WHAT IT DOES WHEN THE DRIVER CANNOT HONOUR IT.
# This section used to be headed ANCHORED_OPENER and was about the driver's repair hatch: a
# callback the worker handed over so the driver could re-ask for text about whatever item its
# tap actually landed on. That hatch was removed on 2026-08-12 -- it repaired the TEXT while
# the LIKE still went to an item the model never chose (ops/OPENER-REDESIGN.md 5.6). What is
# left is the index the worker converts and passes, the stops it makes before calling the
# driver at all, and doc 5.6's stop for when the driver reports the miss itself (bottom of
# this file).
# ---------------------------------------------------------------------------------------

class _SequencedOpenerService:
    """Records every maybe_opener() call's kwargs and returns a SCRIPTED pick per call, in
    call order -- so a test can drive several opener calls through DIFFERENT results and tell
    which call produced which text. A scripted entry of None models a call that produced no
    opener at all."""
    stop_requested = False
    disabled = False

    def __init__(self, picks):
        self.picks = list(picks)
        self.calls = []

    def maybe_opener(self, run_id, app, profile, *, anchor=None, items=None,
                     should_stop=None, advisory=False):
        self.calls.append({"run_id": run_id, "app": app, "anchor": anchor, "items": items,
                           "should_stop": should_stop, "advisory": advisory})
        return self.picks[len(self.calls) - 1]


def test_auto_like_passes_the_translated_item_index_and_no_repair_callback():
    """Pins item_index: it must be whatever OpenerPick.capture_order_index says, never the raw
    pick.index and never a hardcoded 0 -- this pick names model item 3 in the (translatable)
    profile-photos space, so the driver must receive 2, not 3 and not 0.

    And pins what is NO LONGER passed. This test used to assert that the worker also handed over
    a callable `anchored_opener`, the driver's repair hatch for a targeting miss discovered at
    swipe time. That hatch is removed (ops/OPENER-REDESIGN.md 5.6): it repaired the TEXT while the
    LIKE still landed on an item the model never chose. FakeDriver.like's signature no longer
    accepts the keyword, so a worker that started passing one again would raise a TypeError here
    rather than silently reintroducing the substitution path."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = _SequencedOpenerService([
        OpenerPick("hi 1", index=3, referenced="r", index_space=INDEX_SPACE_PROFILE_PHOTOS),
    ])
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.likes == ["hi 1"]
    assert driver.like_item_indexes == [2]


# --- INDEX SPACES AT THE WORKER/DRIVER SEAM (ops/OPENER-REDESIGN.md 5.1/5.3/5.7) ------------
# `pick.index` counts the numbered images the MODEL was sent; driver.like(item_index=...)
# counts the frames the DRIVER captured. Between 2026-08-12 and this regression the worker
# handed one straight to the other, and because the driver's only bounds check is against its
# own frame count, an in-range 1-based item number resolved to a real, adjacent, WRONG frame
# and was reported on-target -- the comment landed one card down with full confidence.
#
# A LATER regression narrowed but did not close the gap: the worker started converting via
# `capture_order_index`, but when that conversion came back None (no sound conversion exists)
# it still called driver.like(item_index=None) and let the driver tap the first item and
# repair the message against whatever it landed on -- which still spent a like on an item the
# model never chose, just with text that matched it. The worker now refuses to call the driver
# AT ALL in that case (ops/OPENER-REDESIGN.md 5.3: "treat a missing table as a hard stop, never
# as a reason to fall back to a fixed coordinate") and stops the run instead. These tests pin
# the crossing itself, at the seam, in the worker, where both bugs lived.

def test_auto_like_translates_a_profile_photo_index_into_the_drivers_capture_order():
    """The legacy frame shape numbers profile.photos 1..N, and the driver's capture order is
    that same list 0-based, so model item 3 is capture-order index 2. The worker must send the
    CONVERTED value -- sending 3 would target the next frame down."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = _SequencedOpenerService([
        OpenerPick("hi 1", index=3, referenced="r", index_space=INDEX_SPACE_PROFILE_PHOTOS),
    ])
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.like_item_indexes == [2]


def test_auto_like_hands_a_model_item_number_to_the_driver_rather_than_converting_it():
    """REWRITTEN IN PLACE, 2026-08-12, when doc 5.5's counting navigation was wired. This test
    used to pin the opposite behaviour -- that a model-item pick was UNTRANSLATABLE and stopped
    the run -- and the reason it stopped was true at the time: nothing could turn that number
    into a tapped heart. `item_nav.navigate_to_item` now can, by walking UP from where the
    profile read left the card, so the number goes to the driver AS a model item number.

    What must NOT happen is a conversion. `capture_order_index` still returns None for this
    space, correctly: the model's number resolves to a HEART ORDINAL, not to a captured frame,
    and inventing a frame index for it is the exact bug the two index spaces exist to prevent.
    So the worker passes `model_item_index=3` and leaves `item_index` at None -- which is the one
    combination `hinge._like_comment_sheet` routes to counting navigation."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = _SequencedOpenerService([
        OpenerPick("hi 1", index=3, referenced="r", index_space=INDEX_SPACE_MODEL_ITEMS),
    ])
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.likes == ["hi 1"]
    assert driver.like_model_item_indexes == [3]      # the model's own number, unconverted
    assert driver.like_item_indexes == [None]         # ...and NO capture-order index beside it
    assert len(store.decisions) == 1


def test_auto_like_never_turns_an_absent_item_index_into_the_first_item():
    """ITEM_INDEX_ABSENT means the model could not pick an item. It is 0, and 0 is a perfectly
    legal FIRST FRAME in the driver's space, so passing it through would turn "no choice" into
    a confident like of item 1 -- which is exactly what happened before this regression. It
    must never even reach the driver -- let alone `_locate_target_heart` -- in any index
    space: the worker refuses to call driver.like() and stops the run instead."""
    for space in (INDEX_SPACE_PROFILE_PHOTOS, INDEX_SPACE_MODEL_ITEMS):
        driver = FakeDriver(1)
        svc = _SequencedOpenerService([
            OpenerPick("hi 1", index=ITEM_INDEX_ABSENT, referenced="r", index_space=space),
        ])
        Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
               threading.Event(), mode="auto").run()

        assert driver.likes == [], space
        assert driver.like_item_indexes == [], space


def test_auto_mode_stop_reason_names_absent_item_index_distinctly_from_an_unknown_space():
    """The operator's next move differs between the two refusal causes above: "the model never
    named an item" is an opener-layer problem (a prompt/model issue -- retrying may help),
    while "the model named one in a space this worker does not act on" is a wiring problem. A
    single stop_reason reading "item None" for both would hide that distinction from the
    operator, so the two must produce recognizably different text -- and both must publish
    stop_kind="opener", the same channel every other opener-triggered stop uses.

    REWRITTEN IN PLACE, 2026-08-12: the second half used to use INDEX_SPACE_MODEL_ITEMS, which
    was untranslatable before doc 5.5's counting navigation was wired and is now the space the
    worker acts on directly. What is left in the "cannot act on it" class is a space this build
    has never heard of, which is what a stale or future producer would emit -- and which must
    still degrade to a stop rather than to a guess."""
    from operation_love.status import RunStatus

    driver = FakeDriver(1)
    svc = _SequencedOpenerService([OpenerPick("hi 1", index=ITEM_INDEX_ABSENT, referenced="r")])
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "stopped"
    assert app["stop_kind"] == "opener"
    assert "no item number" in app["stop_reason"]
    assert driver.likes == []

    driver2 = FakeDriver(1)
    svc2 = _SequencedOpenerService([
        OpenerPick("hi 1", index=3, referenced="r", index_space="some_future_space"),
    ])
    status2 = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver2, FakeDecider("like"), svc2, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status2).run()

    app2 = status2.app_view("bumble")["app"]
    assert app2["state"] == "stopped"
    assert app2["stop_kind"] == "opener"
    assert "some_future_space" in app2["stop_reason"]
    assert "neither" in app2["stop_reason"]
    assert driver2.likes == []


# =====================================================================================
# THE REQUEST SHAPE: numbered crops, or a stop -- never the raw frames
# (ops/OPENER-REDESIGN.md 5.2/5.7)
# =====================================================================================

def test_auto_sends_the_enumerated_item_crops_as_the_request_not_the_scroll_frames():
    """Doc 5.2's whole point: with crops, image k IS item k, so the number the model answers
    with means something. When the capture enumerated the profile, the AUTO loop must build an
    ItemRequest from it and hand it to the opener call -- numbered items in model order, the
    unnumbered context tier after them, her name as text, the truncation flag -- and the raw
    scroll frames must not be what the model is looking at."""
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), FakeStore(), "s")
    driver = FakeDriver(1)
    driver.cards[0] = Profile(
        photos=[b"frame-0", b"frame-1", b"frame-2"],
        name="Ada", items=(b"crop-1", b"crop-2"), item_context=(b"vitals",),
        items_truncated=True)

    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert len(client.items) == 1
    request = client.items[0]
    assert request is not None, "the crops the driver enumerated must reach the client"
    assert request.items == (b"crop-1", b"crop-2")
    assert request.context == (b"vitals",)
    assert request.name == "Ada"
    assert request.truncated is True
    assert request.item_count == 2
    # The frames are still on the Profile for the ranker; they are simply not the request.
    assert b"frame-0" not in request.images


def test_auto_keeps_the_frame_shape_for_a_driver_that_enumerates_nothing():
    """A driver with no item space at all -- every non-Hinge driver today -- must be completely
    unaffected: no ItemRequest, no refusal, exactly the pre-existing call. "This capture has no
    items" and "this capture could not produce its items" are different facts and only the
    second one stops a run."""
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), FakeStore(), "s")
    driver = FakeDriver(1)

    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert client.items == [None]
    assert driver.likes == ["hi 1"]


def test_auto_stops_when_the_capture_could_not_produce_numbered_items():
    """The refusal doc 5.2 requires, at the one place the substitution would otherwise happen.
    A driver that enumerates and failed says so on the Profile; the worker must stop with that
    sentence rather than fall back to sending `profile.photos`, which would hand the model a
    numbering nothing downstream can act on. No opener call is made and no like is sent."""
    from operation_love.status import RunStatus

    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), FakeStore(), "s")
    driver = FakeDriver(1)
    driver.cards[0] = Profile(
        photos=[b"frame-0", b"frame-1"],
        items_unavailable="the card is not confirmed to be at its scroll top (confirmed_not_top)")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")

    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "stopped"
    assert app["stop_kind"] == "opener"
    assert "scroll top" in app["stop_reason"]
    assert "numbered items" in app["stop_reason"]
    assert client.calls == 0, "nothing may be billed for a request that cannot be built"
    assert driver.likes == []


def test_auto_does_not_stop_on_items_unavailable_when_openers_are_disabled():
    """Audit fix, "BUG 2" (2026-08-12): opener.enabled: false must not turn a driver's
    items_unavailable sentence into a hard stop -- there is no consumer for a numbered item list
    when openers are off, so its absence is not a failure. Mirrors
    test_auto_stops_when_the_capture_could_not_produce_numbered_items immediately above with the
    one variable that matters changed: OpenerService(client=None, ...), i.e. `disabled` from
    construction exactly like opener.enabled: false produces (see
    test_worker_with_opener_disabled_by_config_still_likes_normally_in_auto_mode). The driver-side
    fix (HingeDriver.set_opener_enabled) is what stops a real Hinge read from ever setting this
    field in the first place; this test pins the independent worker-side guard that must hold even
    if a driver enumerates anyway (a fake, an older driver, or a future one that forgets the hook)."""
    from operation_love.status import RunStatus

    svc = OpenerService(None, CostTracker(PRICING, None), FakeStore(), "s")   # openers disabled
    driver = FakeDriver(1)
    driver.cards[0] = Profile(
        photos=[b"frame-0", b"frame-1"],
        items_unavailable="the card is not confirmed to be at its scroll top (confirmed_not_top)")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")

    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] != "stopped"
    assert svc.stop_requested is False
    assert driver.likes == [None]     # liked normally, with no opener -- exactly as if openers
                                       # did not exist for this run at all


# --- THE items_unnumbered REGRESSION (found+fixed 2026-08-22, second pass, same day) -----------
# `items_unnumbered` is the third state `Profile` can be in: enumeration ran to completion and
# legitimately numbered nothing (a profile of all video, or a dwell burst that only ever covered
# one card of fifteen -- ops/STILL-PHOTO-DISCRIMINATOR.md 5d). The first pass that introduced it
# made a PASS decision sail past a zero-item profile, correctly. But it did that by leaving the
# whole state invisible to the AUTO loop's opener block -- which meant a LIKE decision fell
# through with `items` left `None`, all the way to `maybe_opener(..., items=None)`. Per that
# method's own docstring, `items=None` means "fall back to the raw scroll frames", exactly the
# ambiguity ops/OPENER-REDESIGN.md 5.2 exists to remove, and a live violation of the owner's
# never-substitute-liked-item rule: there is no verifiable item for that opener to have named.
# These tests pin the fix -- a LIKE decision now stops here, by call count, not just by message --
# and re-confirm the PASS side the first pass already got right.

def test_auto_stops_on_like_when_enumeration_numbered_nothing():
    """The regression itself. `maybe_opener` must never be reached -- not "reached and it
    returned nothing useful", literally never invoked -- and `driver.like` must never be called
    either. Checking call counts is the point: a test that only inspected the stop message would
    have passed on the old, buggy code path too (a stop can still fire later, from `pick is
    None`, after `maybe_opener` already ran on the wrong request)."""
    from operation_love.status import RunStatus

    svc = _SequencedOpenerService([])   # any call at all is a bug; nothing is scripted to return
    driver = FakeDriver(1)
    driver.cards[0] = Profile(
        photos=[b"frame-0"], items=(),
        items_unnumbered="15 selectable card(s) were considered; 15 could not be judged because "
                          "the one dwell burst this capture takes never covered them.")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")

    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "stopped"
    assert app["stop_kind"] == "opener"
    assert svc.calls == [], "maybe_opener must not be invoked when nothing was numbered"
    assert driver.likes == [], "no like may be sent with no verifiable item to attach it to"


def test_auto_does_not_stop_on_pass_when_enumeration_numbered_nothing():
    """The other half of the same profile: a PASS decision is exactly the improvement this
    whole state exists to preserve (a profile of videos is a normal outcome), and it must sail
    past the zero-item profile and keep running -- unaffected by the LIKE-path stop added above,
    because a PASS decision never reaches that opener code at all."""
    svc = _SequencedOpenerService([])
    driver = FakeDriver(2)
    driver.cards[0] = Profile(
        photos=[b"frame-0"], items=(),
        items_unnumbered="15 selectable card(s) were considered; 15 could not be judged because "
                          "the one dwell burst this capture takes never covered them.")

    Worker("bumble", driver, FakeDecider("dislike"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.dislikes == 2, "both cards were reached -- the run did not stop on card one"
    assert svc.calls == []
    assert driver.out_of_profiles()


def test_items_unnumbered_stop_reason_is_distinguishable_from_items_unavailable():
    """The two stops share `stop_kind == "opener"` (the hub branches on that alone), so the
    `stop_reason` TEXT is the only thing that tells an operator which situation they are in: an
    `items_unavailable` stop means the capture itself failed, while an `items_unnumbered` stop
    means the capture worked and nothing survived policy -- different operator next-steps, so
    the wording must not collide. This also pins that the unnumbered reason quotes the profile's
    own sentence rather than a hardcoded cause (doc STILL-PHOTO-DISCRIMINATOR.md 5d's own
    standing rule: guidance must derive from the condition it describes)."""
    from operation_love.status import RunStatus

    unavailable_status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    unavailable_driver = FakeDriver(1)
    unavailable_driver.cards[0] = Profile(
        photos=[b"frame-0"],
        items_unavailable="the card is not confirmed to be at its scroll top (confirmed_not_top)")
    Worker("bumble", unavailable_driver, FakeDecider("like"), _SequencedOpenerService([]),
           FakeStore(), "run1", _Pacing(), threading.Event(),
           mode="auto", status=unavailable_status).run()
    unavailable_reason = unavailable_status.app_view("bumble")["app"]["stop_reason"]

    unnumbered_status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    unnumbered_driver = FakeDriver(1)
    unnumbered_driver.cards[0] = Profile(
        photos=[b"frame-0"], items=(),
        items_unnumbered="15 selectable card(s) were considered; 15 could not be judged because "
                          "the one dwell burst this capture takes never covered them.")
    Worker("bumble", unnumbered_driver, FakeDecider("like"), _SequencedOpenerService([]),
           FakeStore(), "run1", _Pacing(), threading.Event(),
           mode="auto", status=unnumbered_status).run()
    unnumbered_reason = unnumbered_status.app_view("bumble")["app"]["stop_reason"]

    assert "15 selectable card(s) were considered" in unnumbered_reason
    assert unnumbered_reason != unavailable_reason
    assert "confirmed_not_top" not in unnumbered_reason
    assert "15 selectable card(s)" not in unavailable_reason


def test_all_video_profile_reaches_neither_the_opener_nor_the_targeting_path():
    """Re-pins the property `test_all_video_profile_fails_before_opener_or_targeting`
    (tests/test_hinge_item_capture.py, added 2026-08-16) used to encode at the DRIVER level,
    before that test was renamed to `test_all_video_profile_numbers_nothing_but_does_not_stop_the_run`
    on 2026-08-22 (an all-video profile is no longer a driver-level failure -- it is
    `items_unnumbered`, not `items_unavailable`, precisely so a PASS decision does not stop). The
    "fails before opener or targeting" half of that property did not move with it: it belongs
    here, at the WORKER level, and it is specifically a LIKE-decision property, since a PASS
    decision was never at risk of reaching either. `driver.like` is the single call both the
    opener request (via `item_index=`/`model_item_index=`) and the targeting path route through
    -- so asserting it was never invoked pins both at once."""
    from operation_love.status import RunStatus

    svc = _SequencedOpenerService([])
    driver = FakeDriver(1)
    driver.cards[0] = Profile(
        photos=[b"frame-0", b"frame-1"], items=(),
        items_unnumbered="4 selectable card(s) were considered; 4 could not be judged because "
                          "video_mute_v1 matched every one of them.")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")

    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    assert svc.calls == [], "the opener path was never reached"
    assert driver.likes == [] and driver.like_item_indexes == [] \
        and driver.like_model_item_indexes == [], "the targeting path was never reached either"
    assert status.app_view("bumble")["app"]["state"] == "stopped"


# --- DOC 5.6'S HARD STOP: a targeting/verification miss stops the run --------------------------
# WHAT THESE REPLACE. Five tests used to live here pinning the `anchored_opener` repair callback:
# the worker handed the driver a closure, and the driver invoked it once its own heart targeting
# missed, to re-ask the model for text about whatever item the sheet had actually opened on. That
# callback was removed on 2026-08-12. It repaired the TEXT while the LIKE still went to an item
# the model never chose, which is exactly the substitution ops/OPENER-REDESIGN.md 5.6 forbids
# ("no falling back to `hearts[0]`, no 'closest reachable item', no rewriting the opener to match
# whatever we hit"). Their coverage is not deleted, it is inverted: the same situations those
# tests described -- targeting missed, or the sheet turned out to show a different item -- are now
# the stop the tests below pin. The removal itself is pinned by FakeDriver.like's signature (no
# such keyword) and by test_hinge_observe.py's `..._no_longer_accepts_an_anchored_opener_callback`.
class _TargetingMissDriver(FakeDriver):
    """Hinge-shaped: its like() reports it could not put the like on the chosen item.

    Raises the DRIVER-AGNOSTIC base.ItemTargetingError rather than a Hinge symbol, which is the
    contract the worker is written against -- worker.py must not have to import a concrete driver
    to recognise the one failure class with a specific, non-error stop to render."""
    halt_on_error = True

    def __init__(self, n, *, stage="navigate", intended=4, actual=None,
                 index_space="model_items", message="could not reach it"):
        super().__init__(n)
        self.snapshotted = []
        self._exc = ItemTargetingError(message, stage=stage, intended=intended, actual=actual,
                                       index_space=index_space)

    def like(self, opener=None, item_index=None, *, model_item_index=None):
        self.likes.append(opener)          # recorded so a test can prove it was CALLED and failed
        raise self._exc

    def snapshot_failure(self, exc):
        self.snapshotted.append(exc)


def _run_targeting_stop(driver, *, app="bumble"):
    """Run one auto profile against `driver` and hand back the published app status + the store."""
    from operation_love.status import RunStatus
    svc = _SequencedOpenerService([
        OpenerPick("hi 1", index=1, referenced="r", index_space=INDEX_SPACE_PROFILE_PHOTOS),
    ])
    store = FakeStore()
    status = RunStatus("run1", [app], min_labels=1, mode="auto")
    Worker(app, driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()
    return status.app_view(app)["app"], store


def test_a_targeting_miss_stops_the_run_instead_of_crashing_it():
    """The headline. The driver could not put the like on the item the opener was written about,
    so it put it nowhere -- and that is a DECISION the bot made correctly under a standing rule,
    not a crash. Left to run()'s generic handler it would render as state="error" with a single
    traceback line, and an operator who cannot tell a correct refusal from a bug soon stops
    reading red banners. It stops, with the reason on the hub.

    stop_kind="targeting" REPLACED "opener" here (the hub workflow, per doc 5.6's own handover
    note). "opener" was the channel the hub already branched on, and its branch is titled
    "opener capacity exhausted": nothing was exhausted, the opener itself is fine, and the
    operator's next move is to go and read the phone rather than to check a quota. The value is
    asserted rather than merely "not None" because the hub renders a different box for each kind
    and an unrecognized one falls back to neutral wording (assets/hub.html)."""
    app, _store = _run_targeting_stop(_TargetingMissDriver(2))

    assert app["state"] == "stopped"
    assert app["stop_kind"] == "targeting"
    assert app["stop_kind"] != "opener"    # not the capacity/exhaustion channel -- see above
    assert app["error"] is None            # NOT an error banner -- this is a clean stop


def test_the_targeting_stop_records_intended_and_actual():
    """Doc 5.6: "stop the run with intended and actual item both recorded". Both come off the
    exception's own fields rather than being parsed back out of its prose, so the stop record
    cannot drift out of agreement with the stop it describes."""
    app, _store = _run_targeting_stop(
        _TargetingMissDriver(2, stage="verify", intended=4, actual=6,
                             message="the like sheet is NOT showing it"))

    reason = app["stop_reason"]
    assert "item 4" in reason and "item 6" in reason
    assert "Intended" in reason and "Actual" in reason
    assert "model_items" in reason                       # which numbering those two are in
    assert "the like sheet is NOT showing it" in reason  # the driver's own words, kept whole


def test_the_targeting_stop_says_so_when_nothing_was_ever_reached():
    """"we could not reach item 4" and "we reached item 6 while aiming at item 4" are different
    diagnoses, and the second is a much stronger signal that the item list itself is wrong. An
    absent `actual` must therefore say it is absent rather than render as a number."""
    app, _store = _run_targeting_stop(_TargetingMissDriver(2, stage="navigate", intended=4,
                                                          actual=None))

    assert "never got far enough" in app["stop_reason"]


def test_the_targeting_stop_does_not_record_the_like_as_a_decision():
    """The like was NOT sent, so counting it would corrupt both the stats and the daily rate
    limit -- the same rule the pre-existing "do not record a phantom if like() raised" comment
    states. Nothing swipes after it either: the run is over, with two profiles left unseen."""
    driver = _TargetingMissDriver(3)
    app, store = _run_targeting_stop(driver)

    assert store.decisions == []
    assert driver.likes == ["hi 1"]        # exactly one attempt, on the first profile
    assert driver.dislikes == 0
    assert app["state"] == "stopped"


def test_the_targeting_stop_still_snapshots_the_screen():
    """The screen IS the diagnosis here -- doc 5.6 leaves it exactly as it was (a scrolled
    profile, or an open sheet with nothing typed) precisely so it can be read back. Catching the
    exception must not cost the failure snapshot run()'s generic handler would have taken, and
    that snapshot has to happen while the transport is still live, i.e. inside the loop."""
    driver = _TargetingMissDriver(2)
    _run_targeting_stop(driver)

    assert len(driver.snapshotted) == 1
    assert isinstance(driver.snapshotted[0], ItemTargetingError)


def test_an_ordinary_driver_failure_is_still_an_error_not_a_targeting_stop():
    """The catch is narrow on purpose. A stuck deck, a missed Send tap or any other unexpected
    action failure is NOT a decision the bot made correctly, and must keep its red error banner
    and its traceback. Only ItemTargetingError -- "we could not honour the item the model chose"
    -- earns the clean stop."""
    from operation_love.status import RunStatus
    driver = RaisingLikeDriver(2)
    svc = _SequencedOpenerService([
        OpenerPick("hi 1", index=1, referenced="r", index_space=INDEX_SPACE_PROFILE_PHOTOS),
    ])
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "error"
    assert app["stop_kind"] is None


class _ObserveNeverLikesDriver(_ObserveLikeIntentDriver):
    """Observe-capable, and its like() is a tripwire rather than an implementation."""

    def like(self, opener=None, item_index=None, *, model_item_index=None):
        raise AssertionError("observe mode must never call driver.like()")


def test_observe_mode_cannot_reach_the_targeting_stop_at_all():
    """Structural, not incidental: worker.py calls driver.like() from exactly one place, inside
    _auto_loop, and the new handler wraps that one call. _observe_loop never calls it at all --
    the human taps Send Like with the app's own controls -- so there is no code path by which
    this change could touch observe's behaviour. Pinned with a driver whose like() would blow the
    test up if it were ever reached."""
    driver = _ObserveNeverLikesDriver()
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, _ObserveDecider(), svc, store, "run1", _Pacing(),
           threading.Event(), mode="observe").run()

    assert driver.likes == []              # never called, so the tripwire never fired
    assert store.labels                     # and the human's own decision still got recorded


# --- run-level provenance of the numbering licence (owner decision 2026-08-21) ----------
# Numbering behaves identically whether it was licensed by a measured held-out bound or by the
# owner's accepted, UNMEASURED centered-autoplay assumption. The difference can therefore only
# reach a person in words, which is why it is announced rather than merely recorded.

@pytest.fixture
def _licence_slot():
    """Numbering readiness is process-global: never let one test license the next one."""
    from operation_love import targeting_policy as tp

    tp._reset_installed_still_photo_bound_for_tests()
    yield tp
    tp._reset_installed_still_photo_bound_for_tests()


def _assumption_record(tp):
    return tp.StillPhotoAssumptionAcceptance(
        acceptance=tp.STILL_PHOTO_CENTERED_AUTOPLAY_ASSUMPTION,
        accepted_at="2026-08-21", device="synthetic-pixel", hinge_version_name="10.0.1",
        rationale="owner judged the held-out campaign not worth ~420 real passes")


def _measured_record(tp):
    return tp.StillPhotoBoundSummary(
        ground_truth_channel=tp.STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL, human_ground_truth=True,
        video_cards=60, video_accepts=0, photo_cards=60, photo_false_refusals=3,
        max_video_exact_run_s=1.5, artifact_sha256="a" * 64, device="synthetic-pixel",
        hinge_version_name="10.0.1")


def _run_hinge_observe(status):
    driver = _ObserveLikeIntentDriver(gate=_settled(status, app="hinge"))
    store = FakeStore()
    Worker("hinge", driver, _ObserveDecider(), _RecordingOpenerService(), store, "run1",
           _Pacing(), threading.Event(), mode="observe", status=status).run()
    return store


def test_a_run_licensed_by_an_assumption_says_so_in_the_log_and_on_the_hub(
        capsys, _licence_slot):
    """The honesty requirement: an UNMEASURED licence is news on EVERY run, not once ever."""
    from operation_love.status import RunStatus

    tp = _licence_slot
    tp.install_accepted_still_photo_assumption(_assumption_record(tp))
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="observe")

    store = _run_hinge_observe(status)

    out = capsys.readouterr().out
    assert "UNMEASURED assumption (centered autoplay)" in out
    assert "no video false-accept rate has been measured" in out
    # Announced exactly once: run() calls it OUTSIDE the restart loop, so a restart cannot
    # turn a standing fact into a per-card banner.
    assert out.count("targeted suggestions enabled under an UNMEASURED assumption") == 1
    # And it reaches the hub as a run-level field, not as a per-card opener warning (which the
    # banner renders in the WAIT style the owner's status-indicator rule reserves for "hands off").
    view = status.app_view("hinge")["app"]
    assert view["targeting_licence_notice"] == tp.STILL_PHOTO_ASSUMPTION_OPERATOR_NOTICE
    assert view["opener_warning"] is None
    # Labelling is completely unaffected: this is context, never a gate.
    assert len(store.labels) == 1


def test_a_run_licensed_by_a_measured_bound_announces_no_assumption(capsys, _licence_slot):
    """The provenance line must be absent when there is nothing unmeasured to confess."""
    from operation_love.status import RunStatus

    tp = _licence_slot
    tp.install_verified_still_photo_bound(_measured_record(tp))
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="observe")

    _run_hinge_observe(status)

    out = capsys.readouterr().out
    assert "UNMEASURED" not in out
    assert "false-accept rate" not in out
    assert status.app_view("hinge")["app"]["targeting_licence_notice"] is None


def test_an_unlicensed_run_announces_no_provenance_at_all(capsys, _licence_slot):
    from operation_love.status import RunStatus

    status = RunStatus("run1", ["hinge"], min_labels=1, mode="observe")

    _run_hinge_observe(status)

    assert "UNMEASURED" not in capsys.readouterr().out
    assert status.app_view("hinge")["app"]["targeting_licence_notice"] is None


def test_the_hinge_numbering_licence_is_never_announced_on_another_platform(
        capsys, _licence_slot):
    """The licence is Hinge's numbered-item readiness; a Bumble run must not claim it."""
    from operation_love.status import RunStatus

    tp = _licence_slot
    tp.install_accepted_still_photo_assumption(_assumption_record(tp))
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="observe")
    driver = _ObserveLikeIntentDriver(gate=_settled(status))
    Worker("bumble", driver, _ObserveDecider(), _RecordingOpenerService(), FakeStore(), "run1",
           _Pacing(), threading.Event(), mode="observe", status=status).run()

    assert "UNMEASURED" not in capsys.readouterr().out
    assert status.app_view("bumble")["app"]["targeting_licence_notice"] is None


class _UncalibratedHingeObserveDriver(_ObserveLikeIntentDriver):
    """Hinge with a still-photo licence but no `targeting_calibration` — the reported state."""

    def targeted_suggestion_blocker(self):
        return ("apps.hinge.targeting_calibration is unavailable (not configured in "
                "config.yaml); no opener text is offered")


def _run_uncalibrated_hinge_observe(status):
    driver = _UncalibratedHingeObserveDriver(gate=_settled(status, app="hinge"))
    Worker("hinge", driver, _ObserveDecider(), _RecordingOpenerService(), FakeStore(), "run1",
           _Pacing(), threading.Event(), mode="observe", status=status).run()


def test_the_targeting_setup_notice_names_the_calibration_once_numbering_is_licensed(
        capsys, _licence_slot):
    """BUG REPORT 2026-08-22. The setup notice must name the step that is actually left.

    Both operator surfaces stated the still-photo prerequisite unconditionally. That was true
    only while nothing licensed numbering; after the owner's centered-autoplay acceptance
    installed a licence it named a prerequisite ALREADY satisfied and never named the one
    remaining step, so a run whose only gate was "capture the calibration" read as an upstream
    block and the calibration stayed uncaptured. The notice is derived from the installed
    licence for exactly this reason.
    """
    from operation_love.status import RunStatus

    tp = _licence_slot
    tp.install_accepted_still_photo_assumption(_assumption_record(tp))
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="observe")

    _run_uncalibrated_hinge_observe(status)

    out = capsys.readouterr().out
    assert "targeted opener suggestions need setup" in out
    assert tp.TARGETING_SETUP_NEXT_STEP_CALIBRATE in out
    # The satisfied prerequisite is not restated: that sentence is what sent the owner upstream.
    assert tp.TARGETING_SETUP_NEXT_STEP_BLOCKED not in out
    # And it reaches the hub, which cannot work the branch out for itself (the licence lives in
    # this process, installed by config.validate()).
    view = status.app_view("hinge")["app"]
    assert view["targeting_setup_next_step"] == tp.TARGETING_SETUP_NEXT_STEP_CALIBRATE


def test_the_targeting_setup_notice_names_the_still_photo_proof_while_nothing_licenses_numbering(
        capsys, _licence_slot):
    """The other branch, unchanged: with no licence, the calibration is not the next step."""
    from operation_love.status import RunStatus

    tp = _licence_slot
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="observe")

    _run_uncalibrated_hinge_observe(status)

    out = capsys.readouterr().out
    assert tp.TARGETING_SETUP_NEXT_STEP_BLOCKED in out
    assert tp.TARGETING_SETUP_NEXT_STEP_CALIBRATE not in out
    view = status.app_view("hinge")["app"]
    assert view["targeting_setup_next_step"] == tp.TARGETING_SETUP_NEXT_STEP_BLOCKED


def test_a_calibrated_run_publishes_no_targeting_setup_step(_licence_slot):
    """No blocker, no guidance: the field exists to explain a blocker, never as decoration."""
    from operation_love.status import RunStatus

    tp = _licence_slot
    tp.install_accepted_still_photo_assumption(_assumption_record(tp))
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="observe")

    _run_hinge_observe(status)

    assert status.app_view("hinge")["app"]["targeting_setup_next_step"] is None
