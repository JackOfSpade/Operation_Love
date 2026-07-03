"""Worker + OpenerService tests with fakes — no Playwright/emulator/SDK/network."""
import threading
import time

from operation_love.costing import CostTracker, ModelPricing, Usage
from operation_love.drivers.base import DatingAppDriver, DriverClosed
from operation_love.opener.opener import OpenerResult
from operation_love.opener.service import OpenerService
from operation_love.limits import RateLimiter
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import Decision
from operation_love.worker import Worker

PRICING = {"claude-opus-4-8": ModelPricing(input=5.0, output=25.0)}


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
    def generate(self, profile, style):
        self.calls += 1
        return OpenerResult(opener=f"hi {self.calls}", referenced="r",
                            usage=Usage(input_tokens=self.cost_tokens), model="claude-opus-4-8")


class SlowOpenerClient(FakeOpenerClient):
    def generate(self, profile, style):
        time.sleep(0.05)
        return super().generate(profile, style)


class BillingErrClient:
    def generate(self, profile, style):
        raise Exception("Your credit balance is too low to access the Anthropic API")


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
    def label_count(self): return len(self.labels)
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
    svc = OpenerService(client, tracker, FakeStore(), "style", on_exhausted="stop")
    assert svc.maybe_opener("r", "bumble", Profile()).text == "hi 1"   # first allowed
    assert svc.maybe_opener("r", "hinge", Profile()) is None           # second over budget
    assert client.calls == 1 and svc.stop_requested is True


def test_budget_caps_openers_across_concurrent_workers():
    tracker = CostTracker(PRICING, run_budget_usd=0.001)   # first call spends past cap
    client = SlowOpenerClient(cost_tokens=400)
    store = FakeStore()
    svc = OpenerService(client, tracker, store, "style", on_exhausted="stop")
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


def test_out_of_credit_disables_service():
    svc = OpenerService(BillingErrClient(), CostTracker(PRICING, None), FakeStore(), "s", "stop")
    assert svc.maybe_opener("r", "bumble", Profile()) is None
    assert svc.disabled is True and svc.stop_requested is True


def test_swipe_without_opener_mode():
    tracker = CostTracker(PRICING, run_budget_usd=0.001)
    svc = OpenerService(FakeOpenerClient(400), tracker, FakeStore(), "s", on_exhausted="swipe_without_opener")
    svc.maybe_opener("r", "bumble", Profile())          # first spends
    assert svc.maybe_opener("r", "bumble", Profile()) is None
    assert svc.disabled is True and svc.stop_requested is False   # keep swiping, no stop


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
    # Bumble-style driver: no swipe-time opener -> never call Claude (no wasted credits).
    driver = FakeDriver(3)
    driver.accepts_opener = False
    store = FakeStore()
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert driver.likes == [None, None, None]      # liked, but with no opener
    assert client.calls == 0 and store.openers == [] and store.spend == []


def test_worker_stops_when_budget_exhausted():
    driver = FakeDriver(5)
    store = FakeStore()
    client = FakeOpenerClient(cost_tokens=400)  # $0.002/call
    svc = OpenerService(client, CostTracker(PRICING, run_budget_usd=0.001), store, "s", on_exhausted="stop")
    _worker(driver, FakeDecider("like"), svc, store).run()
    # The first opener is allowed, then the over-cap spend stops the run before
    # a second profile is swiped.
    assert client.calls == 1
    assert driver.likes == ["hi 1"]
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


def test_auto_mode_halts_on_unexpected_for_any_app_even_without_halt_flag():
    """AUTO mode must STOP on any unexpected error for EVERY app — including a plain driver with
    no halt_on_error flag (Bumble) — so an autonomous run never keeps swiping blindly. It must
    halt on the FIRST error (no restart retries) and set the stop so buffered data is saved."""
    class BoomDriver(FakeDriver):
        def __init__(self, n):
            super().__init__(n)
            self.calls = 0
        def next_profile(self):
            self.calls += 1
            raise RuntimeError("unexpected boom")

    driver = BoomDriver(3)
    assert not hasattr(driver, "halt_on_error") or driver.halt_on_error is False
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    stop = threading.Event()
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           stop, mode="auto", max_restarts=5).run()

    assert driver.calls == 1        # halted on the first error — did NOT restart-and-retry
    assert stop.is_set()            # stop set (halt path), so the supervisor saves buffered data
    assert driver.closed


def test_observe_mode_keeps_restart_resilience_for_non_halt_driver(monkeypatch):
    """Observe mode is user-driven and low-risk, so a driver that doesn't opt into halt (Bumble)
    keeps restart resilience there: a transient error retries rather than halting the session."""
    import operation_love.worker as wmod
    monkeypatch.setattr(wmod, "human_cooldown", lambda s: 0)   # no backoff sleep in the test

    class FlakyObserve(FakeDriver):
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


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    sys.exit(1 if failed else 0)
