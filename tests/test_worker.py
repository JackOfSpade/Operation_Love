"""Worker + OpenerService tests with fakes — no Playwright/emulator/SDK/network."""
import threading

from operation_love.costing import CostTracker, ModelPricing, Usage
from operation_love.drivers.base import DatingAppDriver
from operation_love.opener.opener import OpenerResult
from operation_love.opener.service import OpenerService
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
    def like(self, opener=None): self.likes.append(opener)
    def dislike(self): self.dislikes += 1
    def out_of_profiles(self): return self.i >= len(self.cards)
    def close(self): self.closed = True


class FakeDecider:
    def __init__(self, decision="dislike"):
        self.decision = decision
    def decide(self, profile):
        return Decision(decision=self.decision, score=0.9, embedding=[0.1, 0.2], source="ranker")


class FakeOpenerClient:
    def __init__(self, cost_tokens=400):
        self.calls = 0; self.cost_tokens = cost_tokens
    def generate(self, profile, style):
        self.calls += 1
        return OpenerResult(opener=f"hi {self.calls}", referenced="r",
                            usage=Usage(input_tokens=self.cost_tokens), model="claude-opus-4-8")


class BillingErrClient:
    def generate(self, profile, style):
        raise Exception("Your credit balance is too low to access the Anthropic API")


class FakeStore:
    def __init__(self):
        self.decisions, self.labels, self.spend, self.openers = [], [], [], []
    def load_labels(self): return []
    def add_label(self, run_id, app, liked, embedding, source="manual", **k): self.labels.append((app, liked))
    def record_decision(self, run_id, app, decision, score): self.decisions.append((app, decision))
    def record_opener(self, run_id, app, model, opener, referenced): self.openers.append(opener)
    def record_spend(self, run_id, model, usage, cost): self.spend.append(cost)
    def label_count(self): return len(self.labels)
    def flush(self): pass
    def close(self): pass


class _Pacing:
    min_delay_s = 0.0
    max_delay_s = 0.0


def _worker(driver, decider, service, store):
    return Worker("bumble", driver, decider, service, store, "run1", _Pacing(), threading.Event())


# --- OpenerService (global budget) --------------------------------------
def test_budget_caps_openers_globally():
    tracker = CostTracker(PRICING, run_budget_usd=0.001)   # 400 input tok = $0.002 > cap
    client = FakeOpenerClient(cost_tokens=400)
    svc = OpenerService(client, tracker, FakeStore(), "style", on_exhausted="stop")
    assert svc.maybe_opener("r", "bumble", Profile()) == "hi 1"   # first allowed
    assert svc.maybe_opener("r", "hinge", Profile()) is None      # second over budget
    assert client.calls == 1 and svc.stop_requested is True


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
    assert len(store.decisions) == 3 and len(store.labels) == 3
    assert driver.opened and driver.closed


def test_worker_likes_with_openers():
    driver = FakeDriver(3)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert driver.likes == ["hi 1", "hi 2", "hi 3"]
    assert len(store.openers) == 3 and len(store.spend) == 3


def test_worker_stops_when_budget_exhausted():
    driver = FakeDriver(5)
    store = FakeStore()
    client = FakeOpenerClient(cost_tokens=400)  # $0.002/call
    svc = OpenerService(client, CostTracker(PRICING, run_budget_usd=0.001), store, "s", on_exhausted="stop")
    _worker(driver, FakeDecider("like"), svc, store).run()
    # 1st profile: opener sent; 2nd: budget hit -> None + stop -> loop ends
    assert client.calls == 1
    assert driver.likes == ["hi 1", None]
    assert driver.closed


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
