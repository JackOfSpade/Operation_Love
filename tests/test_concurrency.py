"""Two app workers run concurrently in one process sharing the store (offline)."""
import threading

from operation_love.costing import CostTracker, ModelPricing, Usage
from operation_love.drivers.base import DatingAppDriver
from operation_love.opener.opener import OpenerResult
from operation_love.opener.service import OpenerService
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import Decision
from operation_love.worker import Worker

PRICING = {"claude-opus-4-8": ModelPricing(input=5.0, output=25.0)}


class _Driver(DatingAppDriver):
    def __init__(self, n):
        self.n = n; self.i = 0; self.closed = False
    def open_session(self): pass
    def next_profile(self):
        if self.i >= self.n:
            return None
        self.i += 1; return Profile(photos=[b"x"])
    def out_of_profiles(self): return self.i >= self.n
    def like(self, opener=None): pass
    def dislike(self): pass
    def close(self): self.closed = True


class _Decider:
    def decide(self, profile):
        return Decision("dislike", 0.1, [0.1], "ranker")


class _Store:
    def __init__(self):
        self._lock = threading.Lock()
        self.decisions = 0
    def record_decision(self, *a, **k):
        with self._lock:
            self.decisions += 1
    def record_profile(self, *a, **k): pass
    def add_label(self, *a, **k): pass
    def count_today(self, app): return 0


class _Pacing:
    swipe_delay_s = 0.0


def test_two_workers_run_concurrently_and_share_store():
    store = _Store()
    stop = threading.Event()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    d1, d2 = _Driver(4), _Driver(6)
    w1 = Worker("bumble", d1, _Decider(), svc, store, "r", _Pacing(), stop, mode="auto")
    w2 = Worker("hinge", d2, _Decider(), svc, store, "r", _Pacing(), stop, mode="auto")

    w1.start(); w2.start()
    w1.join(timeout=10); w2.join(timeout=10)

    assert not w1.is_alive() and not w2.is_alive()
    assert d1.i == 4 and d2.i == 6 and d1.closed and d2.closed
    assert store.decisions == 10           # both workers' swipes recorded to the shared store


class _OpenerClient:
    """One opener costs $0.002 at the PRICING below (400 input tok * $5/MTok)."""
    def generate(self, profile, style):
        return OpenerResult(opener="hi", referenced="r",
                            usage=Usage(input_tokens=400), model="claude-opus-4-8")


class _LikeDecider:
    def decide(self, profile):
        return Decision("like", 0.9, [0.1], "ranker")


class _SpendStore(_Store):
    """Adds the spend/opener sinks the like path needs; records (app, decision) tuples."""
    def __init__(self):
        super().__init__()
        self.rows = []
    def record_decision(self, run_id, app, decision, score, source="auto"):
        with self._lock:
            self.rows.append((app, decision))
    def record_opener(self, *a, **k): pass
    def record_spend(self, *a, **k): pass


def test_one_worker_budget_exhaustion_stops_the_other_worker():
    """Headline multi-worker safety contract: one shared stop_event + one shared OpenerService.
    When worker A exhausts the GLOBAL budget (on_exhausted='stop'), it sets stop_requested ->
    the shared stop_event; worker B, parked in its loop, then halts WITHOUT swiping (and without
    spending more), because the supervisor hands every worker the SAME stop_event."""
    class _Spender(DatingAppDriver):
        def __init__(self, b_parked):
            self.b_parked = b_parked
            self.i = 0; self.likes = []; self.dislikes = 0; self.closed = False
        def open_session(self): pass
        def next_profile(self):
            self.b_parked.wait(timeout=5)         # act only once B is parked (determinism)
            if self.i >= 1:
                return None
            self.i += 1; return Profile(photos=[b"x"])
        def out_of_profiles(self): return False
        def like(self, opener=None, item_index=0): self.likes.append(opener)
        def dislike(self): self.dislikes += 1
        def close(self): self.closed = True

    class _Gated(DatingAppDriver):
        def __init__(self, stop):
            self.stop = stop; self.parked = threading.Event()
            self.likes = []; self.dislikes = 0; self.closed = False
        def open_session(self): pass
        def next_profile(self):
            self.parked.set()                     # signal B is in its loop, waiting
            self.stop.wait(timeout=5)             # released only when A exhausts -> stop set
            return Profile(photos=[b"x"])         # returns AFTER stop; worker breaks before acting
        def out_of_profiles(self): return False
        def like(self, opener=None, item_index=0): self.likes.append(opener)
        def dislike(self): self.dislikes += 1
        def close(self): self.closed = True

    stop = threading.Event()
    store = _SpendStore()
    # budget 0.001 < one opener's $0.002 -> A's first like exhausts the shared budget.
    svc = OpenerService(_OpenerClient(), CostTracker(PRICING, run_budget_usd=0.001), store,
                        "s", on_exhausted="stop")
    b = _Gated(stop)
    a = _Spender(b.parked)
    wa = Worker("hinge", a, _LikeDecider(), svc, store, "r", _Pacing(), stop, mode="auto")
    wb = Worker("bumble", b, _LikeDecider(), svc, store, "r", _Pacing(), stop, mode="auto")

    wb.start()
    assert b.parked.wait(2)                        # B is in its loop, waiting on the shared stop
    wa.start()
    wa.join(timeout=10); wb.join(timeout=10)

    assert not wa.is_alive() and not wb.is_alive()
    assert stop.is_set()                           # A's exhaustion propagated to the shared event
    assert len(a.likes) == 1                       # A liked exactly once, then the budget blew
    assert b.likes == [] and b.dislikes == 0       # B halted WITHOUT swiping
    assert a.closed and b.closed
    assert store.rows == [("hinge", "like")]       # only A's single decision recorded


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
