"""Two app workers run concurrently in one process sharing the store (offline)."""
import threading

from operation_love.costing import CostTracker, ModelPricing
from operation_love.drivers.base import DatingAppDriver
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
    def record_decision(self, *a):
        with self._lock:
            self.decisions += 1
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
