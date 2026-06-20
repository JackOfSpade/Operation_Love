"""RateLimiter unit tests + worker auto-loop cap integration (offline)."""
import threading

from operation_love.costing import CostTracker, ModelPricing
from operation_love.drivers.base import DatingAppDriver
from operation_love.limits import RateLimiter
from operation_love.opener.service import OpenerService
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import Decision
from operation_love.worker import Worker

PRICING = {"claude-opus-4-8": ModelPricing(input=5.0, output=25.0)}


def test_per_run_cap():
    rl = RateLimiter(max_per_run=3)
    assert rl.allow(2, 0) and not rl.allow(3, 0)


def test_per_day_cap():
    rl = RateLimiter(max_per_day=10)
    assert rl.allow(5, 9) and not rl.allow(0, 10)


def test_no_limits_allows_everything():
    assert RateLimiter().allow(10_000, 10_000)


def test_describe():
    assert RateLimiter(60, 100).describe() == "60/run, 100/day"
    assert RateLimiter().describe() == "unlimited"


# --- worker integration --------------------------------------------------
class _Driver(DatingAppDriver):
    def __init__(self, n):
        self.n = n; self.i = 0; self.dislikes = 0; self.closed = False
    def open_session(self): pass
    def next_profile(self):
        if self.i >= self.n:
            return None
        self.i += 1; return Profile(photos=[b"x"])
    def out_of_profiles(self): return self.i >= self.n
    def like(self, opener=None): pass
    def dislike(self): self.dislikes += 1
    def close(self): self.closed = True


class _Decider:
    def decide(self, profile):
        return Decision("dislike", 0.1, [0.1], "ranker")


class _Store:
    def __init__(self): self.decisions = 0
    def count_today(self, app): return 0
    def record_decision(self, *a): self.decisions += 1
    def add_label(self, *a, **k): pass


class _Pacing:
    min_delay_s = max_delay_s = 0.0


def test_worker_stops_at_run_cap():
    driver = _Driver(5)
    store = _Store()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    w = Worker("bumble", driver, _Decider(), svc, store, "r", _Pacing(),
               threading.Event(), mode="auto", limiter=RateLimiter(max_per_run=2))
    w._auto_loop()
    assert driver.dislikes == 2   # capped at 2 even though 5 were available
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
