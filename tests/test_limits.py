"""RateLimiter unit tests + worker auto-loop cap integration (offline)."""
import threading

from operation_love.costing import CostTracker, ModelPricing
from operation_love.drivers.base import DatingAppDriver
from operation_love.limits import RateLimiter
from operation_love.opener.service import OpenerService
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import Decision
from operation_love.worker import Worker

PRICING = {"gemini-test-model": ModelPricing(input=5.0, output=25.0)}


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


# --- allow_like_ratio: score-aware ceiling (was score-blind; see worker.py's
# _auto_loop call site and this module's docstring for the audit that found the bug
# and the numbers behind the fix) -------------------------------------------------
def test_allow_like_ratio_below_ceiling_always_allows_regardless_of_score():
    rl = RateLimiter(target_like_ratio=0.5)
    assert rl.allow_like_ratio(1, 3) is True                          # 33% < 50%
    # Even a maximally marginal score doesn't matter below the ceiling.
    assert rl.allow_like_ratio(1, 3, score=0.5001, like_threshold=0.5) is True


def test_allow_like_ratio_with_no_score_context_stays_score_blind_at_ceiling():
    """A caller with no per-decision score (score/like_threshold omitted) gets the
    original score-blind ceiling -- every like is vetoed once the rate is at/above
    target, exactly as before this fix -- rather than an exception or a silent
    pass-through."""
    rl = RateLimiter(target_like_ratio=0.5)
    assert rl.allow_like_ratio(1, 1) is False                          # 100% >= 50%, no score given
    assert rl.allow_like_ratio(2, 4) is False                          # 50% >= 50%, no score given


def test_allow_like_ratio_at_ceiling_demotes_marginal_but_admits_strong():
    """The core fix: at/above the ceiling, only a MARGINAL like (within
    _STRONG_LIKE_MARGIN=0.15 of like_threshold) is vetoed; a clearly strong one is
    admitted even though that means the caller's ratio briefly overshoots target.
    Worked numbers at like_threshold=0.5 (mirrors the module docstring's audited
    example): 0.51 is 0.01 over -> marginal -> vetoed. 0.90 is 0.40 over -> strong
    (well past the 0.15 margin) -> admitted. 0.65 is exactly AT the margin -> the
    boundary itself counts as strong (>=), not marginal."""
    rl = RateLimiter(target_like_ratio=0.5)
    assert rl.allow_like_ratio(1, 1, score=0.51, like_threshold=0.5) is False   # marginal -> vetoed
    assert rl.allow_like_ratio(1, 1, score=0.90, like_threshold=0.5) is True    # strong -> admitted
    assert rl.allow_like_ratio(1, 1, score=0.65, like_threshold=0.5) is True    # exactly on the margin


def test_allow_like_ratio_reproduces_the_audited_score_blind_bug_scenario_fixed():
    """Direct regression pin for the audit's own demonstration: scores
    [0.80, 0.99, 0.81, 0.82] at target_like_ratio=0.5, threshold=0.5. Every one of
    these is >= threshold + 0.15, so under the fixed rule NONE of them are ever
    marginal enough to be demoted -- in particular the strongest call (0.99) must
    never be the one sacrificed to the ratio ceiling, which is exactly what the
    score-blind version got wrong (it demoted 0.99 while keeping a weaker 0.80,
    purely because of arrival order)."""
    rl = RateLimiter(target_like_ratio=0.5)
    for liked, acted, score in [(1, 1, 0.99), (1, 2, 0.81), (1, 3, 0.82)]:
        assert rl.allow_like_ratio(liked, acted, score=score, like_threshold=0.5) is True


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
    def record_decision(self, *a, **k): self.decisions += 1
    def record_profile(self, *a, **k): pass
    def add_label(self, *a, **k): pass


class _Pacing:
    swipe_delay_s = 0.0


def test_worker_stops_at_run_cap():
    driver = _Driver(5)
    store = _Store()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    w = Worker("bumble", driver, _Decider(), svc, store, "r", _Pacing(),
               threading.Event(), mode="auto", limiter=RateLimiter(max_per_run=2))
    w._auto_loop()
    assert driver.dislikes == 2   # capped at 2 even though 5 were available
    assert driver.closed
