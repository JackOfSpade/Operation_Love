"""OpenerService branch logic: disabled / budget / out-of-credit / stop-vs-continue.

The service is the GLOBAL, budget-aware gate for Claude openers shared by every
worker. It's exercised indirectly elsewhere; this pins its own decision branches
with lightweight fakes (no Anthropic/network).
"""
import operation_love.opener.service as service_mod
from operation_love.opener.service import OpenerService


class _Res:
    model = "claude-x"
    usage = "usage"
    opener = "hey, that hiking photo is great"
    referenced = "hiking"
    referenced_index = 2


class _Client:
    def __init__(self, exc=None):
        self.exc = exc
        self.calls = 0

    def generate(self, profile, style):
        self.calls += 1
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
    assert t.recorded == [("claude-x", "usage")]
    assert s.disabled is False and s.stop_requested is False


def test_budget_reached_before_call_skips_and_stops():
    c, t, st = _Client(), _Tracker([True]), _Store()
    s = OpenerService(c, t, st, "casual", on_exhausted="stop")
    assert s.maybe_opener("r", "bumble", object()) is None
    assert c.calls == 0                                     # never hit the provider
    assert s.disabled is True and s.stop_requested is True


def test_budget_reached_after_call_disables_but_returns_this_opener():
    c, t, st = _Client(), _Tracker([False, True]), _Store()  # ok pre-call, exhausted post-call
    s = OpenerService(c, t, st, "casual", on_exhausted="stop")
    out = s.maybe_opener("r", "bumble", object())
    assert out.text == _Res.opener                          # the in-flight opener still returns
    assert s.disabled is True and s.stop_requested is True  # but no more after this


def test_out_of_credit_disables_without_raising(monkeypatch):
    monkeypatch.setattr(service_mod, "is_out_of_credit", lambda e: True)
    c, t, st = _Client(exc=RuntimeError("HTTP 402")), _Tracker([False]), _Store()
    s = OpenerService(c, t, st, "casual", on_exhausted="continue")
    assert s.maybe_opener("r", "bumble", object()) is None
    assert s.disabled is True
    assert s.stop_requested is False                        # continue -> degrade, don't stop the run


class _NoPricingTracker(_Tracker):
    """record() raises KeyError, like the real CostTracker when the API echoes back
    a model string with no budget.pricing entry."""
    def record(self, model, usage):
        raise KeyError(f"No pricing configured for model {model!r}")


def test_unpriceable_model_disables_but_returns_this_opener():
    c, t, st = _Client(), _NoPricingTracker([False]), _Store()
    s = OpenerService(c, t, st, "casual", on_exhausted="stop")
    out = s.maybe_opener("r", "bumble", object())
    assert out.text == _Res.opener                          # already-spent credits aren't wasted
    assert len(st.spend) == 1 and st.spend[0][-1] == 0.0     # cost recorded as untracked (0.0)
    assert s.disabled is True and s.stop_requested is True   # but no more openers until pricing is fixed


def test_non_credit_error_propagates(monkeypatch):
    monkeypatch.setattr(service_mod, "is_out_of_credit", lambda e: False)
    s = OpenerService(_Client(exc=RuntimeError("boom")), _Tracker([False]), _Store(), "casual")
    try:
        s.maybe_opener("r", "bumble", object())
        raise AssertionError("expected the non-credit error to propagate")
    except RuntimeError as e:
        assert "boom" in str(e)
    assert s.disabled is False                              # a transient error doesn't disable openers


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn() if fn.__code__.co_argcount == 0 else None
            print(f"{'PASS' if fn.__code__.co_argcount == 0 else 'SKIP(monkeypatch)'} {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    sys.exit(1 if failed else 0)
