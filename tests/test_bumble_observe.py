"""Bumble observe-mode decision detection — offline, with a fake Playwright page.

No browser: a FakePage scripts what window.__oplove_decision reads back, so we
exercise wait_for_decision()'s polling/return logic (the DOM click listener is
exercised live by tools/bumble_inspect.py).
"""
from operation_love.drivers.bumble import BumbleDriver


class _Cfg:
    apps = {"bumble": {}}


class FakePage:
    def __init__(self, reads, empty=False):
        self.reads = list(reads)
        self.empty = empty
        self.installed = False

    def evaluate(self, script, arg=None):
        if "__oplove_obs" in script:          # the install snippet
            self.installed = True
            # like/pass/superlike selectors passed through (superswipe -> 'like')
            assert arg == [d.selectors["like"], d.selectors["pass"], d.selectors["superlike"]]
            return None
        return self.reads.pop(0) if self.reads else None   # the read snippet

    def query_selector(self, sel):            # used by out_of_profiles()
        return object() if self.empty else None


def _driver(page):
    drv = BumbleDriver(_Cfg())
    drv.page = page
    return drv


d = BumbleDriver(_Cfg())   # module-level for the selector assertion inside FakePage


def test_like_detected():
    drv = _driver(FakePage(reads=["like"]))
    assert drv.wait_for_decision(timeout=5) is True
    assert drv.page.installed


def test_pass_detected():
    drv = _driver(FakePage(reads=["pass"]))
    assert drv.wait_for_decision(timeout=5) is False


def test_none_then_like():
    drv = _driver(FakePage(reads=[None, None, "like"]))
    assert drv.wait_for_decision(timeout=5) is True


def test_deck_empty_returns_none():
    drv = _driver(FakePage(reads=[None], empty=True))
    assert drv.wait_for_decision(timeout=5) is None


def test_timeout_returns_none():
    drv = _driver(FakePage(reads=[]))            # never any decision, deck not empty
    assert drv.wait_for_decision(timeout=0.2) is None


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
