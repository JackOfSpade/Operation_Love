"""supervisor.run() shutdown/flush path — the run's data-integrity backstop.

Drives the real run() with fakes (monkeypatched module-level collaborators) so no
Playwright/emulator/BigQuery/ML is needed. Pins the two-way finally branch: a clean
flush reports 'stopped'; a flush that RAISES must flip phase->'save_failed' + every app
state->'error' AND re-raise, so a run that lost buffered labels never reports success.
"""
import threading

import pytest

import operation_love.supervisor as sup
from operation_love.drivers.base import DatingAppDriver

_CONFIG = """
enabled_apps: [bumble]
mode: observe
apps:
  bumble: {}
storage:
  backend: sqlite
opener:
  enabled: false
budget:
  run_budget_usd: 5.0
  on_exhausted: stop
  pricing: {}
ranker:
  min_labels_to_engage: 40
"""


class _Caps:
    @classmethod
    def detect(cls):
        return cls()
    def banner(self):
        return ""
    def missing(self, *names):
        return []                     # nothing missing -> no degrade/exit branches


class _FakeDriver(DatingAppDriver):
    def open_session(self):
        pass
    def next_profile(self):
        return None
    def out_of_profiles(self):
        return True                   # observe loop breaks immediately -> worker thread exits fast
    def like(self, opener=None, item_index=0):
        pass
    def dislike(self):
        pass
    def close(self):
        pass


class _FakeStore:
    def __init__(self, flush_error=None):
        self.flush_error = flush_error
        self.closed = False
    def load_labels(self):
        return []
    def flush(self):
        if self.flush_error:
            raise self.flush_error    # raises here = the save-failure branch under test
    def close(self):
        self.closed = True
    def saved_summary(self):
        return ""


def _run_with(monkeypatch, tmp_path, store):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_CONFIG)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)   # don't touch process signals
    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=threading.Event())
    return captured["status"].snapshot()


def test_clean_shutdown_reports_stopped(monkeypatch, tmp_path):
    snap = _run_with(monkeypatch, tmp_path, _FakeStore())
    assert snap["phase"] == "stopped"
    assert all(a["state"] == "stopped" for a in snap["apps"].values())


def test_flush_failure_reports_save_failed_and_reraises(monkeypatch, tmp_path):
    store = _FakeStore(flush_error=RuntimeError("BigQuery insert rejected"))
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_CONFIG)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    captured = {}
    with pytest.raises(RuntimeError, match="BigQuery insert rejected"):
        sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
                stop_event=threading.Event())
    snap = captured["status"].snapshot()
    assert snap["phase"] == "save_failed"                       # NOT a green "stopped"
    assert all(a["state"] == "error" for a in snap["apps"].values())


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
