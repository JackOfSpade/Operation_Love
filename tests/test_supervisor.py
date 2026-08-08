"""supervisor.run() shutdown/flush path — the run's data-integrity backstop.

Drives the real run() with fakes (monkeypatched module-level collaborators) so no
Playwright/emulator/BigQuery/ML is needed. Pins the two-way finally branch: a clean
flush reports 'stopped'; a flush that RAISES must flip phase->'save_failed' + every app
state->'error' AND re-raise, so a run that lost buffered labels never reports success.
"""
import threading
import time

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
    def detect(cls, hinge_adb_path=None):   # mirrors Capabilities.detect's real signature
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


def test_null_per_app_limits_does_not_crash_worker_construction(monkeypatch, tmp_path):
    """config.validate() blesses a bare `apps.<app>.limits:` (YAML null) as an empty
    override -- but pre-fix, supervisor.py's merge `{**cfg.limits, **app_cfg.get("limits",
    {})}` had no `or {}` guard on either side, so `**None` raised TypeError while building
    the worker -- well after the slow ML-warmup phase, past every startup stop-checkpoint.
    validate() passing must not be false confidence that this run() call is safe."""
    cfg_text = _CONFIG.replace("bumble: {}", "bumble:\n    limits:")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)

    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=threading.Event())          # pre-fix: TypeError here, not a clean run

    snap = captured["status"].snapshot()
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


class _CapsMlMissing(_Caps):
    def missing(self, *names):
        return ["arcface", "clip"] if set(names) == {"arcface", "clip"} else []


def test_ml_missing_degrade_message_is_honest_about_no_defer_path(monkeypatch, tmp_path, capsys):
    """VIS-4: caps is never consulted again after this print, and there's no defer path keyed
    on the missing ml extra -- workers just attempt to embed and fail per profile. The message
    must say that, not promise a defer that doesn't exist."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_CONFIG)
    monkeypatch.setattr(sup, "Capabilities", _CapsMlMissing)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)     # message content doesn't need real ML
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)

    sup.run(str(cfg_path), on_status=lambda s: None, stop_event=threading.Event())

    out = capsys.readouterr().out
    assert "Degrade: ml extra not installed" in out
    assert "Workers will defer until it's present" not in out   # the unfulfilled old promise
    assert "no defer path" in out                                # honest about what happens instead


class _SpyWorker:
    """Stand-in for Worker that records construction without doing anything -- lets a test
    prove a worker was (or wasn't) ever launched, without touching drivers/threads."""
    instances: list = []

    def __init__(self, *a, **k):
        _SpyWorker.instances.append(self)

    def start(self):
        pass

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


def test_stop_before_run_aborts_startup_without_launching_workers(monkeypatch, tmp_path):
    """H-10: Stop must be honoured during startup (store load, ranker training, ML warmup,
    ADB preflight), not only after workers exist. Pre-fix, stop_event is never consulted
    before the worker-launch loop, so even a Stop requested before Start still lets
    workers get constructed and started once the (possibly many-second) startup finishes."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_CONFIG)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _SpyWorker.instances = []
    monkeypatch.setattr(sup, "Worker", _SpyWorker)

    stop_event = threading.Event()
    stop_event.set()                        # Stop requested before/at the very start of run()

    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=stop_event)

    assert _SpyWorker.instances == []       # aborted during startup -- no worker ever built
    snap = captured["status"].snapshot()
    assert snap["phase"] == "stopped" and snap["running"] is False
    assert all(a["state"] == "stopped" for a in snap["apps"].values())


class _FastEmbedder:
    """Skips real ArcFace/CLIP loading so warmup() is instant -- this test only needs
    startup to finish fast and deterministically so it can control exactly when Stop
    lands relative to worker launch, not exercise real embedding."""
    def warmup(self):
        pass


class _FastQuality:
    def __init__(self, *a, **k):
        pass

    def warmup(self):
        pass


def test_wedged_worker_is_not_reported_as_unqualified_success(monkeypatch, tmp_path, capsys):
    """WS-008: a worker still alive after its join timeout is WEDGED -- it may write to the
    store DURING or AFTER flush/close, so a clean flush is not an unqualified success. The
    join must still not block quit forever (timeout stays bounded), but the printed summary
    and the status the hub reads must both say the outcome is qualified, not a flat '✅'."""
    class _WedgedDriver(DatingAppDriver):
        def open_session(self):
            time.sleep(0.3)          # ignores stop_event -- simulates a worker stuck mid-capture
        def next_profile(self):
            return None
        def out_of_profiles(self):
            return True
        def like(self, opener=None, item_index=0):
            pass
        def dislike(self):
            pass
        def close(self):
            pass

    monkeypatch.setattr(sup, "_WORKER_JOIN_TIMEOUT_S", 0.01)   # don't actually wait 30s in a test
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)         # keep startup fast/deterministic
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    store = _FakeStore()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_CONFIG)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _WedgedDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)

    stop_event = threading.Event()
    threading.Timer(0.1, stop_event.set).start()   # simulate Stop pressed mid-run, after launch

    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=stop_event)
    snap = captured["status"].snapshot()

    assert store.closed is True                  # a wedged worker must not block quit forever
    assert snap["phase"] != "stopped"             # not the unqualified-success phase
    assert any(a["state"] == "wedged" for a in snap["apps"].values())
    out = capsys.readouterr().out
    assert "✅ all data saved" not in out
