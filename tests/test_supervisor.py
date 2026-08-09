"""supervisor.run() shutdown/flush path — the run's data-integrity backstop.

Drives the real run() with fakes (monkeypatched module-level collaborators) so no
Playwright/emulator/BigQuery/ML is needed. Pins the two-way finally branch: a clean
flush reports 'stopped'; a flush that RAISES must flip phase->'save_failed' + every app
state->'error' AND re-raise, so a run that lost buffered labels never reports success.
"""
import os
import threading
import time

import pytest

import operation_love.supervisor as sup
from operation_love.drivers.base import DatingAppDriver

# enabled_apps: [hinge] -- hinge is the one platform the registry ships available/calibrated
# by default (platforms.py); "bumble" is now an Android target that starts out UNCALIBRATED,
# so it would be rejected by supervisor.run()'s new check_runnable() guard before a worker
# is ever built. __DATA_DIR__ is substituted with an isolated tmp_path by _run_with()/the
# tests below so the new per-run device-lock file never lands in the real repo's data/ dir.
_CONFIG = """
enabled_apps: [hinge]
mode: observe
apps:
  hinge: {}
paths:
  data_dir: __DATA_DIR__
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
    def detect(cls, android_adb_path=None):   # mirrors Capabilities.detect's real signature
        return cls()
    def banner(self):
        return ""
    def missing(self, *names):
        return []                     # nothing missing -> no degrade/exit branches


def _patch_no_adb(monkeypatch):
    """Hard rule: the phone is not connected in this environment, and the real ADB
    preflight shells out to `adb devices` -- never let a test actually invoke it."""
    monkeypatch.setattr(sup, "_android_adb_preflight", lambda app, cfg: None)


def _write_cfg(tmp_path, text=_CONFIG):
    """Write `text` as config.yaml under tmp_path, with __DATA_DIR__ resolved to an
    isolated tmp_path subfolder -- keeps the new per-run Android device-lock file (under
    paths.data_dir) out of the real repo's data/ directory."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(text.replace("__DATA_DIR__", str(tmp_path / "data")))
    return cfg_path


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
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)   # don't touch process signals
    _patch_no_adb(monkeypatch)
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
    cfg_text = _CONFIG.replace("hinge: {}", "hinge:\n    limits:")
    cfg_path = _write_cfg(tmp_path, cfg_text)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=threading.Event())          # pre-fix: TypeError here, not a clean run

    snap = captured["status"].snapshot()
    assert snap["phase"] == "stopped"
    assert all(a["state"] == "stopped" for a in snap["apps"].values())


def test_flush_failure_reports_save_failed_and_reraises(monkeypatch, tmp_path):
    store = _FakeStore(flush_error=RuntimeError("BigQuery insert rejected"))
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)
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
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _CapsMlMissing)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)     # message content doesn't need real ML
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

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
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)
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
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _WedgedDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

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


# --- registry guard: run() rejects an unrunnable platform selection up front ---------------

def test_run_rejects_uncalibrated_platform_before_touching_anything(monkeypatch, tmp_path):
    """The check_runnable() guard at the top of run() must fire BEFORE any driver is built,
    using cfg_mod.validate()'s message verbatim (this exercises the guard itself, not just
    validate() -- see test_config.py for validate()'s own coverage of the same rule)."""
    from operation_love import platforms

    cfg_text = _CONFIG.replace("enabled_apps: [hinge]", "enabled_apps: [bumble]").replace(
        "apps:\n  hinge: {}", "apps:\n  bumble: {}")
    cfg_path = _write_cfg(tmp_path, cfg_text)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    built = []
    monkeypatch.setattr(sup, "make_store", lambda cfg: built.append("store"))
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: built.append("driver"))
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        sup.run(str(cfg_path), stop_event=threading.Event())

    assert str(exc_info.value) == platforms.unavailable_reason("bumble")
    assert built == []                            # no store, no driver -- rejected up front


def test_run_rejects_two_android_platforms_together(monkeypatch, tmp_path):
    cfg_text = _CONFIG.replace("enabled_apps: [hinge]", "enabled_apps: [hinge, bumble]")
    cfg_path = _write_cfg(tmp_path, cfg_text)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    with pytest.raises(ValueError):
        sup.run(str(cfg_path), stop_event=threading.Event())


# --- device lock: advisory, cross-process flock so two Android runs can't overlap ----------

def test_android_device_lock_blocks_concurrent_acquire_and_releases_cleanly(tmp_path):
    if sup.fcntl is None:
        pytest.skip("flock is POSIX-only")
    path = tmp_path / ".android-test.lock"
    first = sup._AndroidDeviceLock(path)
    first.acquire()
    try:
        assert path.read_text().strip() == str(os.getpid())
        second = sup._AndroidDeviceLock(path)
        with pytest.raises(RuntimeError, match=f"pid {os.getpid()}"):
            second.acquire()
    finally:
        first.release()

    # Released -> a fresh acquire succeeds and re-stamps the pid.
    third = sup._AndroidDeviceLock(path)
    third.acquire()
    try:
        assert path.read_text().strip() == str(os.getpid())
    finally:
        third.release()


def test_android_device_lock_release_is_idempotent_and_safe_before_acquire(tmp_path):
    lock = sup._AndroidDeviceLock(tmp_path / ".android-never-acquired.lock")
    lock.release()             # never acquired -- must be a safe no-op, not an error
    lock.release()             # and idempotent


def test_android_lock_path_is_one_file_however_the_serial_is_spelled(tmp_path):
    """The same phone must map to the same lock file no matter how each app block names it.

    Keying the path on the configured serial STRING was a hole, not precision: with
    apps.hinge.serial set explicitly and apps.bumble.serial left blank (blank = adb's
    "first available device", which with one phone plugged in is that same Pixel), the two
    produced different lock files and therefore no mutual exclusion at all -- precisely the
    case the lock exists to prevent. Nothing kept the two blocks' serials in sync, and a
    blank serial cannot be compared against an explicit one without asking adb.
    """
    class _Explicit:
        data_dir = tmp_path
        apps = {"hinge": {"serial": "33111JEHN04475"}, "bumble": {"serial": ""}}

    hinge_lock = sup._android_lock_path(_Explicit(), "hinge")
    bumble_lock = sup._android_lock_path(_Explicit(), "bumble")
    assert hinge_lock == bumble_lock, "same phone, different lock files -> no exclusion"

    class _Weird:                       # a hostile serial must not escape data_dir either
        data_dir = tmp_path
        apps = {"hinge": {"serial": "abc 123/weird:name"}}
    p = sup._android_lock_path(_Weird(), "hinge")
    assert p == hinge_lock              # still the one shared lock, unaffected by the string
    assert p.parent == tmp_path
    assert "/" not in p.name and ":" not in p.name and " " not in p.name


def test_device_lock_prevents_overlapping_runs_even_within_one_process(monkeypatch, tmp_path):
    """Integration companion to the unit tests above: drives the real supervisor.run() twice
    with overlapping lifetimes. flock is scoped to the OPEN FILE DESCRIPTION, not the
    process, so a contention here also proves the lock would stop two SEPARATE processes
    (e.g. the hub plus a manually launched CLI run) from ever sharing the phone."""
    gate = threading.Event()

    class _BlockingDriver(DatingAppDriver):
        def open_session(self):
            gate.set()            # signal: the first run has started -> device lock is held
            time.sleep(0.5)
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

    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _BlockingDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    if sup.fcntl is None:
        pytest.skip("flock is POSIX-only")

    first_error = {}

    def _first():
        try:
            sup.run(str(cfg_path), stop_event=threading.Event())
        except Exception as exc:  # noqa: BLE001
            first_error["error"] = exc

    t = threading.Thread(target=_first)
    t.start()
    assert gate.wait(timeout=5)                    # first run now holds the device lock

    with pytest.raises(RuntimeError, match="already in use"):
        sup.run(str(cfg_path), stop_event=threading.Event())

    t.join(timeout=5)
    assert not t.is_alive()
    assert "error" not in first_error               # the first run completed cleanly
