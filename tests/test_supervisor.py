"""supervisor.run() shutdown/flush path — the run's data-integrity backstop.

Drives the real run() with fakes (monkeypatched module-level collaborators) so no
Playwright/emulator/BigQuery/ML is needed. Pins the two-way finally branch: a clean
flush reports 'stopped'; a flush that RAISES must flip phase->'save_failed' + every app
state->'error' AND re-raise, so a run that lost buffered labels never reports success.
"""
import errno
import math
import os
import subprocess
import stat
import sys
import threading
import time

import pytest

import operation_love.supervisor as sup
from operation_love.config import OpenerCfg
from operation_love.drivers.base import DatingAppDriver
from operation_love.private_files import UnsafePrivatePathError

# Liveness bound, not a performance bound: it exists only so a genuine hang fails a test
# instead of hanging the whole suite forever. Widened 2026-08-22 when `python -m pytest` moved
# to one worker per core (pyproject.toml addopts `-n auto --dist loadgroup`). This file's own
# device-lock/reaper tests coordinate real threads through supervisor.run(), and
# test_wedged_android_worker_retains_device_lock_until_it_really_exits was OBSERVED failing
# under full-suite load at its old 5s bound the same day tests/test_concurrency.py measured a
# ~15x slowdown (0.33s idle vs 5.06s loaded) on a positive liveness wait of the same shape --
# so 5s, not just the <=3s gates elsewhere, needed widening here. Nothing about the property
# under test (does the reaper/worker/lock eventually reach the expected state?) depends on the
# exact number, so widening it loses nothing.
_LIVENESS_TIMEOUT_S = 15.0


@pytest.fixture(autouse=True)
def _isolated_android_lock_root(monkeypatch, tmp_path):
    """No supervisor test may touch the real user's process-wide lock directory."""
    monkeypatch.setattr(sup, "_ANDROID_LOCK_ROOT", tmp_path / "operation-love-locks")

# enabled_apps: [hinge] -- hinge is the one platform the registry ships available/calibrated
# by default (platforms.py); "bumble" is now an Android target that starts out UNCALIBRATED,
# so it would be rejected by supervisor.run()'s new check_runnable() guard before a worker
# is ever built. __DATA_DIR__ is substituted with an isolated tmp_path by _run_with()/the
# tests below so stores and debug output never land in the real repo's data/ dir.  The Android
# lock intentionally no longer uses this path; the autouse fixture above isolates its stable
# per-user root separately.
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


def _patch_hinge_auto_ready(monkeypatch):
    """Bypass the shipped targeting-policy gate in tests of downstream AUTO mechanics."""
    original = sup.platforms.unavailable_reason

    def reason(app, mode=None):
        if app == "hinge" and mode == "auto":
            return None
        return original(app, mode)

    monkeypatch.setattr(sup.platforms, "unavailable_reason", reason)


def _write_cfg(tmp_path, text=_CONFIG):
    """Write `text` as config.yaml under tmp_path, with __DATA_DIR__ resolved to an
    isolated tmp_path subfolder so store/debug output stays out of the real repo."""
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
    def like(self, opener=None, item_index=None, *, model_item_index=None):
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
    def count_today(self, app):
        return 0                      # auto mode's rate limiter reads this before anything else
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
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
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
    assert all(a["state"] == "out_of_profiles" for a in snap["apps"].values())


def test_worker_start_failure_preserves_cause_and_still_saves_and_closes(
        monkeypatch, tmp_path, capsys):
    """An unstarted Thread must never enter the shutdown join list.

    ``Thread.join()`` raises for a thread whose ``start()`` failed.  Before this guard that
    secondary RuntimeError replaced the actual launch failure and escaped before persistence
    cleanup.  Driver cleanup is best-effort too: even its own failure must not replace the
    thread-start error.
    """
    class _TrackingStore(_FakeStore):
        def __init__(self):
            super().__init__()
            self.flushed = False

        def flush(self):
            self.flushed = True

    class _CloseFailsDriver(_FakeDriver):
        def __init__(self):
            self.close_attempted = False

        def close(self):
            self.close_attempted = True
            raise OSError("driver cleanup also failed")

    class _StartFailsWorker:
        def __init__(self, *args, **kwargs):
            self.app = args[0]

        def start(self):
            raise RuntimeError("thread creation refused")

        def join(self, timeout=None):
            raise AssertionError("an unstarted worker must never be joined")

        def is_alive(self):
            raise AssertionError("an unstarted worker must never enter liveness checks")

    class _Bridge:
        def __init__(self):
            self.unregistered = None

        def unregister(self, worker):
            self.unregistered = worker

    store = _TrackingStore()
    driver = _CloseFailsDriver()
    bridge = _Bridge()
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: driver)
    monkeypatch.setattr(sup, "Worker", _StartFailsWorker)
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    def _bind(worker):
        worker.observe_action_bridge = bridge

    with pytest.raises(RuntimeError, match="thread creation refused"):
        sup.run(str(cfg_path), stop_event=threading.Event(), on_worker=_bind)

    assert bridge.unregistered is not None
    assert driver.close_attempted is True
    assert store.flushed is True
    assert store.closed is True
    assert "driver cleanup also failed" in capsys.readouterr().out


def test_null_per_app_limits_does_not_crash_worker_construction(monkeypatch, tmp_path):
    """config.validate() blesses a bare `apps.<app>.limits:` (YAML null) as an empty
    override -- but pre-fix, supervisor.py's merge `{**cfg.limits, **app_cfg.get("limits",
    {})}` had no `or {}` guard on either side, so `**None` raised TypeError while building
    the worker -- well after the slow ML-warmup phase, past every startup stop-checkpoint.
    validate() passing must not be false confidence that this run() call is safe."""
    cfg_text = _CONFIG.replace("hinge: {}", "hinge:\n    limits:")
    cfg_path = _write_cfg(tmp_path, cfg_text)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=threading.Event())          # pre-fix: TypeError here, not a clean run

    snap = captured["status"].snapshot()
    assert snap["phase"] == "stopped"
    assert all(a["state"] == "out_of_profiles" for a in snap["apps"].values())


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
    # A flush() failure must not skip close(): pre-fix, close() sat inside the same try as
    # flush(), so a raising flush() left store.close() unreached and (for SQLiteStore) its
    # sqlite3.Connection open.
    assert store.closed is True


class _CloseAlsoFailsStore(_FakeStore):
    """A store whose close() ALSO raises, on top of flush() -- proves close()'s own
    failure is reported but never allowed to replace the flush() error that's actually
    reraised (flush() is what determines whether buffered data made it out)."""
    def __init__(self, flush_error, close_error):
        super().__init__(flush_error=flush_error)
        self.close_error = close_error

    def close(self):
        super().close()          # still marks .closed -- close() was at least ATTEMPTED
        raise self.close_error


def test_close_failure_after_flush_failure_does_not_mask_the_flush_error(monkeypatch, tmp_path, capsys):
    store = _CloseAlsoFailsStore(
        flush_error=RuntimeError("BigQuery insert rejected"),
        close_error=OSError("connection already gone"),
    )
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    with pytest.raises(RuntimeError, match="BigQuery insert rejected"):   # NOT the OSError
        sup.run(str(cfg_path), stop_event=threading.Event())

    assert store.closed is True                    # close() was attempted despite flush() failing
    assert "connection already gone" in capsys.readouterr().out   # reported, not silently dropped


def test_worker_error_state_survives_shutdown_not_overwritten_to_stopped(monkeypatch, tmp_path):
    """Regression test for the bug this change fixes: worker.py's HALT-on-unexpected path
    (run()'s `except Exception:` branch, auto mode or halt_on_error) publishes
    state='error' on the app BEFORE it sets stop_event and returns. Pre-fix, run()'s
    shutdown `finally` unconditionally stamped every enabled app's state to 'saving' and
    then (with no wedged worker and no flush error) to a flat 'stopped' — silently erasing
    the fact this run halted on an unexpected error and making it indistinguishable from a
    clean end-of-queue stop. The app's FINAL published state after run() returns must still
    say 'error'."""
    class _ErrorDriver(DatingAppDriver):
        def open_session(self):
            raise RuntimeError("boom: simulated UnlocatedControlError-style halt")
        def next_profile(self):
            return None
        def out_of_profiles(self):
            return True
        def like(self, opener=None, item_index=None, *, model_item_index=None):
            pass
        def dislike(self):
            pass
        def close(self):
            pass

    cfg_text = _CONFIG.replace("mode: observe", "mode: auto")
    cfg_path = _write_cfg(tmp_path, cfg_text)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _ErrorDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    # This test deliberately drives the post-validation AUTO worker error path.  The Hinge
    # release gate has its own artifact/config coverage; bypass only that prerequisite here so
    # the synthetic driver can reach the shutdown-state behavior this fixture exists to test.
    monkeypatch.setattr(sup.cfg_mod, "_validate_hinge_auto_release_evidence", lambda cfg: None)
    _patch_hinge_auto_ready(monkeypatch)
    _patch_no_adb(monkeypatch)

    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=threading.Event())

    snap = captured["status"].snapshot()
    assert snap["apps"]["hinge"]["state"] == "error"     # NOT overwritten to "stopped"
    assert snap["apps"]["hinge"]["error"]                # the worker's own message survived too
    assert snap["apps"]["hinge"]["mode"] == "auto"       # even though open_session failed
    assert snap["phase"] == "stopped"                    # the save itself still succeeded


@pytest.mark.parametrize("terminal_state", ["out_of_profiles", "rate_limited"])
def test_normal_terminal_reason_survives_successful_save(monkeypatch, tmp_path, terminal_state):
    class _TerminalWorker:
        def __init__(self, app, *args, status=None, mode="observe", **kwargs):
            self.app = app
            self.status = status
            self.mode = mode
        def start(self):
            self.status.set_app(self.app, mode=self.mode, state=terminal_state)
        def join(self, timeout=None):
            pass
        def is_alive(self):
            return False

    cfg_path = _write_cfg(tmp_path, _CONFIG.replace("mode: observe", "mode: auto"))
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "Worker", _TerminalWorker)
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    # This is a terminal-state preservation test, not a release-evidence integration test.
    monkeypatch.setattr(sup.cfg_mod, "_validate_hinge_auto_release_evidence", lambda cfg: None)
    _patch_hinge_auto_ready(monkeypatch)
    _patch_no_adb(monkeypatch)
    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=threading.Event())
    assert captured["status"].snapshot()["apps"]["hinge"]["state"] == terminal_state


def test_stop_reason_survives_both_shutdown_restamps(monkeypatch, tmp_path):
    """Regression pin (adversarial audit item F). worker.py's OpenerService-triggered stop
    path (_finish_session(state="stopped", stop_reason=...)) publishes a human-readable
    cause -- e.g. "run budget reached" -- alongside state="stopped" BEFORE run()'s shutdown
    `finally` block ever touches the status. That finally block then re-stamps every app's
    state TWICE: first to "saving" (line ~413), then to its final terminal state computed
    from `terminal_states` captured just before that first stamp (line ~430-440). This
    project has a documented history of a bug where that kind of re-stamping overwrote
    per-app terminal state wholesale and erased the reason, rendering an opener-exhaustion
    stop indistinguishable from an operator's plain Stop click.

    The current code is correct: RunStatus.set_app (status.py) does `setattr(s, k, v)` per
    field actually passed in **fields, and neither shutdown re-stamp ever passes
    stop_reason -- so the field is simply never touched and survives by construction. This
    test drives the REAL supervisor shutdown sequence against a REAL RunStatus (not a mock
    of set_app) so a future "simplification" of set_app (e.g. replacing the AppStatus
    instance wholesale, or resetting unspecified fields to their dataclass defaults) fails
    this test immediately instead of silently reintroducing the bug.
    """
    class _OpenerStopWorker:
        """Stand-in for a real Worker whose opener service exhausted mid-run: publishes
        exactly what worker.py's _finish_session(state="stopped", stop_reason=...) does,
        then exits -- before run()'s shutdown finally block ever touches the status."""
        def __init__(self, app, *args, status=None, mode="observe", **kwargs):
            self.app = app
            self.status = status
            self.mode = mode
        def start(self):
            self.status.set_app(self.app, mode=self.mode, state="stopped",
                                stop_reason="run budget reached")
        def join(self, timeout=None):
            pass
        def is_alive(self):
            return False

    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "Worker", _OpenerStopWorker)
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=threading.Event())

    snap = captured["status"].snapshot()
    app = snap["apps"]["hinge"]
    assert app["state"] == "stopped"
    assert app["stop_reason"] == "run budget reached"   # survived BOTH shutdown re-stamps
    assert snap["phase"] == "stopped"                    # the save itself still succeeded


def test_plain_stop_mid_run_reports_stopped_not_a_stale_non_terminal_state(monkeypatch, tmp_path):
    """The other side of the rule the two tests above pin. A run the operator simply STOPS
    mid-swipe has NO distinct terminal reason (no error, no empty queue, no rate limit), so
    its durable per-app result must be a plain 'stopped'. Since shutdown now restores each
    app's pre-'saving' state, the risk is the mirror image of the bug those tests cover:
    a NON-terminal state left behind mid-run ('acting'/'scoring'/'capturing'/'saving') must
    not be resurrected as the run's final answer — the hub would then show a live-looking
    state for a run that has already ended."""
    class _StoppedMidRunWorker:
        """Publishes the mid-swipe state a real Worker publishes while acting, then exits
        as soon as Stop lands — without publishing any terminal reason of its own."""
        def __init__(self, app, driver, decider, openers, store, run_id, pacing, stop_event,
                     *args, status=None, mode="observe", **kwargs):
            self.app = app
            self.status = status
            self.mode = mode
            self.stop_event = stop_event
        def start(self):
            self.status.set_app(self.app, mode=self.mode, state="acting")
            # Stop pressed mid-run: armed only now, so it can't land on one of run()'s
            # startup stop-checkpoints and abort before a worker was ever launched.
            threading.Timer(0.05, self.stop_event.set).start()
        def join(self, timeout=None):
            pass
        def is_alive(self):
            return not self.stop_event.is_set()    # responsive: stops cleanly, never wedged

    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "Worker", _StoppedMidRunWorker)
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=threading.Event())

    snap = captured["status"].snapshot()
    assert snap["phase"] == "stopped" and snap["running"] is False
    app = snap["apps"]["hinge"]
    assert app["state"] == "stopped"          # not the stale 'acting', not 'saving'
    assert app["error"] is None               # a plain stop is not an error outcome


class _CapsMlMissing(_Caps):
    def missing(self, *names):
        return ["arcface", "clip"] if set(names) == {"arcface", "clip"} else []


def test_missing_ml_extra_hard_gates_with_no_defer_path_message(monkeypatch, tmp_path):
    """Sibling of the storage.backend=bigquery gate just above it in supervisor.py: both
    are equally, deterministically fatal -- there is no defer path, every profile embed
    would fail immediately -- so both must hard-gate the run instead of just one of them.

    Old contract (VIS-4): this only printed a 'Degrade: ml extra not installed' warning
    and let the run continue past make_store(), the ML warmup, the device lock, and into
    a live driver.open_session() before dying on the first real embed -- see
    test_missing_ml_extra_aborts_before_store_and_device_lock below for what that let
    through. The message content this test used to check for ('no defer path', not
    promising a fulfilled 'Workers will defer until it's present') is preserved, just now
    inside the SystemExit that actually stops the run rather than a print that let it
    continue."""
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _CapsMlMissing)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        sup.run(str(cfg_path), on_status=lambda s: None, stop_event=threading.Event())

    msg = str(excinfo.value)
    assert "ML extra not installed" in msg
    assert "no defer path" in msg


def test_missing_ml_extra_aborts_before_store_and_device_lock(monkeypatch, tmp_path):
    """Mirrors test_gemini_missing_key_fails_before_store_startup: a missing ML extra must
    abort BEFORE make_store()'s BigQuery/label-load work, and before the Android device
    lock is ever constructed -- not just eventually, or after a worker is already running.
    Pre-fix, the missing-extra case was only a warning, so a real run would sail through
    make_store(), the ML warmup, the device lock, and driver.open_session(), capturing a
    REAL profile off the live phone before dying on the very first embed."""
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _CapsMlMissing)
    touched = []
    monkeypatch.setattr(sup, "make_store", lambda cfg: touched.append(True))

    lock_instances: list = []

    class _SpyDeviceLock:
        def __init__(self, path):
            lock_instances.append(self)

        def acquire(self):
            raise AssertionError("device lock must never be acquired")

        def release(self):
            pass

    monkeypatch.setattr(sup, "_AndroidDeviceLock", _SpyDeviceLock)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    with pytest.raises(SystemExit, match="ML extra not installed"):
        sup.run(str(cfg_path), stop_event=threading.Event())

    assert touched == []             # make_store() never reached
    assert lock_instances == []      # device lock never even constructed


def test_absent_adb_device_fails_early_and_actionably(monkeypatch, tmp_path):
    """The cheap 'is the phone connected' check must be a hard, actionable gate now, not a
    warning -- and it must run BEFORE make_store()'s BigQuery/label-load work, not after it
    and the ML warmup (that was the pre-fix ordering: a disconnected phone used to cost a
    full BigQuery + ML warmup cycle before the operator found out). Simulates `adb devices`
    reporting no connected device -- deliberately does NOT use _patch_no_adb, since this
    test needs the REAL _android_adb_preflight to run against a stubbed subprocess.run."""
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    touched = []
    monkeypatch.setattr(sup, "make_store", lambda cfg: touched.append(True))
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)

    class _NoDevices:
        stdout = "List of devices attached\n\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _NoDevices())

    with pytest.raises(SystemExit, match="no connected device"):
        sup.run(str(cfg_path), stop_event=threading.Event())

    assert touched == []                # make_store() never reached


def test_adb_binary_missing_fails_early_with_install_guidance(monkeypatch, tmp_path):
    """Distinct from 'binary works, no device' above: when `adb` itself can't even be
    invoked, there's no way to ask whether a phone is connected, so the actionable next
    step is different (install adb / set apps.<app>.adb_path, not plug in the phone) --
    but it must be equally fatal, and equally early (before make_store())."""
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    touched = []
    monkeypatch.setattr(sup, "make_store", lambda cfg: touched.append(True))
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)

    def _missing_binary(*a, **k):
        raise FileNotFoundError("adb")

    monkeypatch.setattr(subprocess, "run", _missing_binary)

    with pytest.raises(SystemExit, match="was not found"):
        sup.run(str(cfg_path), stop_event=threading.Event())

    assert touched == []


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


def test_startup_load_labels_failure_closes_store_before_propagating(monkeypatch, tmp_path):
    class BrokenLoadStore(_FakeStore):
        def load_labels(self):
            raise RuntimeError("persisted labels unreadable")

    store = BrokenLoadStore()
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda _cfg: store)
    monkeypatch.setattr(
        sup, "make_driver", lambda *_args: pytest.fail("driver must not be constructed"))
    _patch_no_adb(monkeypatch)

    with pytest.raises(RuntimeError, match="persisted labels unreadable"):
        sup.run(str(cfg_path), stop_event=threading.Event())

    assert store.closed is True


@pytest.mark.parametrize("value", [-1, math.nan, math.inf, True, "1.0", 10 ** 1_000])
def test_invalid_persisted_daily_spend_closes_store_and_fails_startup(
        monkeypatch, tmp_path, value):
    class LedgerStore(_FakeStore):
        def spend_today(self):
            return value

    store = LedgerStore()
    cfg_text = _CONFIG.replace(
        "run_budget_usd: 5.0", "run_budget_usd: 5.0\n  day_budget_usd: 10.0")
    cfg_path = _write_cfg(tmp_path, cfg_text)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda _cfg: store)
    monkeypatch.setattr(
        sup, "make_driver", lambda *_args: pytest.fail("driver must not be constructed"))
    _patch_no_adb(monkeypatch)

    with pytest.raises(ValueError, match="spend_today"):
        sup.run(str(cfg_path), stop_event=threading.Event())

    assert store.closed is True


def test_startup_warmup_failure_closes_store_without_masking_cause(monkeypatch, tmp_path):
    class BrokenEmbedder:
        def warmup(self):
            raise RuntimeError("embedder warmup failed")

    store = _FakeStore()
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda _cfg: store)
    monkeypatch.setattr(sup, "Embedder", BrokenEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(
        sup, "make_driver", lambda *_args: pytest.fail("driver must not be constructed"))
    _patch_no_adb(monkeypatch)

    with pytest.raises(RuntimeError, match="embedder warmup failed"):
        sup.run(str(cfg_path), stop_event=threading.Event())

    assert store.closed is True


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
        def like(self, opener=None, item_index=None, *, model_item_index=None):
            pass
        def dislike(self):
            pass
        def close(self):
            pass

    # don't actually wait ~30-105s in a test -- see _worker_join_timeout_s's own tests below for
    # the arithmetic this stands in for.
    monkeypatch.setattr(sup, "_worker_join_timeout_s", lambda cfg: 0.01)
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
    assert "provider_calls=" in out
    assert "provider_spend=$" in out
    assert "openers=" not in out


# --- registry guard: run() rejects an unrunnable platform selection up front ---------------

@pytest.mark.parametrize("mode", ["observe", "auto"])
def test_run_rejects_uncalibrated_bumble_before_touching_anything(
        monkeypatch, tmp_path, mode):
    """The check_runnable() guard at the top of run() must fire BEFORE any driver is built,
    using cfg_mod.validate()'s message verbatim (this exercises the guard itself, not just
    validate() -- see test_config.py for validate()'s own coverage of the same rule)."""
    from operation_love import platforms

    cfg_text = _CONFIG.replace("enabled_apps: [hinge]", "enabled_apps: [bumble]").replace(
        "mode: observe", f"mode: {mode}").replace(
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

    assert str(exc_info.value) == platforms.unavailable_reason("bumble", mode)
    assert built == []                            # no store, no driver -- rejected up front


def test_run_rejects_two_android_platforms_together(monkeypatch, tmp_path):
    cfg_text = _CONFIG.replace("enabled_apps: [hinge]", "enabled_apps: [hinge, bumble]")
    cfg_path = _write_cfg(tmp_path, cfg_text)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    with pytest.raises(ValueError):
        sup.run(str(cfg_path), stop_event=threading.Event())


def test_effective_config_checks_the_per_app_mode_after_global_override(
        monkeypatch, tmp_path):
    """An apps.<app>.mode override wins over the global run-mode override at the shared
    Hub/direct-supervisor gate, just as it does when constructing the Worker."""
    from operation_love import platforms

    platforms.all_platforms()  # materialize spec-derived modes before narrowing the fixture
    monkeypatch.setitem(platforms._AVAILABLE_MODES, "hinge", frozenset({"observe"}))
    cfg_text = _CONFIG.replace("hinge: {}", "hinge:\n    mode: auto")
    cfg_path = _write_cfg(tmp_path, cfg_text)

    with pytest.raises(ValueError, match="Hinge Auto is blocked"):
        sup.load_effective_config(str(cfg_path), mode="observe", enabled_apps=["hinge"])


def test_effective_config_rejects_explicit_falsy_overrides(tmp_path):
    cfg_path = _write_cfg(tmp_path)

    with pytest.raises(ValueError, match="Select a platform to run"):
        sup.load_effective_config(str(cfg_path), enabled_apps=[])
    with pytest.raises(ValueError, match="Unsupported mode ''"):
        sup.load_effective_config(str(cfg_path), mode="")


@pytest.mark.parametrize("value", [[{}], ["hinge", {}], "hinge", 42, {"hinge"}])
def test_effective_config_shape_checks_app_override_before_registry_lookup(tmp_path, value):
    cfg_path = _write_cfg(tmp_path)
    with pytest.raises(ValueError, match="enabled_apps"):
        sup.load_effective_config(str(cfg_path), enabled_apps=value)


@pytest.mark.parametrize("value", [True, -1, 1.5, "8", sup.MAX_PER_RUN_OVERRIDE + 1])
def test_run_rejects_invalid_max_per_run_before_loading_config(monkeypatch, value):
    loaded = []
    monkeypatch.setattr(
        sup, "load_effective_config", lambda *args, **kwargs: loaded.append(True))

    with pytest.raises(ValueError, match="max_per_run"):
        sup.run(max_per_run=value)

    assert loaded == []


@pytest.mark.parametrize("value", [True, -1, 1.5, "8", sup.MAX_PER_RUN_OVERRIDE + 1])
def test_resolve_run_cap_repeats_direct_caller_validation(value):
    with pytest.raises(ValueError, match="max_per_run"):
        sup._resolve_run_cap(12, value)


@pytest.mark.parametrize(
    ("override", "expected"), [(None, 12), (0, None), (7, 7)])
def test_resolve_run_cap_preserves_config_unlimited_and_positive_semantics(
        override, expected):
    assert sup._resolve_run_cap(12, override) == expected


def _gemini_cfg_text(extra_opener_yaml="", thinking_yaml="    gemini-test: {}\n"):
    """Build _CONFIG with a valid single-model Gemini opener block (including the
    opener.thinking entry every configured model now requires -- see config.py's
    _validate_gemini_thinking). `extra_opener_yaml` is appended under the `opener:` key,
    indented to match, for tests that need to add e.g. `preflight: false`. `thinking_yaml`
    overrides the (4-space-indented) body of the `thinking:` mapping."""
    opener_yaml = (
        "enabled: true\n  provider: gemini\n  model: gemini-test\n  models: [gemini-test]\n"
        "  thinking:\n" + thinking_yaml
        + extra_opener_yaml
    )
    return _CONFIG.replace("enabled: false", opener_yaml).replace(
        "pricing: {}", "pricing:\n    gemini-test: {input: 0, output: 0}")


class _SpyGeminiOpener:
    """Stand-in for GeminiOpener that records construction args and whether/how preflight()
    was invoked, without ever touching the network -- used by every Gemini-path supervisor
    test below so none of them can accidentally make a real HTTP call.

    `next_preflight_error` is a CLASS attribute a test sets before calling sup.run() to make
    the next-constructed instance's preflight() raise; it's read (then left alone -- reset
    explicitly per test, same pattern as `instances`) rather than passed through __init__
    because supervisor.py constructs this with GeminiOpener's real positional/keyword
    signature, which a test must not need to change just to inject a failure.
    """
    instances: list = []
    next_preflight_error: Exception | None = None

    def __init__(self, models, max_tokens, request_timeout_s, *, api_key=None, env=None,
                 transport=None, thinking=None):
        self.models = models
        self.max_tokens = max_tokens
        self.request_timeout_s = request_timeout_s
        self.api_key = api_key
        self.thinking = thinking
        self.preflight_called = False
        _SpyGeminiOpener.instances.append(self)

    def preflight(self):
        self.preflight_called = True
        if _SpyGeminiOpener.next_preflight_error is not None:
            raise _SpyGeminiOpener.next_preflight_error

    def generate(self, profile, style):
        raise AssertionError("generate() should never be reached in these startup-only tests")


def test_gemini_missing_key_fails_before_store_startup(monkeypatch, tmp_path):
    cfg_path = _write_cfg(tmp_path, _gemini_cfg_text())
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    touched = []
    monkeypatch.setattr(sup, "make_store", lambda cfg: touched.append(True))

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        sup.run(str(cfg_path), stop_event=threading.Event())

    assert touched == []


def test_gemini_preflight_failure_aborts_before_store_startup(monkeypatch, tmp_path):
    """Mirrors test_gemini_missing_key_fails_before_store_startup: a preflight() failure
    (typo'd model id, invalid key rejected by ListModels, ...) must abort BEFORE
    make_store()'s slow BigQuery/embedder warmup runs, exactly like the missing-key case
    above -- and its message must survive unchanged."""
    cfg_path = _write_cfg(tmp_path, _gemini_cfg_text())
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _SpyGeminiOpener.instances = []
    _SpyGeminiOpener.next_preflight_error = RuntimeError(
        "Gemini preflight failed -- missing: gemini-test")
    monkeypatch.setattr(sup, "GeminiOpener", _SpyGeminiOpener)
    touched = []
    monkeypatch.setattr(sup, "make_store", lambda cfg: touched.append(True))

    try:
        with pytest.raises(RuntimeError, match="Gemini preflight failed"):
            sup.run(str(cfg_path), stop_event=threading.Event())
    finally:
        _SpyGeminiOpener.next_preflight_error = None   # don't leak into later tests

    assert touched == []                          # make_store() never reached
    assert len(_SpyGeminiOpener.instances) == 1
    assert _SpyGeminiOpener.instances[0].preflight_called is True


def test_gemini_preflight_false_skips_the_network_call(monkeypatch, tmp_path, capsys):
    """opener.preflight: false must skip the ListModels call entirely -- offline dev and
    the rest of this test suite rely on this to never touch the network -- and print a
    line saying model ids are unvalidated."""
    cfg_path = _write_cfg(tmp_path, _gemini_cfg_text("  preflight: false\n"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _SpyGeminiOpener.instances = []
    monkeypatch.setattr(sup, "GeminiOpener", _SpyGeminiOpener)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    sup.run(str(cfg_path), stop_event=threading.Event())

    assert len(_SpyGeminiOpener.instances) == 1
    assert _SpyGeminiOpener.instances[0].preflight_called is False
    assert "preflight is false" in capsys.readouterr().out


def test_gemini_thinking_config_reaches_the_constructed_opener(monkeypatch, tmp_path):
    """cfg.opener.thinking must reach GeminiOpener's constructor unchanged -- this is the
    only path that turns config.yaml's per-model thinkingConfig into what actually gets
    sent to Gemini (see GeminiOpener._payload)."""
    cfg_text = _gemini_cfg_text(
        "  preflight: false\n", thinking_yaml="    gemini-test: {thinkingLevel: minimal}\n")
    cfg_path = _write_cfg(tmp_path, cfg_text)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _SpyGeminiOpener.instances = []
    monkeypatch.setattr(sup, "GeminiOpener", _SpyGeminiOpener)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    sup.run(str(cfg_path), stop_event=threading.Event())

    assert len(_SpyGeminiOpener.instances) == 1
    assert _SpyGeminiOpener.instances[0].thinking == {"gemini-test": {"thinkingLevel": "minimal"}}


def _wiring_cfg_text():
    """A Gemini opener config whose every constructor-bound field is a DISTINCT,
    distinguishable value -- deliberately unlike _gemini_cfg_text()'s defaults, where
    `model` and `models` share one value and max_tokens/request_timeout_s are never
    asserted at all. `model` (the legacy singular fallback key) is set to a value that
    never appears in `models`, so a mutation that wires `cfg.opener.model` into the
    constructor instead of `cfg.opener.effective_models` produces a visibly wrong string
    where a two-element list is expected. `max_tokens` and `request_timeout_s` are two
    different numbers so a positional swap between them cannot accidentally still pass."""
    opener_yaml = (
        "enabled: true\n"
        "  provider: gemini\n"
        "  model: gemini-legacy-unused-fallback\n"
        "  models: [gemini-wired-a, gemini-wired-b]\n"
        "  max_tokens: 777\n"
        "  request_timeout_s: 55\n"
        "  thinking:\n"
        "    gemini-wired-a: {}\n"
        "    gemini-wired-b: {}\n"
    )
    return _CONFIG.replace("enabled: false", opener_yaml).replace(
        "pricing: {}",
        "pricing:\n    gemini-wired-a: {input: 0, output: 0}\n"
        "    gemini-wired-b: {input: 0, output: 0}")


def test_gemini_constructor_args_wired_correctly_and_distinguishably(monkeypatch, tmp_path):
    """Pins supervisor.run()'s GeminiOpener(...) construction (~line 205-210) against every
    argument arriving in the wrong slot -- the mutation audit found this call is built from
    POSITIONAL args (models, max_tokens, request_timeout_s) plus keyword api_key/thinking,
    and _SpyGeminiOpener already recorded all four, yet nothing asserted three of them, so a
    swap or a wrong hardcode sailed through fully green.

    Uses _wiring_cfg_text(): max_tokens=777 and request_timeout_s=55 are distinct numbers,
    so swapping the two positional args is caught (max_tokens would read 55, or
    request_timeout_s would read 777). opener.model is a THIRD, never-used value distinct
    from opener.models, so passing the singular `cfg.opener.model` instead of the plural
    `cfg.opener.effective_models` produces a bare string where a two-element list is
    expected, not something that could coincidentally match. api_key comes from a
    monkeypatched GEMINI_API_KEY with a value found nowhere else in this test, so a
    hardcoded wrong key is caught too.

    Concretely, in the shipped config this exact positional swap yields max_tokens=90 (the
    request_timeout_s default that used to live in that slot) -- low enough to truncate
    essentially every real opener into a MAX_TOKENS failure -- and a 2048-second timeout;
    that real-world blast radius is why this is HOLE 1, the most important of the four."""
    cfg_path = _write_cfg(tmp_path, _wiring_cfg_text())
    monkeypatch.setenv("GEMINI_API_KEY", "distinctive-fake-test-key-88221")
    _SpyGeminiOpener.instances = []
    monkeypatch.setattr(sup, "GeminiOpener", _SpyGeminiOpener)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    sup.run(str(cfg_path), stop_event=threading.Event())

    assert len(_SpyGeminiOpener.instances) == 1
    spy = _SpyGeminiOpener.instances[0]
    assert spy.models == ["gemini-wired-a", "gemini-wired-b"]   # effective_models, not .model
    assert spy.max_tokens == 777
    assert spy.request_timeout_s == 55
    assert spy.api_key == "distinctive-fake-test-key-88221"


# --- opener.max_attempts: owner rule, "stop after 5 bad AI responses" ----------------------
# OpenerService is constructed unconditionally in run() (opener.enabled or not -- see
# supervisor.py), so this doesn't need a Gemini-configured opener block at all; the base
# _CONFIG (opener.enabled: false) is enough to exercise the wiring.

class _SpyOpenerService:
    """Stand-in for OpenerService that records constructor args, without ever running a
    worker against it -- used to pin cfg.opener.max_attempts reaching the constructed
    service (contract: OpenerService(client, tracker, store, style, max_attempts=5))."""
    instances: list = []

    def __init__(self, client, tracker, store, style, max_attempts=5,
                 advisory_max_attempts=3, advisory_deadline_s=60.0):
        self.client = client
        self.tracker = tracker
        self.store = store
        self.style = style
        self.max_attempts = max_attempts
        self.advisory_max_attempts = advisory_max_attempts
        self.advisory_deadline_s = advisory_deadline_s
        _SpyOpenerService.instances.append(self)


def test_opener_max_attempts_reaches_the_constructed_opener_service(monkeypatch, tmp_path):
    """cfg.opener.max_attempts must reach OpenerService's constructor as the keyword
    max_attempts -- this is the setting that controls how many times a rejected AI response
    is re-asked before the run stops (owner rule, 2026-08-10: no commentless likes). Uses a
    distinctive value (7) rather than the class default (5) so a mutation that drops the
    kwarg entirely -- silently falling back to OpenerService's own default -- cannot pass
    unnoticed."""
    cfg_text = _CONFIG.replace(
        "opener:\n  enabled: false",
        "opener:\n  enabled: false\n  max_attempts: 7\n"
        "  advisory_max_attempts: 2\n  advisory_deadline_s: 17.5")
    cfg_path = _write_cfg(tmp_path, cfg_text)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)
    _SpyOpenerService.instances = []
    monkeypatch.setattr(sup, "OpenerService", _SpyOpenerService)

    sup.run(str(cfg_path), stop_event=threading.Event())

    assert len(_SpyOpenerService.instances) == 1
    assert _SpyOpenerService.instances[0].max_attempts == 7
    assert _SpyOpenerService.instances[0].advisory_max_attempts == 2
    assert _SpyOpenerService.instances[0].advisory_deadline_s == 17.5


def test_opener_max_attempts_default_reaches_the_constructed_opener_service(monkeypatch, tmp_path):
    """The shipped default (5, from OpenerCfg) must also reach the constructor when
    config.yaml doesn't override it -- not just an explicitly-set value."""
    cfg_path = _write_cfg(tmp_path)   # base _CONFIG sets no opener.max_attempts
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)
    _SpyOpenerService.instances = []
    monkeypatch.setattr(sup, "OpenerService", _SpyOpenerService)

    sup.run(str(cfg_path), stop_event=threading.Event())

    assert len(_SpyOpenerService.instances) == 1
    assert _SpyOpenerService.instances[0].max_attempts == 5
    assert _SpyOpenerService.instances[0].advisory_max_attempts == 3
    assert _SpyOpenerService.instances[0].advisory_deadline_s == 60.0


def test_on_opener_service_callback_receives_the_live_opener_service(monkeypatch, tmp_path):
    """The hub's bug report needs the OpenerService this run actually built: only its own
    ring buffer (OpenerService.recent_openers_snapshot) records what each opener said and
    whether it was anchored to the live like screen -- RunStatus's raw `openers` counter
    can't show either. Mirrors the existing on_status/on_store callback contract: invoked
    once, right after construction, with the SAME instance (not a copy) that gets handed to
    Worker."""
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)
    _SpyOpenerService.instances = []
    monkeypatch.setattr(sup, "OpenerService", _SpyOpenerService)
    captured = {}

    sup.run(str(cfg_path), on_opener_service=lambda svc: captured.__setitem__("svc", svc),
            stop_event=threading.Event())

    assert len(_SpyOpenerService.instances) == 1
    assert captured["svc"] is _SpyOpenerService.instances[0]


# --- device lock: OS-backed cross-process exclusion so two Android runs can't overlap -------

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


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_android_device_lock_tightens_leaf_directory_and_existing_lock_file(tmp_path):
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir(mode=0o755)
    path = lock_dir / "android.lock"
    path.write_text("stale")
    lock_dir.chmod(0o755)
    path.chmod(0o644)

    lock = sup._AndroidDeviceLock(path)
    lock.acquire()
    try:
        assert stat.S_IMODE(lock_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        lock.release()


@pytest.mark.skipif(os.name != "posix", reason="symlink and POSIX permission semantics")
def test_android_device_lock_rejects_symlink_without_mutating_target(tmp_path):
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("valuable content")
    unrelated.chmod(0o644)
    path = lock_dir / "android.lock"
    path.symlink_to(unrelated)

    with pytest.raises(UnsafePrivatePathError):
        sup._AndroidDeviceLock(path).acquire()

    assert unrelated.read_text() == "valuable content"
    assert stat.S_IMODE(unrelated.stat().st_mode) == 0o644


def test_android_lock_path_is_one_file_across_serials_and_data_dirs(tmp_path):
    """The same phone must map to one stable per-user file for every config.

    Keying the path on the configured serial STRING was a hole, not precision: with
    apps.hinge.serial set explicitly and apps.bumble.serial left blank (blank = adb's
    "first available device", which with one phone plugged in is that same Pixel), the two
    produced different lock files and therefore no mutual exclusion at all -- precisely the
    case the lock exists to prevent. Nothing kept the two blocks' serials in sync, and a
    blank serial cannot be compared against an explicit one without asking adb.
    """
    class _Explicit:
        data_dir = tmp_path / "first-data-root"
        apps = {"hinge": {"serial": "33111JEHN04475"}, "bumble": {"serial": ""}}

    class _OtherDataRoot:
        data_dir = tmp_path / "second-data-root"
        apps = {"hinge": {"serial": "33111JEHN04475"}}

    hinge_lock = sup._android_lock_path(_Explicit(), "hinge")
    bumble_lock = sup._android_lock_path(_Explicit(), "bumble")
    other_config_lock = sup._android_lock_path(_OtherDataRoot(), "hinge")
    assert hinge_lock == bumble_lock, "same phone, different lock files -> no exclusion"
    assert hinge_lock == other_config_lock, "different data_dir values bypassed the device lock"

    class _Weird:                       # a hostile serial must not escape data_dir either
        data_dir = tmp_path / "third-data-root"
        apps = {"hinge": {"serial": "abc 123/weird:name"}}
    p = sup._android_lock_path(_Weird(), "hinge")
    assert p == hinge_lock              # still the one shared lock, unaffected by the string
    assert p.parent == sup._ANDROID_LOCK_ROOT
    assert "/" not in p.name and ":" not in p.name and " " not in p.name


def test_android_lock_root_default_is_the_operators_real_home_directory():
    """Pins the PRODUCTION default of ``_ANDROID_LOCK_ROOT`` -- the counterweight to
    tests/conftest.py's session-scoped ``_machine_global_state_is_never_the_operators``
    fixture, which monkeypatches ``sup._ANDROID_LOCK_ROOT`` to a per-worker tmp_path for this
    entire test session so no test here can ever collide with (or acquire) the operator's real
    device lock. That means ``sup._ANDROID_LOCK_ROOT`` -- the live module attribute -- is NOT
    the production value for as long as the suite runs; every other test in this file that
    reads it (e.g. the one above, via ``p.parent == sup._ANDROID_LOCK_ROOT``) is only proving
    internal consistency with whatever the isolation fixture installed, not what a real run
    actually uses.

    This test reads the module's own SOURCE rather than the (deliberately redirected) live
    attribute, and touches no lock file and no real filesystem path: ``_ANDROID_LOCK_ROOT`` is
    supposed to be ``Path.home() / ".operation-love" / "locks"`` -- one fixed, per-user
    directory outside the repo and outside any config's ``data_dir`` -- exactly so that every
    Operation Love process on this machine, launched from any config, agrees on the one lock
    file that keeps two Android runs from ever sharing the phone. If someone changes this
    default, this is the one test that must change with them, deliberately.
    """
    import ast
    import inspect

    source = inspect.getsource(sup)
    tree = ast.parse(source)
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_ANDROID_LOCK_ROOT" for t in node.targets)
    ]
    assert len(assignments) == 1, "expected exactly one module-level _ANDROID_LOCK_ROOT default"

    expected = ast.parse('Path.home() / ".operation-love" / "locks"', mode="eval").body
    assert ast.unparse(assignments[0].value) == ast.unparse(expected)


def test_android_lock_contends_across_processes_and_distinct_data_dirs(tmp_path):
    class _First:
        data_dir = tmp_path / "one"

    class _Second:
        data_dir = tmp_path / "two"

    first_path = sup._android_lock_path(_First(), "hinge")
    second_path = sup._android_lock_path(_Second(), "hinge")
    assert first_path == second_path

    first = sup._AndroidDeviceLock(first_path)
    first.acquire()
    try:
        script = (
            "import sys\n"
            "from pathlib import Path\n"
            "from operation_love.supervisor import _AndroidDeviceLock\n"
            "lock = _AndroidDeviceLock(Path(sys.argv[1]))\n"
            "try:\n"
            "    lock.acquire()\n"
            "except RuntimeError as exc:\n"
            "    print(exc)\n"
            "    raise SystemExit(17)\n"
            "else:\n"
            "    lock.release()\n"
            "    raise SystemExit(2)\n"
        )
        child = subprocess.run(
            [sys.executable, "-c", script, str(second_path)],
            capture_output=True, text=True, timeout=10, check=False,
        )
        assert child.returncode == 17
        assert "already in use" in child.stdout
        assert f"pid {os.getpid()}" in child.stdout
    finally:
        first.release()


def test_windows_lock_backend_locks_and_unlocks_the_same_byte(monkeypatch, tmp_path):
    class _FakeMsvcrt:
        LK_NBLCK = 10
        LK_UNLCK = 11

        def __init__(self):
            self.calls = []

        def locking(self, fd, mode, count):
            self.calls.append((mode, count, os.lseek(fd, 0, os.SEEK_CUR)))

    backend = _FakeMsvcrt()
    monkeypatch.setattr(sup, "fcntl", None)
    monkeypatch.setattr(sup, "msvcrt", backend)
    path = tmp_path / "windows.lock"
    lock = sup._AndroidDeviceLock(path)
    lock.acquire()
    assert path.read_text() == str(os.getpid())
    lock.release()
    assert backend.calls == [
        (backend.LK_NBLCK, 1, 0),
        (backend.LK_UNLCK, 1, 0),
    ]


def test_windows_lock_contention_reports_holder_and_missing_backend_fails_closed(
        monkeypatch, tmp_path):
    class _BusyMsvcrt:
        LK_NBLCK = 10
        LK_UNLCK = 11

        @staticmethod
        def locking(_fd, mode, _count):
            if mode == _BusyMsvcrt.LK_NBLCK:
                raise OSError(errno.EACCES, "locked")

    path = tmp_path / "windows-busy.lock"
    path.write_text("4321")
    monkeypatch.setattr(sup, "fcntl", None)
    monkeypatch.setattr(sup, "msvcrt", _BusyMsvcrt())
    with pytest.raises(RuntimeError, match="pid 4321"):
        sup._AndroidDeviceLock(path).acquire()

    monkeypatch.setattr(sup, "msvcrt", None)
    with pytest.raises(RuntimeError, match="neither fcntl nor msvcrt"):
        sup._AndroidDeviceLock(tmp_path / "unsupported.lock").acquire()


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
        def like(self, opener=None, item_index=None, *, model_item_index=None):
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
    assert gate.wait(timeout=_LIVENESS_TIMEOUT_S)  # first run now holds the device lock

    with pytest.raises(RuntimeError, match="already in use"):
        sup.run(str(cfg_path), stop_event=threading.Event())

    t.join(timeout=_LIVENESS_TIMEOUT_S)
    assert not t.is_alive()
    assert "error" not in first_error               # the first run completed cleanly


def test_wedged_android_worker_retains_device_lock_until_it_really_exits(
        monkeypatch, tmp_path, capsys):
    if sup.fcntl is None:
        pytest.skip("flock is POSIX-only")

    entered = threading.Event()
    release = threading.Event()

    class _WedgedAndroidDriver(DatingAppDriver):
        def open_session(self):
            entered.set()
            release.wait(timeout=_LIVENESS_TIMEOUT_S)

        def next_profile(self):
            return None

        def out_of_profiles(self):
            return True

        def like(self, opener=None, item_index=None, *, model_item_index=None):
            pass

        def dislike(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(sup, "_worker_join_timeout_s", lambda cfg: 0.01)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _WedgedAndroidDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)
    stop = threading.Event()

    def stop_after_worker_enters():
        assert entered.wait(timeout=_LIVENESS_TIMEOUT_S)
        stop.set()

    threading.Thread(target=stop_after_worker_enters, daemon=True).start()
    sup.run(str(_write_cfg(tmp_path)), stop_event=stop)

    path = sup._ANDROID_LOCK_ROOT / "android-device.lock"
    contender = sup._AndroidDeviceLock(path)
    with pytest.raises(RuntimeError, match="already in use"):
        contender.acquire()
    assert "retaining Android device lock" in capsys.readouterr().out

    release.set()
    # This poll is the exact gate that was OBSERVED to flake under full-suite load on
    # 2026-08-22 (see _LIVENESS_TIMEOUT_S above): the reaper thread's `worker.join()` plus the
    # worker actually unwinding through next_profile()/out_of_profiles()/close() has to be
    # scheduled fairly against every other pytest-xdist worker's CPU-bound test, and 5s of
    # wall-clock budget was not always enough margin for that under contention.
    deadline = time.monotonic() + _LIVENESS_TIMEOUT_S
    while True:
        try:
            contender.acquire()
            break
        except RuntimeError as exc:
            if time.monotonic() >= deadline:
                raise AssertionError("reaper did not release lock after worker exit") from exc
            time.sleep(0.01)
    contender.release()
    with sup._RETAINED_DEVICE_LOCKS_GUARD:
        assert not sup._RETAINED_DEVICE_LOCKS


def test_reaper_start_failure_keeps_lock_but_does_not_skip_store_shutdown(
        monkeypatch, tmp_path, capsys):
    if sup.fcntl is None:
        pytest.skip("flock is POSIX-only")

    release = threading.Event()
    stop = threading.Event()
    captured = {}

    class _WedgedAndroidDriver(DatingAppDriver):
        def open_session(self):
            stop.set()
            release.wait(timeout=_LIVENESS_TIMEOUT_S)

        def next_profile(self):
            return None

        def out_of_profiles(self):
            return True

        def like(self, opener=None, item_index=None, *, model_item_index=None):
            pass

        def dislike(self):
            pass

        def close(self):
            pass

    class _RecordingStore(_FakeStore):
        flushed = False

        def flush(self):
            self.flushed = True
            return super().flush()

    class _UnstartableThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread quota exhausted")

    store = _RecordingStore()
    monkeypatch.setattr(sup, "_worker_join_timeout_s", lambda cfg: 0.01)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _WedgedAndroidDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda event: None)
    _patch_no_adb(monkeypatch)

    def capture_worker(worker):
        captured["worker"] = worker
        monkeypatch.setattr(sup.threading, "Thread", _UnstartableThread)

    sup.run(str(_write_cfg(tmp_path)), stop_event=stop, on_worker=capture_worker)

    assert store.flushed is True
    assert store.closed is True
    assert "retaining the device lock for this process's lifetime" in capsys.readouterr().out
    contender = sup._AndroidDeviceLock(sup._ANDROID_LOCK_ROOT / "android-device.lock")
    with pytest.raises(RuntimeError, match="already in use"):
        contender.acquire()

    release.set()
    captured["worker"].join(timeout=_LIVENESS_TIMEOUT_S)
    with sup._RETAINED_DEVICE_LOCKS_GUARD:
        retained = list(sup._RETAINED_DEVICE_LOCKS)
        sup._RETAINED_DEVICE_LOCKS.clear()
    for lock in retained:
        lock.release()


# --- E: _worker_join_timeout_s must derive from opener.request_timeout_s, not a stale flat
# constant -- see the constants' own comments in supervisor.py for the full arithmetic. -------

class _JoinTimeoutCfg:
    """Bare stand-in carrying only the one field _worker_join_timeout_s actually reads --
    real Config/OpenerCfg objects carry a lot more than this function needs."""
    def __init__(self, opener):
        self.opener = opener


def test_join_timeout_exceeds_request_timeout_s_when_openers_enabled():
    """A worker can legitimately still be riding out ONE in-flight opener HTTP request
    (bounded by request_timeout_s -- every should_stop check runs BETWEEN attempts/models,
    never while a request is on the wire) when stop_event is set. The join timeout must
    comfortably exceed that bound, or a perfectly healthy worker gets misreported wedged."""
    cfg = _JoinTimeoutCfg(OpenerCfg(enabled=True, request_timeout_s=90))
    timeout = sup._worker_join_timeout_s(cfg)
    assert timeout > 90
    assert timeout == 90 + sup._WORKER_JOIN_TIMEOUT_MARGIN_S    # exact arithmetic, not just ">"


def test_join_timeout_scales_with_a_nondefault_request_timeout_s():
    """Not hardcoded to the shipped 90s default -- a differently configured
    request_timeout_s must move the join timeout with it."""
    cfg = _JoinTimeoutCfg(OpenerCfg(enabled=True, request_timeout_s=45))
    assert sup._worker_join_timeout_s(cfg) == 45 + sup._WORKER_JOIN_TIMEOUT_MARGIN_S


def test_join_timeout_falls_back_to_the_sane_floor_when_openers_disabled():
    """opener.enabled=False -> no opener HTTP call can ever be in flight, so there's nothing
    analogous to ride out. The floor equals the ORIGINAL flat constant this replaces, so a
    run with openers off stays exactly as responsive to a genuinely wedged worker as before."""
    cfg = _JoinTimeoutCfg(OpenerCfg(enabled=False, request_timeout_s=90))
    assert sup._worker_join_timeout_s(cfg) == sup._WORKER_JOIN_TIMEOUT_FLOOR_S == 30.0


# --- audit fix: an honest "stopping" tail between running and stopped ----------------------
# supervisor.py's shutdown `finally` used to publish phase="saving data" (and nothing for
# `stopping`) the INSTANT shutdown began -- before stop_event.set(), before any worker was
# even asked to notice it -- so the hub showed "saving data…" (and kept its green observe
# GO cue up) for the whole worker-join window, while a worker could still be mid-swipe and
# any decision it recorded there would be silently discarded (worker.py re-checks
# stop_event around every decision point). These tests pin the fixed ordering.

def test_stopping_is_true_and_phase_is_stopping_before_saving_data_begins(monkeypatch, tmp_path):
    """Core ordering fix: while a still-live worker is inside the join wait (not yet
    joined/declared wedged), status must read phase='stopping' with stopping=True -- NOT
    'saving data'. Uses a worker driver that blocks in open_session() until released, so the
    finally block is provably still stuck in its join loop when this samples status."""
    release = threading.Event()

    class _BlockedUntilReleased(DatingAppDriver):
        def open_session(self):
            # This is only a failsafe against the fake driver hanging forever if `release`
            # somehow never fires -- the test always calls release.set() explicitly at ~0.5s
            # (line below), well inside this bound regardless of machine load. It is
            # deliberately independent from the mocked `_worker_join_timeout_s=5.0` below,
            # which IS an input to the production code under test and must stay as authored.
            release.wait(timeout=_LIVENESS_TIMEOUT_S)
        def next_profile(self):
            return None
        def out_of_profiles(self):
            return True
        def like(self, opener=None, item_index=None, *, model_item_index=None):
            pass
        def dislike(self):
            pass
        def close(self):
            pass

    monkeypatch.setattr(sup, "_worker_join_timeout_s", lambda cfg: 5.0)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    store = _FakeStore()
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _BlockedUntilReleased())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    captured = {}
    stop_event = threading.Event()
    threading.Timer(0.15, stop_event.set).start()   # give startup time to reach 'live' first

    run_thread = threading.Thread(
        target=lambda: sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
                               stop_event=stop_event))
    run_thread.start()

    # stop_event fired at ~0.15s; the worker is still blocked in open_session() (release not
    # set yet), so the finally block must still be stuck inside its join loop right now.
    time.sleep(0.35)
    mid = captured["status"].snapshot()
    release.set()                            # let the worker (and thus the join loop) finish
    run_thread.join(timeout=_LIVENESS_TIMEOUT_S)

    assert mid["phase"] == "stopping"
    assert mid["stopping"] is True
    final = captured["status"].snapshot()
    assert final["phase"] == "stopped"
    assert final["stopping"] is False        # cleared once the terminal phase lands


def test_saving_data_phase_and_flush_see_stopping_still_true(monkeypatch, tmp_path):
    """The other end of the same fix: by the time store.flush() actually runs, every worker
    has already been joined/declared wedged (phase has moved on to 'saving data'), but
    `stopping` itself must still read True -- it only clears at the very end, once the
    terminal phase (stopped/wedged/save_failed) is published."""
    captured = {}

    class _RecordingStore(_FakeStore):
        def flush(self):
            captured["phase_at_flush"] = captured["status"].snapshot()["phase"]
            captured["stopping_at_flush"] = captured["status"].snapshot()["stopping"]
            return super().flush()

    store = _RecordingStore()
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=threading.Event())

    assert captured["phase_at_flush"] == "saving data"
    assert captured["stopping_at_flush"] is True
    final = captured["status"].snapshot()
    assert final["phase"] == "stopped"
    assert final["stopping"] is False


def test_stopping_print_names_the_real_join_timeout_and_worker_count(monkeypatch, tmp_path, capsys):
    """The new operator-facing line printed before the join loop must name the ACTUAL
    computed bound (join_timeout_s), not a guess -- and how many workers it's waiting on."""
    monkeypatch.setattr(sup, "_worker_join_timeout_s", lambda cfg: 42.0)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    store = _FakeStore()
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    sup.run(str(cfg_path), stop_event=threading.Event())

    out = capsys.readouterr().out
    assert ("Supervisor: stopping — waiting up to 42s for 1 worker(s) to finish what they "
            "are doing") in out


def test_stopping_print_omitted_when_no_worker_was_ever_launched(monkeypatch, tmp_path, capsys):
    """The device-lock-contention path (and anything else that fails between `workers = []`
    and the launch loop) reaches this finally block with an empty worker list -- 'waiting
    for 0 worker(s)' would be nonsensical, so the print is skipped entirely."""
    class _AlwaysBusyLock:
        def __init__(self, path):
            pass
        def acquire(self):
            raise RuntimeError("Android device is already in use by another run")
        def release(self):
            pass

    monkeypatch.setattr(sup, "_AndroidDeviceLock", _AlwaysBusyLock)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)
    cfg_path = _write_cfg(tmp_path)

    with pytest.raises(RuntimeError, match="already in use"):
        sup.run(str(cfg_path), stop_event=threading.Event())

    out = capsys.readouterr().out
    assert "Supervisor: stopping — waiting up to" not in out


def test_wedged_summary_uses_the_real_join_timeout_not_a_hardcoded_30s(monkeypatch, tmp_path, capsys):
    """Audit fix: the printed save summary used to hardcode 'did not stop within 30s'
    regardless of the run's ACTUAL bound (105s with the shipped opener config) -- a stale,
    wrong number in an operator-facing message. It must use join_timeout_s like the
    per-worker line just above it already did."""
    class _WedgedDriver(DatingAppDriver):
        def open_session(self):
            time.sleep(2.0)          # ignores stop_event -- simulates a wedged worker;
                                      # must outlast the 1.0s join timeout below to actually wedge
        def next_profile(self):
            return None
        def out_of_profiles(self):
            return True
        def like(self, opener=None, item_index=None, *, model_item_index=None):
            pass
        def dislike(self):
            pass
        def close(self):
            pass

    monkeypatch.setattr(sup, "_worker_join_timeout_s", lambda cfg: 1.0)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    store = _FakeStore()
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _WedgedDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    stop_event = threading.Event()
    threading.Timer(0.1, stop_event.set).start()

    sup.run(str(cfg_path), stop_event=stop_event)

    out = capsys.readouterr().out
    assert "did not stop within 1s" in out
    assert "30s" not in out
