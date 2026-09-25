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
from pathlib import Path
from types import SimpleNamespace

import pytest

import operation_love.supervisor as sup
from operation_love.config import OpenerCfg
from operation_love.drivers.base import DatingAppDriver
from operation_love.opener.opener import OpenerError
from operation_love.opener.service import OpenerService
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
    # This file tests supervisor lifecycle and shutdown behavior with fake drivers.  The
    # released Hinge targeting artifact and live-device calibration are covered in their
    # focused suites; keep this synthetic baseline runnable without weakening production.
    monkeypatch.setattr(sup.cfg_mod, "_validate_hinge_auto_release_evidence", lambda _cfg: None)
    original_reason = sup.platforms.unavailable_reason
    monkeypatch.setattr(
        sup.platforms, "unavailable_reason",
        lambda app, mode=None: None if app == "hinge" and mode == "auto"
        else original_reason(app, mode))

# enabled_apps: [hinge] -- hinge is the one platform the registry ships available/calibrated
# by default (platforms.py); "bumble" is now an Android target that starts out UNCALIBRATED,
# so it would be rejected by supervisor.run()'s new check_runnable() guard before a worker
# is ever built. __DATA_DIR__ is substituted with an isolated tmp_path by _run_with()/the
# tests below so stores and debug output never land in the real repo's data/ dir.  The Android
# lock intentionally no longer uses this path; the autouse fixture above isolates its stable
# per-user root separately.
_CONFIG = """
enabled_apps: [hinge]
mode: auto
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


def test_training_refuses_a_per_app_auto_override_before_startup(tmp_path):
    cfg = _CONFIG.replace("mode: auto", "mode: training").replace(
        "hinge: {}", "hinge:\n    mode: auto")
    cfg_path = _write_cfg(tmp_path, cfg)

    with pytest.raises(ValueError, match="Training cannot run while apps override"):
        sup.load_effective_config(str(cfg_path))


def test_training_keeps_clean_shape_errors_for_malformed_app_overrides(tmp_path):
    cfg_path = _write_cfg(tmp_path)

    with pytest.raises(ValueError, match="Config: enabled_apps must be a YAML list"):
        sup.load_effective_config(str(cfg_path), mode="training", enabled_apps={})


def test_direct_training_run_refuses_before_model_or_device_setup(monkeypatch):
    cfg = SimpleNamespace(
        enabled_apps=["hinge"], mode="training", apps={"hinge": {}},
    )
    monkeypatch.setattr(sup, "load_effective_config", lambda *_a, **_kw: cfg)

    with pytest.raises(ValueError, match="requires the local Hub decision bridge"):
        sup.run("unused-config.yaml")


class _FakeDriver(DatingAppDriver):
    def open_session(self):
        pass
    def next_profile(self):
        return None
    def out_of_profiles(self):
        return True                   # worker loop breaks immediately -> worker thread exits fast
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


def _run_with(monkeypatch, tmp_path, store, driver_factory=_FakeDriver):
    """Drive a whole real run() against `store`, with `driver_factory` built per app.

    `driver_factory` exists so a test can choose the run's TERMINAL STATE (a driver that
    raises ends its app in 'error') while still exercising the one shutdown path under test;
    the default stays the clean out-of-profiles driver every other caller wants.
    """
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: driver_factory())
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


class _DroppingStore(_FakeStore):
    """A store whose flush SUCCEEDS while rows were permanently lost earlier in the run.

    This is not a contrived shape -- it is what BigQueryStore actually does: a row BigQuery
    rejects forever (bad UTF-8 in raw_opener, an over-length field) is dropped after
    _MAX_INSERT_ATTEMPTS so the rest of the buffer can drain, opener/service.py catches the
    RuntimeError and only warns so the run continues, and the shutdown flush then finds an
    empty buffer and returns cleanly.
    """

    def saved_summary(self):
        return "labels=2 (DROPPED, never written: openers=1)"

    def dropped_rows(self):
        return {"openers": 1}


def test_permanently_dropped_rows_are_never_reported_as_all_data_saved(
        monkeypatch, tmp_path, capsys):
    """A successful flush proves the buffer is empty, not that every row landed.

    Every terminal signal in this run says success -- flush returned, save_err is None, phase is
    'stopped', no app errored -- and the operator was still told "✅ all data saved" over a row
    that is gone for good. The shutdown line must name the loss instead, and the bug report must
    be able to READ that line back: the tally lives on the store, which never reaches RunStatus,
    so the wording is the only channel there is (hence the shared DROPPED_ROWS_NOTICE literal
    both sides import, exercised end to end here so a reworded print cannot silently stop being
    detected).
    """
    from operation_love import bugreport

    snap = _run_with(monkeypatch, tmp_path, _DroppingStore())
    out = capsys.readouterr().out

    assert snap["phase"] == "stopped"                 # the save itself really did succeed
    assert "✅ all data saved" not in out
    assert "PERMANENTLY DROPPED" in out
    assert "(openers=1)" in out                       # which table lost a row, not just "some"
    assert "NOT an unqualified success" in out
    assert "accounted_provider_results=" in out       # the usual run tail is still printed

    line = next(ln for ln in out.splitlines() if "PERMANENTLY DROPPED" in ln)
    assert line.startswith(f"Run {snap['run_id']}: ")   # bugreport counts no tally without it
    assert bugreport._dropped_row_tallies([line]) == ["openers=1"]


class _HaltingDriver(DatingAppDriver):
    """Ends its app in the terminal 'error' state, the way an unrecognized-screen halt does."""

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


def test_dropped_rows_are_still_reported_when_a_worker_also_errored(
        monkeypatch, tmp_path, capsys):
    """A run that BOTH errored and permanently lost rows must report BOTH, not pick one.

    The loss notice used to be the third arm of the shutdown's if/elif chain, below wedged and
    errored, so this exact combination -- an errored run, i.e. the kind most likely to have
    CAUSED the drop -- took the errored arm and printed no loss line at all. That printed line
    is the tally's only channel out (the count lives on the store, which never reaches
    RunStatus, and bugreport._dropped_row_tallies reads it back out of the log ring), so the
    report rendered "COMPLETED WITH ERRORS" with no data-loss limitation on it whatsoever and
    the permanent loss vanished from the record. Headline and loss are independent facts:
    assert the errored headline SURVIVES alongside the notice, so a fix cannot trade one away.
    """
    from operation_love import bugreport

    snap = _run_with(monkeypatch, tmp_path, _DroppingStore(), driver_factory=_HaltingDriver)
    out = capsys.readouterr().out

    assert snap["apps"]["hinge"]["state"] == "error"     # the errored headline really applies
    assert "worker(s) ended with errors (hinge)" in out  # ... and is still the headline
    assert "✅ all data saved" not in out

    assert bugreport.DROPPED_ROWS_NOTICE in out, "the permanent loss went unreported"
    line = next(ln for ln in out.splitlines() if bugreport.DROPPED_ROWS_NOTICE in ln)
    # bugreport only counts a tally on a line carrying this prefix -- every terminal line in
    # this block prints it, and it is what ties the loss to THIS run in a shared log ring.
    assert line.startswith(f"Run {snap['run_id']}: ")
    assert bugreport._dropped_row_tallies([line]) == ["openers=1"]


def test_dropped_rows_are_still_reported_when_a_worker_also_wedged(
        monkeypatch, tmp_path, capsys):
    """Same additive rule on the other qualified headline: wedged does not hide the loss either.

    A wedged worker is the one terminal state that ends with phase != 'stopped', so it reaches
    the shutdown summary by a different route than the errored run above; pin it separately
    rather than assuming the two arms stay in step.
    """
    from operation_love import bugreport

    release = threading.Event()
    stop_event = threading.Event()

    class _WedgedDriver(DatingAppDriver):
        def open_session(self):
            # Armed from the worker's own first driver call (not a wall-clock timer) so the
            # stop cannot land on a startup checkpoint and abort before a worker exists, then
            # deliberately ignored until run() has returned -- see
            # test_wedged_worker_is_not_reported_as_unqualified_success for the full rationale.
            stop_event.set()
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

    monkeypatch.setattr(sup, "_worker_join_timeout_s", lambda cfg: 0.01)  # don't wait ~105s
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _DroppingStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _WedgedDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)
    cfg_path = _write_cfg(tmp_path)

    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=stop_event)
    release.set()                    # let the deliberately-wedged thread finish now
    snap = captured["status"].snapshot()
    out = capsys.readouterr().out

    assert any(a["state"] == "wedged" for a in snap["apps"].values())
    assert "did not stop within" in out                  # the wedged headline is still there
    assert "✅ all data saved" not in out

    assert bugreport.DROPPED_ROWS_NOTICE in out, "the permanent loss went unreported"
    line = next(ln for ln in out.splitlines() if bugreport.DROPPED_ROWS_NOTICE in ln)
    assert line.startswith(f"Run {snap['run_id']}: ")
    assert bugreport._dropped_row_tallies([line]) == ["openers=1"]


def test_clean_shutdown_still_reports_all_data_saved_when_nothing_was_dropped(
        monkeypatch, tmp_path, capsys):
    """The loss branch must not swallow the one line that reports a genuinely clean run."""
    _run_with(monkeypatch, tmp_path, _FakeStore())
    out = capsys.readouterr().out

    assert "✅ all data saved" in out
    assert "PERMANENTLY DROPPED" not in out


@pytest.mark.parametrize(
    "tally",
    [{}, {"openers": 0}, {"openers": True}, {"openers": "1"}, None, "openers=1"],
    ids=["empty", "zero", "bool", "text-count", "none", "not-a-mapping"],
)
def test_dropped_row_tally_never_invents_a_loss_from_an_unusable_answer(tally):
    """Nothing but a positive integer count is a dropped row.

    `True` is an int in Python and would otherwise read as "1 row lost"; a zero entry is the
    healthy case spelled the long way. Inventing a loss here would teach the operator to ignore
    the one line that reports a real one.
    """
    store = SimpleNamespace(dropped_rows=lambda: tally)

    assert sup._dropped_row_tally(store) == {}


def test_dropped_row_tally_survives_a_store_that_cannot_answer(capsys):
    """This runs AFTER the final flush: an exception raised over a diagnostic tally would turn a
    successful save into a crash. Report the failure, then behave like every store that never
    had the method at all."""
    class _Broken:
        def dropped_rows(self):
            raise RuntimeError("no tally for you")

    assert sup._dropped_row_tally(_Broken()) == {}
    assert sup._dropped_row_tally(SimpleNamespace()) == {}      # a test double without it
    assert "no tally for you" in capsys.readouterr().out


def test_terminal_phase_is_published_only_after_app_states_are_terminal(monkeypatch, tmp_path):
    """A terminal global phase is the Hub/bug-report publication boundary."""
    terminal_publications = []
    real_status = sup.RunStatus

    class _ObservedStatus(real_status):
        def set_global(self, **fields):
            if fields.get("phase") in {"stopped", "wedged", "save_failed"}:
                terminal_publications.append(self.snapshot())
            return super().set_global(**fields)

    monkeypatch.setattr(sup, "RunStatus", _ObservedStatus)
    snap = _run_with(monkeypatch, tmp_path, _FakeStore())

    assert snap["phase"] == "stopped"
    assert len(terminal_publications) == 1
    assert all(row["state"] != "saving"
               for row in terminal_publications[0]["apps"].values())


def test_startup_stop_reports_save_failed_when_store_cleanup_fails(monkeypatch, tmp_path):
    stop = threading.Event()

    class _StopAfterLoadStore(_FakeStore):
        def load_labels(self):
            stop.set()
            return []

    store = _StopAfterLoadStore(flush_error=RuntimeError("startup drain rejected"))
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(
        sup, "make_driver", lambda *_args: pytest.fail("worker must not be constructed"))
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda event: None)
    _patch_no_adb(monkeypatch)
    captured = {}

    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=stop)

    snap = captured["status"].snapshot()
    assert store.closed is True
    assert snap["phase"] == "save_failed"
    assert all(row["state"] == "error" for row in snap["apps"].values())


def test_startup_stop_reports_save_failed_when_store_close_fails(monkeypatch, tmp_path):
    stop = threading.Event()

    class _StopAfterLoadStore(_FakeStore):
        def load_labels(self):
            stop.set()
            return []

        def close(self):
            self.closed = True
            raise RuntimeError("startup close rejected")

    store = _StopAfterLoadStore()
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(
        sup, "make_driver", lambda *_args: pytest.fail("worker must not be constructed"))
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda event: None)
    _patch_no_adb(monkeypatch)
    captured = {}

    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=stop)

    snap = captured["status"].snapshot()
    assert store.closed is True
    assert snap["phase"] == "save_failed"
    assert all(row["state"] == "error" for row in snap["apps"].values())


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
    training_bridge = _Bridge()
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
        worker.training_action_bridge = training_bridge

    with pytest.raises(RuntimeError, match="thread creation refused"):
        sup.run(str(cfg_path), stop_event=threading.Event(), on_worker=_bind)

    assert training_bridge.unregistered is not None
    assert driver.close_attempted is True
    assert store.flushed is True
    assert store.closed is True
    assert "driver cleanup also failed" in capsys.readouterr().out


def test_worker_binding_failure_releases_driver_and_registered_training_bridge(monkeypatch, tmp_path):
    """A bridge can reject a replacement before Thread.start(); that is still a launch failure."""
    class _TrackingStore(_FakeStore):
        def __init__(self):
            super().__init__()
            self.flushed = False

        def flush(self):
            self.flushed = True

    class _TrackingDriver(_FakeDriver):
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _UnstartedWorker:
        def __init__(self, *args, **kwargs):
            self.app = args[0]

        def start(self):
            raise AssertionError("on_worker failure must prevent Thread.start")

        def join(self, timeout=None):
            raise AssertionError("an unstarted worker must never be joined")

        def is_alive(self):
            raise AssertionError("an unstarted worker must never enter liveness checks")

    class _Bridge:
        def __init__(self):
            self.unregistered = None

        def unregister(self, worker):
            self.unregistered = worker

    store, driver = _TrackingStore(), _TrackingDriver()
    training_bridge = _Bridge()
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: driver)
    monkeypatch.setattr(sup, "Worker", _UnstartedWorker)
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    def _bind_then_fail(worker):
        worker.training_action_bridge = training_bridge
        raise RuntimeError("training bridge rejected replacement")

    with pytest.raises(RuntimeError, match="bridge rejected replacement"):
        sup.run(str(cfg_path), stop_event=threading.Event(), on_worker=_bind_then_fail)

    assert training_bridge.unregistered is not None
    assert driver.closed and store.flushed and store.closed


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


def test_close_only_failure_is_a_save_failure_not_a_green_shutdown(monkeypatch, tmp_path):
    """The final close can flush rows that raced the supervisor's earlier flush.

    A close-only error must therefore set the same terminal failure state and propagate. The
    companion test above still proves an earlier flush error remains the more useful cause.
    """
    store = _CloseAlsoFailsStore(
        flush_error=None,
        close_error=RuntimeError("final archive flush rejected"),
    )
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)
    captured = {}

    with pytest.raises(RuntimeError, match="final archive flush rejected"):
        sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
                stop_event=threading.Event())

    snap = captured["status"].snapshot()
    assert store.closed is True
    assert snap["phase"] == "save_failed"
    assert all(app["state"] == "error" for app in snap["apps"].values())


def test_worker_error_state_survives_shutdown_not_overwritten_to_stopped(
        monkeypatch, tmp_path, capsys):
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

    cfg_text = _CONFIG
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
    output = capsys.readouterr().out
    assert "buffered rows saved" in output
    assert "does not prove every landed action" in output
    assert "✅ all data saved" not in output


def test_status_callback_receives_effective_app_mode_before_workers_start(monkeypatch, tmp_path):
    """The Hub captures RunStatus before workers are constructed, so its first snapshot must
    already carry AUTO rather than AppStatus's training-shaped default."""
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: _FakeStore())
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    published = []
    sup.run(str(cfg_path), on_status=lambda status: published.append(status.snapshot()),
            stop_event=threading.Event())

    assert len(published) == 1
    assert published[0]["apps"]["hinge"]["mode"] == "auto"


@pytest.mark.parametrize("terminal_state", ["out_of_profiles", "rate_limited"])
def test_normal_terminal_reason_survives_successful_save(monkeypatch, tmp_path, terminal_state):
    class _TerminalWorker:
        def __init__(self, app, *args, status=None, mode="auto", **kwargs):
            self.app = app
            self.status = status
            self.mode = mode
        def start(self):
            self.status.set_app(self.app, mode=self.mode, state=terminal_state)
        def join(self, timeout=None):
            pass
        def is_alive(self):
            return False

    cfg_path = _write_cfg(tmp_path, _CONFIG)
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
        def __init__(self, app, *args, status=None, mode="auto", **kwargs):
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


def test_blocked_terminal_state_and_reason_survive_successful_save(monkeypatch, tmp_path):
    """A driver-detected deck block is a terminal result, not a plain stopped run.

    Worker correctly publishes ``state='blocked'`` with a ``deck_blocked`` reason before
    returning.  Supervisor then temporarily stamps every app ``saving`` during flush.  Keep
    the distinct terminal result when it restores states after a successful save; otherwise a
    completed run hides the actionable Hinge+ / foreground-block diagnosis from the hub and
    bug report.
    """
    class _BlockedWorker:
        def __init__(self, app, *args, status=None, mode="auto", **kwargs):
            self.app = app
            self.status = status
            self.mode = mode

        def start(self):
            self.status.set_app(
                self.app, mode=self.mode, state="blocked",
                stop_reason="Hinge is out of free likes for today — the Hinge+ upgrade screen is up",
                stop_kind="deck_blocked",
            )

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
    monkeypatch.setattr(sup, "Worker", _BlockedWorker)
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=threading.Event())

    snap = captured["status"].snapshot()
    app = snap["apps"]["hinge"]
    assert snap["phase"] == "stopped"
    assert app["state"] == "blocked"
    assert app["stop_reason"] == "Hinge is out of free likes for today — the Hinge+ upgrade screen is up"
    assert app["stop_kind"] == "deck_blocked"


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
                     *args, status=None, mode="auto", **kwargs):
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


def test_pre_requested_stop_skips_provider_and_device_preflight(monkeypatch, tmp_path):
    """A cancellation that predates run() must not start network/adb startup work."""
    cfg_path = _write_cfg(tmp_path, _gemini_cfg_text())
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _SpyGeminiOpener.instances = []
    monkeypatch.setattr(sup, "GeminiOpener", _SpyGeminiOpener)
    touched = []

    class _NeverProbeCaps:
        @classmethod
        def detect(cls, *args, **kwargs):
            touched.append("capabilities")
            raise AssertionError("Capabilities.detect must not run after Stop")

    monkeypatch.setattr(sup, "Capabilities", _NeverProbeCaps)
    monkeypatch.setattr(
        sup, "_android_adb_preflight",
        lambda *args: (_ for _ in ()).throw(AssertionError("adb preflight must not run after Stop")),
    )
    monkeypatch.setattr(
        sup, "make_store",
        lambda *_args: (_ for _ in ()).throw(AssertionError("store must not open after Stop")),
    )
    stop_event = threading.Event()
    stop_event.set()
    captured = {}

    sup.run(str(cfg_path), stop_event=stop_event,
            on_status=lambda status: captured.__setitem__("status", status))

    assert _SpyGeminiOpener.instances == []
    assert touched == []
    assert captured["status"].snapshot()["phase"] == "stopped"


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
    release = threading.Event()

    class _WedgedDriver(DatingAppDriver):
        def open_session(self):
            # Stop pressed mid-run: armed from the worker's own first driver call, so it
            # cannot land on one of run()'s startup stop-checkpoints and abort before a
            # worker was ever launched (a 0.1s wall-clock timer here lost exactly that race
            # on a slow CI container). The wait below then IGNORES the stop it just set --
            # simulating a worker stuck mid-capture -- held until the test releases it after
            # run() returns, so the wedge verdict cannot depend on machine speed either.
            stop_event.set()
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

    stop_event = threading.Event()   # set by _WedgedDriver.open_session, i.e. after launch

    captured = {}
    sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
            stop_event=stop_event)
    release.set()                    # let the deliberately-wedged thread finish now

    snap = captured["status"].snapshot()

    assert store.closed is True                  # a wedged worker must not block quit forever
    assert snap["phase"] != "stopped"             # not the unqualified-success phase
    assert any(a["state"] == "wedged" for a in snap["apps"].values())
    out = capsys.readouterr().out
    assert "✅ all data saved" not in out
    assert "accounted_provider_results=" in out
    assert "HTTP fallback failures logged separately and excluded" in out
    assert "provider_spend=$" in out
    assert "openers=" not in out
    assert "Python stack at wedge" in out
    assert "test_supervisor.py" in out
    assert "open_session" in out


def test_wedged_worker_stack_snapshot_is_bounded_and_locals_free():
    """A supervisor wedge report identifies the live code location without dumping data.

    Use a deeper-than-cap stack and an intentionally secret-looking local value: the former
    must be bounded and marked truncated; the latter must never be read or printed.
    """
    release = threading.Event()
    entered = threading.Event()
    secret = "OPLOVE_TEST_SECRET_must_not_appear"

    def stuck(depth):
        private_request_body = secret
        if depth:
            return stuck(depth - 1)
        entered.set()
        release.wait()
        return private_request_body

    worker = threading.Thread(target=stuck, args=(sup._WEDGED_WORKER_STACK_MAX_FRAMES + 3,))
    worker.start()
    assert entered.wait(_LIVENESS_TIMEOUT_S)
    try:
        lines = sup._wedged_worker_stack_lines(worker)
    finally:
        release.set()
        worker.join(_LIVENESS_TIMEOUT_S)

    rendered = "\n".join(lines)
    assert "test_supervisor.py" in rendered
    assert "stuck" in rendered
    assert secret not in rendered
    assert "older frames omitted" in rendered
    assert len(lines) == sup._WEDGED_WORKER_STACK_MAX_FRAMES + 1


# --- registry guard: run() rejects an unrunnable platform selection up front ---------------

@pytest.mark.parametrize("mode", ["training", "auto"])
def test_run_rejects_uncalibrated_bumble_before_touching_anything(
        monkeypatch, tmp_path, mode):
    """The check_runnable() guard at the top of run() must fire BEFORE any driver is built,
    using cfg_mod.validate()'s message verbatim (this exercises the guard itself, not just
    validate() -- see test_config.py for validate()'s own coverage of the same rule)."""
    from operation_love import platforms

    cfg_text = _CONFIG.replace("enabled_apps: [hinge]", "enabled_apps: [bumble]").replace(
            "mode: auto", f"mode: {mode}").replace(
        "apps:\n  hinge: {}", "apps:\n  bumble: {}")
    if mode == "training":
        cfg_text = cfg_text.replace(
            "opener:\n  enabled: false", "opener:\n  enabled: true\n  thinking:\n    gemini-3.6-flash: {}")
        cfg_text = cfg_text.replace("pricing: {}", "pricing:\n    gemini-3.6-flash: {input: 0, output: 0}")
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
    cfg_text = _CONFIG.replace("hinge: {}", "hinge:\n    mode: auto")
    cfg_path = _write_cfg(tmp_path, cfg_text)

    with pytest.raises(ValueError, match="Training cannot run while apps override"):
        sup.load_effective_config(str(cfg_path), mode="training", enabled_apps=["hinge"])


def test_effective_config_installs_hinge_auto_readiness_before_registry_check():
    """A fresh process accepts the released config without relying on a manual live probe."""
    from operation_love import targeting_policy as tp

    tp._reset_installed_still_photo_bound_for_tests()
    cfg = sup.load_effective_config("config.yaml", mode="auto", enabled_apps=["hinge"])

    assert cfg.mode == "auto"


@pytest.mark.parametrize("initial_is_licensed", [True, False])
def test_effective_config_registry_check_uses_its_validation_snapshot_during_a_race(
        monkeypatch, tmp_path, initial_is_licensed):
    """Another validation cannot change the readiness answer after this config snapshots it."""
    from operation_love import targeting_policy as tp

    tp._reset_installed_still_photo_bound_for_tests()
    registry_entered = threading.Event()
    allow_registry = threading.Event()
    original_snapshot = sup.cfg_mod.validate_and_snapshot_still_photo_licence
    original_check = sup.platforms.check_runnable
    fallback_reason = sup.platforms.unavailable_reason
    result = {}

    def capture_snapshot(cfg):
        snapshot = original_snapshot(cfg)
        result["snapshot"] = snapshot
        return snapshot

    def policy_reason(app, mode=None):
        if app == "hinge" and mode == "auto":
            blocker = tp.hinge_targeting_unavailable_reason()
            return None if blocker is None else f"Hinge Auto is blocked: {blocker}."
        return fallback_reason(app, mode)

    def paused_check(*args, **kwargs):
        registry_entered.set()
        assert allow_registry.wait(_LIVENESS_TIMEOUT_S)
        return original_check(*args, **kwargs)

    monkeypatch.setattr(sup.cfg_mod, "validate_and_snapshot_still_photo_licence",
                        capture_snapshot)
    # This file's autouse fixture masks the real Hinge AUTO policy gate for unrelated lifecycle
    # tests. Restore its exact policy dependency here so `check_runnable` is genuinely tested.
    monkeypatch.setattr(sup.platforms, "unavailable_reason", policy_reason)
    monkeypatch.setattr(sup.platforms, "check_runnable", paused_check)
    initial_path = "config.yaml" if initial_is_licensed else str(_write_cfg(tmp_path))

    def load_initial():
        try:
            result["config"] = sup.load_effective_config(
                initial_path, mode="auto", enabled_apps=["hinge"])
        except BaseException as exc:  # the assertion below needs the exact registry result
            result["error"] = exc

    loading = threading.Thread(target=load_initial)
    loading.start()
    assert registry_entered.wait(_LIVENESS_TIMEOUT_S)
    assert (result["snapshot"] is not None) is initial_is_licensed

    if initial_is_licensed:
        # An invalid config clears the mutable default before it reports its own error.  The
        # already-snapshotted licensed config must nevertheless pass its delayed registry check.
        invalid = sup.cfg_mod.load("config.yaml")
        invalid.enabled_apps = []
        with pytest.raises(ValueError, match="enabled_apps is empty"):
            sup.cfg_mod.validate(invalid)
    else:
        # Conversely, a previously unlicensed config must not borrow readiness from a different
        # config validated while it waits to enter the registry.
        sup.cfg_mod.validate(sup.cfg_mod.load("config.yaml"))
    allow_registry.set()
    loading.join(_LIVENESS_TIMEOUT_S)

    assert not loading.is_alive()
    if initial_is_licensed:
        assert "error" not in result
        assert result["config"].mode == "auto"
    else:
        assert isinstance(result.get("error"), ValueError)
        assert "Hinge Auto is blocked" in str(result["error"])


def test_effective_config_rejects_explicit_falsy_overrides(tmp_path):
    cfg_path = _write_cfg(tmp_path)

    with pytest.raises(ValueError, match="Config: enabled_apps is empty"):
        sup.load_effective_config(str(cfg_path), enabled_apps=[])
    with pytest.raises(ValueError, match="Config: mode must be 'training' or 'auto'"):
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
    service (contract: OpenerService(client, tracker, store, style, max_attempts=5,
    replay_corpus_dir=None, replay_corpus_max_captures=0, replay_corpus_max_age_days=0,
    deadletter_path=None))."""
    instances: list = []

    def __init__(self, client, tracker, store, style, max_attempts=5, *,
                 replay_corpus_dir=None, replay_corpus_max_captures=0,
                 replay_corpus_max_age_days=0, deadletter_path=None):
        self.client = client
        self.tracker = tracker
        self.store = store
        self.style = style
        self.max_attempts = max_attempts
        self.replay_corpus_dir = replay_corpus_dir
        self.replay_corpus_max_captures = replay_corpus_max_captures
        self.replay_corpus_max_age_days = replay_corpus_max_age_days
        self.deadletter_path = deadletter_path
        _SpyOpenerService.instances.append(self)


def test_opener_rejection_deadletter_path_reaches_the_constructed_opener_service(
        monkeypatch, tmp_path):
    """The rejection dead-letter must actually be WIRED, not merely available.

    `OpenerService(deadletter_path=...)` defaults to None, and a None path makes
    `_write_opener_rejection_deadletter` return immediately -- so a supervisor that forgets to
    pass it leaves the diagnostic silently inert in production while every unit test around it
    stays green, because those tests inject their own tmp_path. That is precisely the failure
    this diagnostic exists to end: production BigQuery `opener_rejections` sat at zero rows for
    the table's whole history while rejections were demonstrably happening, because the only
    record of the failure was a print() nothing captured.

    The path must also be derived from the configured data_dir rather than a hardcoded literal,
    the same way db_file is (see config.py), so an operator who relocates data_dir takes the
    dead-letter with it instead of leaving it writing somewhere they are not looking."""
    cfg_path = _write_cfg(tmp_path, _CONFIG)
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
    wired = _SpyOpenerService.instances[0].deadletter_path
    assert wired, "supervisor must pass a deadletter_path; None leaves the diagnostic inert"
    from operation_love import config as _cfg_mod
    expected = _cfg_mod.load(str(cfg_path)).data_dir / "opener_rejection_deadletter.jsonl"
    assert Path(wired) == expected


class _StoreWithStrandedOpenerRejections(_FakeStore):
    """Simulates BigQueryStore at storage.bigquery.flush_every > 1 (config.yaml currently
    ships 1, but the constructor/`make_store` DEFAULT is 25 -- see that comment and
    _write_opener_rejection_deadletter's "THE flush_every DEPENDENCY" docstring section): a
    rejection row was accepted into the buffer WITHOUT raising (BigQueryStore._maybe_flush only
    flushes once the buffer reaches flush_every), so opener/service.py's own per-call
    `except Exception as store_exc:` around record_opener_rejection never ran and never
    dead-lettered it. The wire failure only surfaces here, at the shutdown flush -- exactly the
    blind spot Task 1(a) (2026-09-17) closes."""

    def __init__(self, flush_error, pending_rows):
        super().__init__(flush_error=flush_error)
        self._pending_rows = pending_rows

    def pending_opener_rejections(self):
        return list(self._pending_rows)


def test_shutdown_flush_failure_deadletters_stranded_opener_rejections(monkeypatch, tmp_path):
    """A wire failure at the FINAL shutdown flush must still dead-letter whatever
    opener_rejections rows the store was still holding -- not only rows that happened to raise
    synchronously inside maybe_opener's own try/except (that narrower case is already covered
    by test_opener_rejection_deadletter_path_reaches_the_constructed_opener_service and by
    tests/test_opener_service.py; this test is the flush_every > 1 gap Task 1 exists for).

    Uses the REAL OpenerService (not a spy) so the REAL _write_opener_rejection_deadletter
    actually runs and a real file lands on disk -- a spy would only prove the path was wired,
    not that the shutdown path calls the writer correctly.
    """
    import json

    cfg_path = _write_cfg(tmp_path, _CONFIG)
    flush_exc = RuntimeError("BigQuery insert errors for opener_rejections: [wire down]")
    stranded_row = {
        "run_id": "run-stranded-1", "app": "hinge", "created_at": 0, "model": "gemini-3-flash",
        "attempt": 2, "reason_code": "unconfirmed_location_followup",
        "reason": "the model referenced a location detail that could not be confirmed",
        "raw_opener": "I noticed the mountain in your third photo...",
        "prompt_sha256": "deadbeef" * 8,
    }
    store = _StoreWithStrandedOpenerRejections(flush_exc, [stranded_row])
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _FakeDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    with pytest.raises(RuntimeError, match="BigQuery insert errors"):
        sup.run(str(cfg_path), stop_event=threading.Event())

    from operation_love import config as _cfg_mod
    deadletter_path = _cfg_mod.load(str(cfg_path)).data_dir / "opener_rejection_deadletter.jsonl"
    assert deadletter_path.exists(), (
        "a flush failure at shutdown must still dead-letter whatever opener_rejections rows "
        "the store was still holding -- see supervisor._deadletter_stranded_opener_rejections")
    entries = [json.loads(line) for line in deadletter_path.read_text().splitlines() if line.strip()]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["branch"] == "shutdown_flush"
    assert entry["row"]["run_id"] == "run-stranded-1"
    assert entry["row"]["reason_code"] == "unconfirmed_location_followup"
    assert entry["row"]["raw_opener"] == "I noticed the mountain in your third photo..."
    assert "wire down" in entry["exc_str"]


class _StoreWithRejectionThatStaysBuffered:
    """Reproduces the ACTUAL BigQueryStore shutdown bug end to end, not just its symptom.

    ranker/bigquery_store.py's own `_flush_table` comment: "Keep the whole batch buffered so
    it gets resent on the next flush" -- a FAILED FLUSH NEVER CLEARS THE ROW. So
    record_opener_rejection here does two things a real flush_every=1 wire failure also does
    in one call: it appends the row (so a later pending_opener_rejections() snapshot can see
    it) AND it raises (so opener/service.py's own per-call `except Exception as store_exc:`
    fires and writes ITS dead-letter entry). Both effects land on the SAME row, which is
    exactly the shape that made supervisor._deadletter_stranded_opener_rejections double-count
    before the 2026-09-17 fix: the row the shutdown sweep finds is not new evidence, it is the
    one the per-call site already covered.
    """

    def __init__(self):
        self._buf: list[dict] = []

    def record_opener_rejection(self, run_id, app, model, attempt, reason_code, reason,
                                raw_opener, *, prompt_sha256=None):
        self._buf.append({
            "run_id": run_id, "app": app, "created_at": "2026-09-17T00:00:00+00:00",
            "model": model, "attempt": int(attempt), "reason_code": reason_code,
            "reason": reason, "raw_opener": raw_opener, "prompt_sha256": prompt_sha256,
        })
        raise RuntimeError("BigQuery insert errors for opener_rejections: [wire down]")

    def pending_opener_rejections(self):
        return list(self._buf)


class _OpenerErrorClient:
    """A minimal OpenerClient whose every call raises a per-profile OpenerError -- the
    simplest of the four maybe_opener() failure branches (no retry loop, no billed usage to
    track), so driving one real maybe_opener() call is enough to reach the SAME per-call
    `except Exception as store_exc:` / _write_opener_rejection_deadletter path the double-
    dead-lettering bug lived in."""

    def generate(self, *args, **kwargs):
        raise OpenerError("Gemini opener: photo index 0 could not be decoded")


class _MinimalTracker:
    """Just enough CostTracker surface for the OpenerError branch: it checks
    budget_reached() once, up front, and never calls record() (that branch has no billed
    usage -- see service.py's own comment on why OpenerError is not retried)."""

    def budget_reached(self):
        return False


def _read_deadletter_jsonl(path):
    import json as _json
    return [_json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_shutdown_sweep_does_not_double_deadletter_a_row_already_recorded_per_call(tmp_path):
    """Regression pin for the double-dead-lettering bug fixed 2026-09-17.

    The bug: `_deadletter_stranded_opener_rejections` asked the store what opener_rejections
    rows were still buffered and wrote a fresh dead-letter entry for every one of them --
    without checking whether the SAME row had already earned an entry from OpenerService's
    own per-call `except Exception as store_exc:` sites. Since a failed flush never clears
    the buffered row (ranker/bigquery_store.py's `_flush_table` comment), and the shipped
    config pins `storage.bigquery.flush_every: 1` (every rejection flushes synchronously),
    every rejection that failed during an outage got dead-lettered TWICE: once from inside
    maybe_opener() the instant the write raised, and again from the shutdown sweep because
    the row was still sitting in the buffer. A 3-rejection outage produced 6 entries for 3
    actually-lost rows.

    This drives a REAL OpenerService.maybe_opener() call (not a spy) so the real per-call
    dead-letter write and the real in-run identity tracking
    (OpenerService._deadlettered_rejection_keys / is_rejection_row_deadlettered) both
    execute, then calls the real `_deadletter_stranded_opener_rejections` directly with the
    store's own pending_opener_rejections() snapshot -- exactly the two calls a real
    shutdown-during-outage run makes, in the same order.
    """
    deadletter_path = tmp_path / "deadletter.jsonl"
    store = _StoreWithRejectionThatStaysBuffered()
    service = OpenerService(_OpenerErrorClient(), _MinimalTracker(), store, "casual",
                            deadletter_path=str(deadletter_path))

    # One profile's opener call: OpenerError -> record_opener_rejection raises -> the
    # SERVICE's own per-call except already writes ONE dead-letter entry (branch=
    # "opener_error") and leaves the row stuck in the store's buffer -- exactly like the
    # real bug scenario.
    out = service.maybe_opener("run-1", "hinge", object())
    assert out is None

    after_percall = _read_deadletter_jsonl(deadletter_path)
    assert len(after_percall) == 1, "the per-call path itself must write exactly one entry"
    assert after_percall[0]["branch"] == "opener_error"

    # Shutdown: the SAME row is still buffered (a failed flush never clears it) and
    # store.flush() fails again for the same wire reason, so supervisor.py's shutdown
    # handler calls _deadletter_stranded_opener_rejections. Pre-fix this wrote a SECOND
    # entry for the identical row; the fix must skip it.
    sup._deadletter_stranded_opener_rejections(
        store, service,
        RuntimeError("BigQuery insert errors for opener_rejections: [wire down]"))

    after_shutdown = _read_deadletter_jsonl(deadletter_path)
    assert len(after_shutdown) == 1, (
        "shutdown sweep wrote a duplicate dead-letter entry for a row the per-call path "
        "already recorded -- see supervisor._deadletter_stranded_opener_rejections and "
        "OpenerService.is_rejection_row_deadlettered")


def test_opener_max_attempts_reaches_the_constructed_opener_service(monkeypatch, tmp_path):
    """cfg.opener.max_attempts must reach OpenerService's constructor as the keyword
    max_attempts -- this is the setting that controls how many times a rejected AI response
    is re-asked before the run stops (owner rule, 2026-08-10: no commentless likes). Uses a
    distinctive value (7) rather than the class default (5) so a mutation that drops the
    kwarg entirely -- silently falling back to OpenerService's own default -- cannot pass
    unnoticed."""
    cfg_text = _CONFIG.replace(
        "opener:\n  enabled: false",
        "opener:\n  enabled: false\n  max_attempts: 7")
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


def test_opener_replay_corpus_disabled_by_default_reaches_the_constructed_opener_service(
        monkeypatch, tmp_path):
    """The shipped default (opener.replay_corpus_enabled: false) must reach OpenerService's
    constructor as replay_corpus_dir=None -- this writes REAL PEOPLE'S PHOTOS to local disk, so
    the off-by-default contract must hold even when config.yaml never mentions the flag."""
    cfg_path = _write_cfg(tmp_path)   # base _CONFIG sets no opener.replay_corpus_enabled
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
    assert _SpyOpenerService.instances[0].replay_corpus_dir is None


def test_opener_replay_corpus_enabled_reaches_the_constructed_opener_service(monkeypatch, tmp_path):
    """Turning the owner's flag on must reach OpenerService as a real directory (the module's
    own DEFAULT_CORPUS_DIR), not merely a truthy placeholder."""
    from operation_love.opener.replay_corpus import DEFAULT_CORPUS_DIR
    cfg_text = _CONFIG.replace(
        "opener:\n  enabled: false",
        "opener:\n  enabled: false\n  replay_corpus_enabled: true")
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
    assert _SpyOpenerService.instances[0].replay_corpus_dir == DEFAULT_CORPUS_DIR


# --- opener.replay_corpus_max_captures / opener.replay_corpus_max_age_days: the retention
# bounds must reach the constructed OpenerService, threaded unconditionally (they are inert
# whenever replay_corpus_dir above is None -- see OpenerService._capture_replay_corpus's own
# early return) -- so both defaults and explicit overrides must be pinned, exactly like
# max_attempts above. ------------------------------------------------------------------------

def test_opener_replay_corpus_retention_defaults_reach_the_constructed_opener_service(
        monkeypatch, tmp_path):
    """The shipped OpenerCfg defaults (400 captures / 180 days) must reach the constructor when
    config.yaml never mentions either knob -- not just when they are set explicitly."""
    cfg_path = _write_cfg(tmp_path)   # base _CONFIG sets neither retention knob
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
    assert _SpyOpenerService.instances[0].replay_corpus_max_captures == 400
    assert _SpyOpenerService.instances[0].replay_corpus_max_age_days == 180


def test_opener_replay_corpus_retention_overrides_reach_the_constructed_opener_service(
        monkeypatch, tmp_path):
    """Explicit config.yaml overrides for both retention knobs must reach the constructor --
    distinctive values (11 / 22), neither of which is either knob's own class default, so a
    mutation that drops a kwarg (silently falling back to OpenerCfg's default) cannot pass
    unnoticed."""
    cfg_text = _CONFIG.replace(
        "opener:\n  enabled: false",
        "opener:\n  enabled: false\n  replay_corpus_max_captures: 11"
        "\n  replay_corpus_max_age_days: 22")
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
    assert _SpyOpenerService.instances[0].replay_corpus_max_captures == 11
    assert _SpyOpenerService.instances[0].replay_corpus_max_age_days == 22


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
    # A SECOND wait, not a bare read (found 2026-08-23, an intermittent under the parallel
    # default): the poll above proves the reaper called `device_lock.release()`, which is a
    # DIFFERENT event from the reaper finishing its bookkeeping. `_reap_wedged_android_lock`'s
    # reaper releases the OS lock first and only then discards itself from
    # `_RETAINED_DEVICE_LOCKS` (supervisor.py -- and that order is correct in production; you
    # would never drop the registry entry while the flock is still held). So there is a real
    # window where the contender can acquire while the set still holds the entry, and reading it
    # instantaneously here is a race the test loses under CPU contention. This is the same
    # liveness bound as `_LIVENESS_TIMEOUT_S` above: the property is "the reaper eventually
    # finishes", never "it finishes within N seconds".
    deadline = time.monotonic() + _LIVENESS_TIMEOUT_S
    while True:
        with sup._RETAINED_DEVICE_LOCKS_GUARD:
            retained = set(sup._RETAINED_DEVICE_LOCKS)
        if not retained:
            break
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"reaper released the lock but never cleared its registry entry: {retained}")
        time.sleep(0.01)


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


# --- F: an already-landed decision's ARCHIVE is the other thing a healthy worker can be
# inside when Stop arrives, and it outlasts the opener bound above. ------------------------

def _archive_cfg(apps):
    return SimpleNamespace(enabled_apps=list(apps), apps=apps)


def test_archive_grace_is_read_off_the_stores_own_whole_archive_deadline():
    """The grace must track the store's deadline for THIS run's capture budget, not restate a
    number: the shipped 12-screencap Hinge read is priced far above the 105s opener bound, so a
    flat bound is exactly what misreports a healthy archiving worker as wedged."""
    grace = sup._archive_join_grace_s(_archive_cfg({"hinge": {"scroll_captures": 12}}))
    assert grace == sup._profile_upload_deadline_s(12)
    # Several times the 105s opener bound at the shipped capture budget -- the gap this whole
    # branch exists to cover. Compared against that bound rather than a literal so a retune of
    # either side keeps the relationship under test instead of pinning two numbers.
    assert grace > sup._worker_join_timeout_s(
        _JoinTimeoutCfg(OpenerCfg(enabled=True, request_timeout_s=90))) * 3
    # A budget the config does not pin is priced at the Training review ceiling, which is also
    # the largest capture config validation admits for a Training run.
    assert sup._archive_join_grace_s(_archive_cfg({"hinge": {}})) == (
        sup._profile_upload_deadline_s(sup.MAX_PROFILE_REVIEW_FRAMES))
    # Never below the store's own floor, whatever the config says.
    assert sup._archive_join_grace_s(_archive_cfg({"hinge": {"scroll_captures": 1}})) == 180.0


def test_only_a_store_reporting_a_registered_archive_can_extend_the_join():
    """A store with no such seam (sqlite, every test double here) keeps the flat bound, so a
    genuinely wedged worker stays exactly as quick to give up on as before."""
    assert sup._store_archive_in_flight(_FakeStore()) is False
    assert sup._store_archive_in_flight(SimpleNamespace(_active_async_writes=0)) is False
    assert sup._store_archive_in_flight(SimpleNamespace(_active_async_writes=1)) is True
    assert sup._store_archive_in_flight(
        SimpleNamespace(archive_writes_in_flight=lambda: 2)) is True
    # A probe that misbehaves must never take shutdown down with it.
    assert sup._store_archive_in_flight(SimpleNamespace(
        archive_writes_in_flight=lambda: (_ for _ in ()).throw(RuntimeError("boom")))) is False


def test_worker_still_archiving_a_landed_decision_is_not_reported_wedged(
        monkeypatch, tmp_path, capsys):
    """The consequence of the old flat bound was not a cosmetic label: the supervisor flushes
    and CLOSES the store on the far side of this join, and the decision/opener/label rows that
    follow an already-landed Like are then refused as late writes."""
    stop_event = threading.Event()
    archive_done = threading.Event()
    driver_closed = threading.Event()

    class _ArchivingStore(_FakeStore):
        def archive_writes_in_flight(self):
            return 0 if archive_done.is_set() else 1

    class _ArchivingDriver(DatingAppDriver):
        def open_session(self):
            # Stands in for the post-action archive: the phone already accepted the decision
            # and the store is uploading that profile's screenshots. Stop is requested from
            # here so run() must join a worker that is provably mid-archive.
            stop_event.set()
            time.sleep(1.0)          # outlasts the 0.2s flat bound mocked below
            archive_done.set()
        def next_profile(self):
            return None
        def out_of_profiles(self):
            return True
        def like(self, opener=None, item_index=None, *, model_item_index=None):
            pass
        def dislike(self):
            pass
        def close(self):
            driver_closed.set()

    monkeypatch.setattr(sup, "_worker_join_timeout_s", lambda cfg: 0.2)
    monkeypatch.setattr(sup, "_archive_join_grace_s", lambda cfg: 5.0)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    store = _ArchivingStore()
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _ArchivingDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)
    captured = {}

    try:
        sup.run(str(_write_cfg(tmp_path)),
                on_status=lambda s: captured.__setitem__("status", s), stop_event=stop_event)

        out = capsys.readouterr().out
        assert "is still archiving a decision that already landed on the phone" in out
        assert "did not stop within" not in out
        snap = captured["status"].snapshot()
        assert snap["phase"] == "stopped"
        assert snap["apps"]["hinge"]["state"] != "wedged"
    finally:
        assert driver_closed.wait(_LIVENESS_TIMEOUT_S)


# --- audit fix: an honest "stopping" tail between running and stopped ----------------------
# supervisor.py's shutdown `finally` used to publish phase="saving data" (and nothing for
# `stopping`) the INSTANT shutdown began -- before stop_event.set(), before any worker was
# even asked to notice it -- so the hub showed "saving data…" (and kept its green live-run
# cue up) for the whole worker-join window, while a worker could still be mid-swipe and
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
            # Stop pressed mid-run: armed from the worker's own first driver call, so it
            # cannot land on one of run()'s startup stop-checkpoints and abort before a
            # worker was ever launched (a wall-clock timer here loses that race on a slow
            # CI container). The wait below is only a failsafe against the fake driver
            # hanging forever if `release` somehow never fires -- the test always calls
            # release.set() explicitly once it has sampled the mid-shutdown status. It is
            # deliberately independent from the mocked `_worker_join_timeout_s=5.0` below,
            # which IS an input to the production code under test and must stay as authored.
            stop_event.set()
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
    stop_event = threading.Event()   # set by _BlockedUntilReleased.open_session, after launch

    run_thread = threading.Thread(
        target=lambda: sup.run(str(cfg_path), on_status=lambda s: captured.__setitem__("status", s),
                               stop_event=stop_event))
    run_thread.start()

    # The worker set stop from open_session and is still blocked there (release not set), so
    # once run() notices the stop and publishes 'stopping', its finally block is provably
    # stuck inside the join loop. Poll for that state instead of sleeping a fixed interval --
    # a wall-clock sleep is exactly what flaked on a slow CI container.
    deadline = time.monotonic() + _LIVENESS_TIMEOUT_S
    while time.monotonic() < deadline:
        status = captured.get("status")
        if status is not None and status.snapshot()["phase"] == "stopping":
            break
        time.sleep(0.02)
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
    # Do not schedule Stop from the test thread at an arbitrary wall-clock delay: startup has
    # several deliberate cancellation checks, so under suite load that old 0.1s timer could
    # fire before this worker was launched.  In that case it cleanly exits rather than wedges,
    # and this test ends up asserting a shutdown branch it never arranged to exercise.
    stop_event = threading.Event()
    driver_closed = threading.Event()

    class _WedgedDriver(DatingAppDriver):
        def open_session(self):
            # This is the precise point at which the worker is uninterruptibly wedged.  Request
            # shutdown from here so supervisor.run() must join this live worker for its full
            # mocked one-second bound.
            stop_event.set()
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
            driver_closed.set()

    monkeypatch.setattr(sup, "_worker_join_timeout_s", lambda cfg: 1.0)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    store = _FakeStore()
    cfg_path = _write_cfg(tmp_path)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    driver = _WedgedDriver()
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: driver)
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    _patch_no_adb(monkeypatch)

    try:
        sup.run(str(cfg_path), stop_event=stop_event)

        out = capsys.readouterr().out
        assert "did not stop within 1s" in out
        assert "30s" not in out
    finally:
        # Let the intentionally wedged daemon finish before this test returns -- even if the
        # assertion above fails.  Otherwise it can print into a later test's capture buffer,
        # making an unrelated assertion timing-sensitive.
        assert driver_closed.wait(_LIVENESS_TIMEOUT_S)
