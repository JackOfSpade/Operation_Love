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
from operation_love.config import OpenerCfg
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
        def like(self, opener=None, item_index=0):
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

    def __init__(self, client, tracker, store, style, max_attempts=5):
        self.client = client
        self.tracker = tracker
        self.store = store
        self.style = style
        self.max_attempts = max_attempts
        _SpyOpenerService.instances.append(self)


def test_opener_max_attempts_reaches_the_constructed_opener_service(monkeypatch, tmp_path):
    """cfg.opener.max_attempts must reach OpenerService's constructor as the keyword
    max_attempts -- this is the setting that controls how many times a rejected AI response
    is re-asked before the run stops (owner rule, 2026-08-10: no commentless likes). Uses a
    distinctive value (7) rather than the class default (5) so a mutation that drops the
    kwarg entirely -- silently falling back to OpenerService's own default -- cannot pass
    unnoticed."""
    cfg_text = _CONFIG.replace("opener:\n  enabled: false", "opener:\n  enabled: false\n  max_attempts: 7")
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
