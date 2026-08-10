"""Supervisor — one process runs all enabled apps concurrently.

Loads the shared resources once (ranker, BigQuery store, GLOBAL budget tracker),
launches one Worker per enabled app, and supervises them: clean shutdown on
Ctrl-C / SIGTERM (so it runs as an always-on service on any OS), flush + close
on exit. This replaces the single-app loop.
"""
from __future__ import annotations

import os
import signal
import threading
import uuid
from pathlib import Path

try:
    import fcntl                 # POSIX only; _AndroidDeviceLock degrades to a no-op without it
except ImportError:               # pragma: no cover — exercised only on Windows
    fcntl = None

from . import config as cfg_mod
from . import platforms
from .costing import CostTracker
from .drivers import make_driver
from .opener.opener import GeminiOpener
from .limits import RateLimiter
from .opener.service import OpenerService
from .ranker import make_store
from .ranker.decider import RankerDecider
from .ranker.model import PreferenceModel
from .runtime import Capabilities
from .status import RunStatus
from .vision.embed import Embedder
from .vision.quality import QualityFilter
from .worker import Worker

_STATUS_POLL_INTERVAL_S = 0.5

# The join timeout below run()'s shutdown `finally` gives each worker to notice stop_event and
# return on its own, before it's reported (and treated) as WEDGED -- see that block's own
# comments for what "wedged" costs (the store gets flushed/closed while a wedged worker may
# still be about to write to it).
#
# A worker can legitimately be blocked inside ONE already-in-flight opener HTTP request when
# Stop lands: every should_stop check in the opener call chain (OpenerService.maybe_opener,
# GeminiOpener.generate) runs BETWEEN attempts/models, never while a request is actually on the
# wire, so the worst case is bounded by cfg.opener.request_timeout_s, not by max_attempts or the
# model cascade length (see opener.py's/service.py's own should_stop docstrings for why that
# bound holds). The old flat _WORKER_JOIN_TIMEOUT_S = 30.0 predates that cancellation work and
# is now too short whenever openers are enabled with the shipped 90s request_timeout_s: a
# perfectly healthy worker riding out that one call gets misreported as wedged.
#
# _WORKER_JOIN_TIMEOUT_MARGIN_S is headroom for the REST of a worker's own shutdown path after
# that one call returns -- closing the driver, capturing a failure snapshot, the loop's own
# finally -- so a worker that finishes at (or a moment after) the request timeout doesn't get
# flagged wedged by its own ordinary cleanup work.
#
# Arithmetic at the shipped default (opener.request_timeout_s=90s):
#   90.0 (one in-flight opener call) + 15.0 (shutdown-path margin) = 105.0s
# -- comfortably more than the stale 30.0s constant, and still a bounded wait, not "forever".
#
# When openers are disabled (cfg.opener.enabled=False -> OpenerService(client=None, ...), or no
# client configured), no opener HTTP call can ever be in flight, so there's nothing analogous to
# ride out -- _WORKER_JOIN_TIMEOUT_FLOOR_S (the ORIGINAL flat constant, unchanged) applies
# instead, keeping those runs exactly as responsive to a genuinely wedged worker as before.
_WORKER_JOIN_TIMEOUT_FLOOR_S = 30.0
_WORKER_JOIN_TIMEOUT_MARGIN_S = 15.0


def _worker_join_timeout_s(cfg) -> float:
    """How long run()'s shutdown gives each worker to notice stop_event before it's reported
    wedged -- see the constants above for the arithmetic. A function of cfg (not a module
    constant) because the right bound depends on whether THIS run's openers are enabled and, if
    so, how long a single opener call is allowed to run -- both are per-config, not fixed."""
    if cfg.opener.enabled:
        return cfg.opener.request_timeout_s + _WORKER_JOIN_TIMEOUT_MARGIN_S
    return _WORKER_JOIN_TIMEOUT_FLOOR_S


def _android_app(enabled_apps: list[str]) -> str | None:
    """The (at most one — platforms.check_runnable enforces this) enabled Android-kind app,
    or None. Shared by the ADB preflight, the Capabilities probe, and the device lock so
    all three agree on which app's config block to read."""
    return next((a for a in enabled_apps if platforms.get(a).kind == platforms.KIND_ANDROID), None)


def _android_lock_path(cfg, app: str) -> Path:
    """ONE lock file for "the Android phone", deliberately not keyed by serial.

    Keying it on the configured serial string looked more precise and was actually a hole:
    the same physical device is named two different ways depending on the app block. With
    `apps.hinge.serial: "33111JEHN04475"` and `apps.bumble.serial: ""` (blank = adb's
    "first available device", which with one phone plugged in IS that same Pixel), the two
    resolved to `.android-33111JEHN04475.lock` and `.android-default.lock` — two files, no
    mutual exclusion at all, in exactly the scenario the lock exists to prevent. Nothing
    kept the two config blocks' serials in sync, and a blank serial cannot be compared to
    an explicit one without asking adb.

    A single shared lock cannot have that failure mode. The cost is that it would also
    serialise two runs against two DIFFERENT phones — a configuration this project does not
    support (ops/HINGE-PIXEL-RUNBOOK.md is one Pixel throughout). Between "occasionally too
    conservative in an unsupported setup" and "silently lets two processes fight over the
    one real phone", the conservative failure is the right one.

    `app` and `cfg` stay in the signature so a future multi-device setup can reintroduce
    per-device keying deliberately — with the serial actually RESOLVED through adb, not
    compared as a raw config string.
    """
    return Path(cfg.data_dir) / ".android-device.lock"


class _AndroidDeviceLock:
    """Advisory cross-process lock so two Android runs (Hinge, Bumble-once-calibrated, or
    two copies of the same one) can never overlap on the one physical Pixel — Android shows
    a single app in the foreground and `adb exec-out screencap`/the UHID touchscreen both
    act on whatever currently holds it. platforms.check_runnable() already stops two Android
    platforms being requested in the SAME run; this covers the case check_runnable can't see:
    two separate processes (e.g. the hub plus a manually launched CLI run).

    Uses flock(2) on a lockfile keyed by adb serial under paths.data_dir. flock ties the
    lock to this process's open file descriptor, so a crash or `kill -9` releases it
    automatically when the fd closes — no separate cleanup path needed, and no stale lock
    left behind for the next run to trip over.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None:
            return   # non-POSIX (Windows): no flock -- best-effort; check_runnable is the
                      # remaining guard against overlap within a single process/hub.
        fh = open(self.path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.seek(0)
            holder = fh.read().strip() or "an unknown process"
            fh.close()
            raise RuntimeError(
                f"Android device is already in use by another Operation Love run (lock held "
                f"by pid {holder}, {self.path}). Android shows one app in the foreground at "
                "a time, so two Android runs can never share the phone -- stop that run "
                "first.") from None
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            self._fh.close()
        except OSError:
            pass
        self._fh = None


def _resolve_run_cap(config_cap: int | None, override: int | None) -> int | None:
    """Per-run override of the max-swipes-per-run cap (set from the hub, auto mode).

    override is None -> no override, use the config value.
    override == 0    -> UNLIMITED for this run (no per-run cap).
    override > 0     -> cap this run at that many swipes.
    No cap is set by default in config.yaml. A positive hub/CLI override can add a
    temporary per-run cap; zero can explicitly clear a configured per-run cap.
    """
    if override is None:
        return config_cap
    return override or None


def _stop_requested(stop_event: threading.Event | None) -> bool:
    return stop_event is not None and stop_event.is_set()


def _abort_startup(run_id: str, status: RunStatus, cfg, store=None) -> None:
    """Stop was requested during startup, before any worker was launched. Startup (store
    setup, ranker training, embedder/quality warmup) can take many seconds with no other
    interrupt point (H-10), so honour Stop here too: publish a clean 'stopped' status
    (not a stuck 'starting'/'live') and close whatever store handle was already opened —
    nothing has been swiped yet, so there's nothing worth keeping it open for."""
    print(f"Run {run_id}: stop requested during startup; aborting before launching workers.")
    if store is not None:
        try:
            store.flush()
            store.close()
        except Exception as exc:  # noqa: BLE001 — best-effort; nothing was buffered yet
            print(f"Run {run_id}: warning closing store during startup abort: {exc}")
    for app in cfg.enabled_apps:
        status.set_app(app, state="stopped")
    status.set_global(running=False, phase="stopped")


def run(config_path: str = "config.yaml", *, stop_event: threading.Event | None = None,
        on_status=None, on_store=None, mode: str | None = None, enabled_apps=None,
        max_per_run: int | None = None) -> None:
    from ._warnings import configure_warnings
    configure_warnings()

    cfg = cfg_mod.load(config_path)
    if mode:                                  # hub/CLI override of config.yaml
        cfg.mode = mode
    if enabled_apps:
        cfg.enabled_apps = list(enabled_apps)

    # Hard guard, ahead of everything else (including validate()'s heavier checks, which
    # would also catch this but with less specific ordering): reject an unrunnable platform
    # selection before any driver is constructed, any browser launches, or any tap reaches
    # the phone. This is what stops two Android apps from ever being started together and
    # stops an uncalibrated/unavailable platform from ever getting this far. HubState.start()
    # applies the same check before it even spins up the run thread — this is the backstop
    # for callers that invoke supervisor.run() directly (CLI, tests) without going through it.
    unrunnable = platforms.check_runnable(cfg.enabled_apps)
    if unrunnable:
        raise ValueError(unrunnable)   # same exception type cfg_mod.validate() raises for this

    cfg_mod.validate(cfg)
    # Gemini uses the stdlib REST transport, so there is no SDK capability gate. Check
    # credentials here, before status/store/model setup, to avoid an expensive startup
    # followed by an inevitable provider failure.
    if (cfg.opener.enabled and cfg.opener.provider == "gemini"
            and not os.environ.get("GEMINI_API_KEY")):
        raise RuntimeError("GEMINI_API_KEY is required when opener.provider is 'gemini'")

    # Constructed here -- right after the key-presence check above, before make_store()'s
    # slow BigQuery/embedder warmup and before any device/ADB setup -- and preflighted
    # against the real ListModels endpoint (unless opener.preflight is off). Without this,
    # a valid-looking-but-wrong key or a typo'd model id used to survive all the way past a
    # full slow startup before failing. Note preflight narrows but does not eliminate this:
    # ListModels can list a model that still 404s the instant generateContent is actually
    # called (see GeminiOpener.preflight's docstring) -- but that residual case is no longer
    # fatal to the whole run either way, since GeminiOpener.generate() retires just the
    # offending model and cascades to the next configured one instead of failing identically
    # on every remaining profile. Reused unchanged below; never constructed twice.
    opener_client = None
    if cfg.opener.enabled and cfg.opener.provider == "gemini":
        opener_client = GeminiOpener(cfg.opener.effective_models, cfg.opener.max_tokens,
                                     cfg.opener.request_timeout_s,
                                     api_key=os.environ.get("GEMINI_API_KEY"),
                                     thinking=cfg.opener.thinking)
        if cfg.opener.preflight:
            opener_client.preflight()   # RuntimeError propagates as-is, aborting the run
        else:
            print("Gemini opener: opener.preflight is false -- configured model ids are "
                  "UNVALIDATED; a typo or an invalid key will only surface as a failure "
                  "once a real profile is processed.")

    run_id = uuid.uuid4().hex[:12]

    # Create + publish status up front (before the slow store/model setup) so the
    # hub shows a live "phase" immediately rather than appearing to hang on Start.
    status = RunStatus(run_id, cfg.enabled_apps, min_labels=cfg.ranker.min_labels_to_engage,
                       mode=cfg.mode, budget_cap=cfg.budget.run_budget_usd)
    if on_status:
        on_status(status)

    # At most one enabled app is Android-kind (check_runnable guarantees it above).
    android_app = _android_app(cfg.enabled_apps)

    # Pass the configured adb path: the android_driver capability resolves `adb` via PATH, so a
    # machine that sets apps.<app>.adb_path (adb not on PATH) would otherwise get a false
    # "not installed" warning even though the driver and the preflight both honour it.
    android_adb_path = ((cfg.apps or {}).get(android_app, {}) or {}).get("adb_path") if android_app else None
    caps = Capabilities.detect(android_adb_path=android_adb_path)
    print(caps.banner())
    missing_cloud = caps.missing("bigquery", "cloud_storage") if cfg.storage.backend == "bigquery" else []
    if missing_cloud:
        raise SystemExit("Storage.backend=bigquery but cloud storage dependencies are missing "
                         f"({', '.join(missing_cloud)}). Install `pip install -e '.[bq]'`, "
                         "or set storage.backend: sqlite.")

    missing_ml = caps.missing("arcface", "clip")
    if missing_ml:
        # Equally, deterministically fatal as missing_cloud above -- there's no defer path:
        # every profile embed calls Embedder._ensure(), which needs both libraries, so the
        # very first profile embed would fail regardless of mode. Pre-fix this only printed
        # a "Degrade" warning and let the run continue past make_store()'s BigQuery/label-
        # load work, past the ADB check, past the device lock, and into a live
        # driver.open_session() -- capturing a REAL profile off the live phone before dying
        # on that very first embed. (Embedder.warmup() used to independently swallow the
        # same class of failure too -- see its docstring in vision/embed.py, now fixed to
        # match QualityFilter.warmup()'s fail-loud contract.) Hard-gate it here, before any
        # of that, exactly like missing_cloud.
        raise SystemExit(
            f"ML extra not installed ({', '.join(missing_ml)} missing) -> ranking unavailable. "
            "There is no defer path: every profile embed would fail immediately. Install it "
            "with `pip install -e '.[ml]'` before starting a run."
        )

    # Cheapest possible "can this Android run even work" check, moved here -- BEFORE
    # make_store()'s BigQuery/label-load work and the ML warmup below, and BEFORE the
    # device lock is ever taken -- and turned into a hard, actionable failure instead of a
    # warning. Measured on a real supervisor.run(): pre-fix this ran AFTER both of those
    # (paying their combined cost) and only printed a WARNING, so a disconnected or
    # unauthorized phone cost a full BigQuery + ML warmup cycle before the operator found
    # out, and the device got marked in-use for a run that could never work -- the actual
    # hard gate lived deep inside AndroidDriver.open_session(), in the worker thread, AFTER
    # the lock was already held. See _android_adb_preflight's docstring for the
    # binary-missing vs no-device-connected distinction. android_app is None for a web-kind
    # selection, so a web platform is never touched by this check.
    if android_app is not None:
        _android_adb_preflight(android_app, cfg)

    if _stop_requested(stop_event):
        _abort_startup(run_id, status, cfg)
        return

    status.set_global(phase="loading saved data")     # BigQuery ensure-tables + label load
    store = make_store(cfg)
    if on_store:
        on_store(store)            # publish the live store so the hub eval reads live in-memory labels
    labels = store.load_labels()
    status.set_global(labels=len(labels))
    print(f"Store: backend={cfg.storage.backend} labels={len(labels)}  "
          f"apps={cfg.enabled_apps}")

    if _stop_requested(stop_event):
        _abort_startup(run_id, status, cfg, store)
        return

    effective_budget = cfg.budget.run_budget_usd
    if cfg.budget.day_budget_usd is not None:
        today_spend = getattr(store, "spend_today", lambda: 0.0)()
        remaining_today = max(0.0, cfg.budget.day_budget_usd - today_spend)
        print(f"Daily budget: ${cfg.budget.day_budget_usd:.2f}  "
              f"spent today: ${today_spend:.4f}  remaining: ${remaining_today:.4f}")
        if effective_budget is None:
            effective_budget = remaining_today
        else:
            effective_budget = min(effective_budget, remaining_today)
    status.set_global(budget_cap=effective_budget)   # hub must show the EFFECTIVE cap, not the raw config value
    tracker = CostTracker(cfg.budget.pricing, effective_budget)
    # opener_client for provider=="gemini" was already constructed (and preflighted) in the
    # early startup section above, before make_store() -- not rebuilt here. The legacy
    # Anthropic opener path has been removed entirely, and config.validate() rejects any
    # opener.provider other than "gemini", so by this point opener_client is either a
    # working GeminiOpener or None (opener.enabled=false) -- there is no other branch to
    # decide here.
    opener_service = OpenerService(opener_client, tracker, store, cfg.opener.style,
                                   max_attempts=cfg.opener.max_attempts)

    if _stop_requested(stop_event):
        _abort_startup(run_id, status, cfg, store)
        return

    status.set_global(phase="training ranker")
    model = PreferenceModel(min_labels=cfg.ranker.min_labels_to_engage,
                            threshold=cfg.ranker.like_threshold,
                            min_per_class=cfg.ranker.min_per_class)
    ready = model.train(labels)
    print(f"Ranker: labels={len(labels)} ready={ready} "
          f"(min={cfg.ranker.min_labels_to_engage}, threshold={cfg.ranker.like_threshold})")
    quality = QualityFilter(cfg.quality_filter.enabled, cfg.quality_filter.min_score,
                            cfg.quality_filter.metric)
    if not cfg.quality_filter.enabled:
        print("Quality filter: DISABLED — all photos are scored as passing quality threshold")
    embedder = Embedder()
    decider = RankerDecider(quality, embedder, model)

    if _stop_requested(stop_event):
        _abort_startup(run_id, status, cfg, store)
        return

    status.set_global(ranker_ready=ready, phase="loading ML models")
    print("Warming up embedder and quality filter (avoids first-profile delay and init races)…")
    embedder.warmup()
    quality.warmup()

    if _stop_requested(stop_event):
        _abort_startup(run_id, status, cfg, store)
        return

    stop_event = stop_event if stop_event is not None else threading.Event()
    _install_signal_handlers(stop_event)

    # Device lock: an advisory, cross-process flock so two Android runs can never overlap on
    # the one physical Pixel, even from two separate processes (check_runnable above only
    # guards within THIS run). Acquired before any worker starts; released in the finally
    # below regardless of how the run ends.
    device_lock = None

    # Launch construction+start lives INSIDE this try so the finally below (which stops,
    # joins, flushes and closes) always covers any worker already started — even if
    # make_driver() raises while building a LATER app (e.g. app #2's driver construction
    # fails after app #1's worker is already live). Without this, an already-started
    # worker/browser/ADB session would be orphaned and its buffered rows never flushed.
    workers = []
    try:
        if android_app is not None:
            device_lock = _AndroidDeviceLock(_android_lock_path(cfg, android_app))
            device_lock.acquire()      # raises RuntimeError naming the holder if contended

        for app in cfg.enabled_apps:
            app_cfg = (cfg.apps or {}).get(app, {}) or {}
            mode = app_cfg.get("mode", cfg.mode)                          # per-app override
            # Defended the same way config.validate() defends the equivalent read (and must
            # stay in lockstep with it — see _validate_limits' docstring): a bare `limits:`
            # (top-level or per-app) is valid YAML null, and validate() blesses it as an
            # empty override, not a crash. Without `or {}` on BOTH sides here, `**None`
            # raises TypeError deep in startup, after the slow ML warmup — validate() would
            # have given false confidence that this exact config was safe to run.
            lim = {**(cfg.limits or {}), **(app_cfg.get("limits", {}) or {})}
            run_cap = _resolve_run_cap(lim.get("max_per_run"), max_per_run)
            limiter = RateLimiter(run_cap, lim.get("max_per_day"),
                                  lim.get("max_likes_per_run"),
                                  target_like_ratio=lim.get("target_like_ratio"))
            driver = make_driver(app, cfg)
            w = Worker(app, driver, decider, opener_service, store, run_id, cfg.pacing,
                       stop_event, mode=mode, retrain_every=cfg.ranker.retrain_every,
                       limiter=limiter, status=status)
            print(f"{app.title()} worker mode={mode} limits={limiter.describe()}")
            workers.append(w)
            w.start()

        status.set_global(phase="live")
        # Break on stop_event too — not only when every worker has died. Otherwise a
        # worker that's slow to exit (mid-capture/embed) makes Ctrl-C busy-spin here
        # forever and never reach the flush/close below. With this, shutdown always
        # proceeds to join(timeout) -> flush -> close.
        while not stop_event.is_set() and any(w.is_alive() for w in workers):
            status.set_global(budget_spent=tracker.run_spend_usd, openers=tracker.calls)
            stop_event.wait(_STATUS_POLL_INTERVAL_S)
    finally:
        status.set_global(phase="saving data", budget_spent=tracker.run_spend_usd, openers=tracker.calls)
        stop_event.set()
        # A worker still alive after its join timeout is WEDGED, not stopped: it may write
        # to the store DURING or AFTER the flush/close below, so a clean flush here is not
        # an unqualified success. Detect it (without waiting any longer — a wedged worker
        # must not block quit) so the summary and status can say so honestly.
        #
        # Computed from cfg (see _worker_join_timeout_s) rather than a flat constant: a worker
        # can legitimately still be riding out ONE in-flight opener request when stop_event was
        # set, bounded by cfg.opener.request_timeout_s, not by some fixed guess.
        join_timeout_s = _worker_join_timeout_s(cfg)
        for w in workers:
            w.join(timeout=join_timeout_s)
            if w.is_alive():
                # Proceeding anyway (below) rather than blocking forever: the worker's
                # own stop_event is set, but it's still stuck mid-capture/embed/API-call.
                # It may call store.add_label()/record_decision() after store.close()
                # runs — a store write racing a closed store is the tradeoff for not
                # hanging shutdown indefinitely on one wedged app.
                print(f"Supervisor: worker '{w.app}' did not stop within "
                      f"{join_timeout_s:.0f}s; proceeding to save without it "
                      "(it may still be running in the background).")
        wedged = [w for w in workers if w.is_alive()]
        if device_lock is not None:
            # Released once workers are joined (or accepted as wedged, same tradeoff as
            # above) — placed here, ahead of the store flush/close and any `raise save_err`
            # below, so it unconditionally runs exactly once whenever this finally block is
            # reached, however run() exits.
            device_lock.release()
        # Capture every terminal reason atomically BEFORE the transient "saving" stamp.
        # Error is not the only durable result: out_of_profiles and rate_limited are exactly
        # the explanations a person returning to the hub later needs to see.
        before_save = status.snapshot()["apps"]
        terminal_states = {
            app: before_save.get(app, {}).get("state")
            for app in cfg.enabled_apps
        }
        for app in cfg.enabled_apps:
            status.set_app(app, state="saving")
        save_err = None
        try:
            store.flush()                 # raises if any buffered insert was rejected
        except Exception as exc:  # noqa: BLE001 — report a clear save outcome, then re-raise
            save_err = exc
        # close() must run whether or not flush() raised above -- pre-fix, close() sat
        # inside the same try as flush(), so a flush() failure skipped it entirely and
        # (for SQLiteStore) left its sqlite3.Connection open. Benign -- GC reclaims it
        # eventually -- but sloppy in a shutdown path this file otherwise treats carefully.
        # A close() failure here is reported but must never be allowed to shadow save_err:
        # flush() is what determines whether buffered data actually made it out, and that's
        # the error this function reports/re-raises below, unchanged either way.
        try:
            store.close()
        except Exception as close_exc:  # noqa: BLE001 — logged, never replaces save_err
            print(f"Run {run_id}: warning closing store during shutdown: "
                  f"{type(close_exc).__name__}: {close_exc}")
        if save_err is not None:
            phase = "save_failed"
        elif wedged:
            phase = "wedged"
        else:
            phase = "stopped"
        status.set_global(running=False, phase=phase,
                          budget_spent=tracker.run_spend_usd, openers=tracker.calls)
        wedged_apps = {w.app for w in wedged}
        for app in cfg.enabled_apps:
            if save_err is not None:
                app_state = "error"
            elif app in wedged_apps:
                app_state = "wedged"
            else:
                prior = terminal_states.get(app)
                app_state = prior if prior in {
                    "error", "out_of_profiles", "rate_limited", "stopped"
                } else "stopped"
            status.set_app(app, state=app_state)
        tail = f"openers={tracker.calls} spend=${tracker.run_spend_usd:.4f}"
        if save_err is not None:
            print(f"Run {run_id}: ❌ SAVE FAILED to {cfg.storage.backend} "
                  f"({type(save_err).__name__}: {save_err}) — buffered data may be incomplete; {tail}")
            raise save_err
        saved = getattr(store, "saved_summary", lambda: "")()
        detail = f" [{saved}]" if saved else ""
        if wedged:
            names = ", ".join(w.app for w in wedged)
            print(f"Run {run_id}: saved to {cfg.storage.backend}{detail}, but {len(wedged)} "
                  f"worker(s) did not stop within 30s ({names}) — NOT an unqualified success, "
                  f"a late write from a wedged worker could still land after this save; {tail}")
        else:
            print(f"Run {run_id}: ✅ all data saved to {cfg.storage.backend}{detail}; {tail}")


def _android_adb_preflight(app: str, cfg) -> None:
    """Hard-gate early if the Android phone isn't visible to adb for `app`, before any of
    the expensive startup work (make_store()'s BigQuery ensure-tables + label load, the
    ArcFace/CLIP/quality-filter warmup) runs and before the device lock is ever taken.
    Generalised over any Android-kind platform (Hinge today, Bumble once calibrated — see
    platforms.py): both drive the same physical Pixel over host-side ADB, so whichever one
    is enabled needs the same early connectivity check. Never called for a web-kind
    platform (the caller only invokes this when android_app is not None), so a web
    selection is never blocked by an ADB check.

    Deliberately two DIFFERENT fatal outcomes, not one:

      * the adb BINARY itself is missing (FileNotFoundError) -- nothing can even be asked
        whether a phone is connected, so the fix is install-adb-or-set-adb_path, not
        connect-the-phone.
      * the binary runs fine but reports no device (or the configured serial isn't among
        what it reports) -- adb itself works, so the fix is a physical one: plug in the
        Pixel and authorize the RSA key (or fix apps.<app>.serial).

    Both are equally fatal -- an Android run cannot work in either case, exactly like the
    caps.missing("arcface", "clip") / missing_cloud gates above -- but they're worth
    telling apart because the operator's next action differs. This used to only print a
    WARNING and let the run continue regardless (the real hard gate lived deep inside
    AndroidDriver.open_session(), in the worker thread, after the device lock was already
    held) -- see supervisor.run()'s call site for the measured cost that let through.
    """
    import subprocess

    from .drivers.adb import parse_devices_output

    label = platforms.get(app).label
    app_cfg = (cfg.apps or {}).get(app, {}) or {}
    serial = (app_cfg.get("serial") or "").strip()
    adb = (app_cfg.get("adb_path") or "adb").strip() or "adb"
    try:
        result = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=5)
    except FileNotFoundError:
        raise SystemExit(
            f"{label} is enabled but `{adb}` was not found. Install Android platform-tools "
            f"(adb) and put it on PATH, or set apps.{app}.adb_path in config.yaml."
        ) from None
    except Exception as exc:  # noqa: BLE001 — e.g. a hung/misbehaving adb server; still fatal,
        # since there's no way to confirm a phone is reachable, and re-running this 5s
        # check is cheap next to the BigQuery/ML startup it would otherwise gate.
        raise SystemExit(f"{label} ADB preflight failed: {type(exc).__name__}: {exc}") from exc

    visible = parse_devices_output(result.stdout)   # X8: the ONE canonical parser
    if not visible:
        raise SystemExit(
            f"{label} is enabled but `adb devices` shows no connected device. Connect the "
            "Pixel 7a via USB and authorize the RSA key before starting a run."
        )
    if serial and serial not in visible:
        raise SystemExit(
            f"apps.{app}.serial={serial!r} not in `adb devices` output: {visible}. "
            f"Check config.yaml → apps.{app}.serial."
        )
    dev = serial if serial else visible[0]
    print(f"{label} ADB preflight OK: {dev} (device connected)")


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _stop(*_):
        print("\nSupervisor: shutdown requested; stopping workers...")
        stop_event.set()
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):  # not in main thread / unsupported on this OS
            pass
