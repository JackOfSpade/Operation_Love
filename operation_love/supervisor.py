"""Supervisor — one process runs all enabled apps concurrently.

Loads the shared resources once (ranker, BigQuery store, GLOBAL budget tracker),
launches one Worker per enabled app, and supervises them: clean shutdown on
Ctrl-C / SIGTERM (so it runs as an always-on service on any OS), flush + close
on exit. This replaces the single-app loop.
"""
from __future__ import annotations

import errno
import math
import os
import signal
import sys
import threading
import uuid
from contextlib import contextmanager
from numbers import Real
from pathlib import Path

try:
    import fcntl                 # POSIX advisory file locking
except ImportError:               # pragma: no cover — exercised only on Windows
    fcntl = None
try:
    import msvcrt                # Windows byte-range file locking
except ImportError:               # pragma: no cover — exercised on POSIX
    msvcrt = None

from . import config as cfg_mod
from . import platforms
from .costing import CostTracker
from .drivers import make_driver
from .opener.opener import GeminiOpener
from .limits import RateLimiter
from .opener.service import OpenerService
from .private_files import ensure_private_dir, open_private_rw
from .ranker import make_store
from .ranker.decider import RankerDecider
from .ranker.model import PreferenceModel
from .runtime import Capabilities
from .status import RunStatus
from .vision.embed import Embedder
from .vision.quality import QualityFilter
from .worker import Worker

_STATUS_POLL_INTERVAL_S = 0.5
_ANDROID_LOCK_ROOT = Path.home() / ".operation-love" / "locks"

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
_WEDGED_WORKER_STACK_MAX_FRAMES = 12


def _worker_join_timeout_s(cfg) -> float:
    """How long run()'s shutdown gives each worker to notice stop_event before it's reported
    wedged -- see the constants above for the arithmetic. A function of cfg (not a module
    constant) because the right bound depends on whether THIS run's openers are enabled and, if
    so, how long a single opener call is allowed to run -- both are per-config, not fixed."""
    if cfg.opener.enabled:
        return cfg.opener.request_timeout_s + _WORKER_JOIN_TIMEOUT_MARGIN_S
    return _WORKER_JOIN_TIMEOUT_FLOOR_S


def _wedged_worker_stack_lines(worker) -> list[str]:
    """Return a compact, locals-free snapshot of a live worker's Python stack.

    ``sys._current_frames()`` is an in-process memory lookup: it neither talks to the
    device nor asks the worker to cooperate, which matters precisely when the worker
    is stuck. Deliberately record only code metadata (basename, line number, function),
    never source text, arguments, or locals: a shutdown diagnostic must not leak profile
    data, API credentials, or an in-flight request body into stdout/bug reports.
    """
    app = getattr(worker, "app", "unknown")
    ident = getattr(worker, "ident", None)
    if ident is None:
        return [f"Supervisor: worker '{app}' stack unavailable (no thread identity)."]
    try:
        frame = sys._current_frames().get(ident)
    except BaseException:  # noqa: BLE001 -- diagnostics must never disrupt shutdown
        return [f"Supervisor: worker '{app}' stack unavailable."]
    if frame is None:
        return [f"Supervisor: worker '{app}' stack unavailable (thread frame not found)."]

    frames: list[tuple[str, int, str]] = []
    while frame is not None and len(frames) < _WEDGED_WORKER_STACK_MAX_FRAMES:
        code = frame.f_code
        # A basename preserves the useful file signal while avoiding an absolute local path.
        filename = os.path.basename(code.co_filename).replace("\n", "\\n").replace("\r", "\\r")
        function = code.co_name.replace("\n", "\\n").replace("\r", "\\r")
        frames.append((filename, frame.f_lineno, function))
        frame = frame.f_back

    omitted = " (older frames omitted)" if frame is not None else ""
    lines = [
        f"Supervisor: worker '{app}' Python stack at wedge "
        f"({len(frames)} frame(s), newest first{omitted}):"
    ]
    lines.extend(f"  {filename}:{lineno} in {function}" for filename, lineno, function in frames)
    return lines


def _android_app(enabled_apps: list[str]) -> str | None:
    """The (at most one — platforms.check_runnable enforces this) enabled Android-kind app,
    or None. Shared by the ADB preflight, the Capabilities probe, and the device lock so
    all three agree on which app's config block to read."""
    return next((a for a in enabled_apps if platforms.get(a).kind == platforms.KIND_ANDROID), None)


def _android_lock_path(cfg, app: str) -> Path:
    """ONE per-user lock file for "the Android phone", independent of config.

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

    The path must also be independent of ``paths.data_dir``.  That directory is an operator
    setting, so two otherwise valid config files can name different data roots; putting the
    lock below either root gives them different locks and silently defeats cross-process
    exclusion.  A fixed directory below the current user's home is stable across launchers,
    repositories, and configs while remaining writable without elevated privileges.

    ``app`` and ``cfg`` stay in the signature so a future multi-device setup can reintroduce
    per-device keying deliberately — with the serial actually RESOLVED through adb, not
    compared as a raw config string.  They intentionally do not influence today's path.
    """
    del cfg, app
    return _ANDROID_LOCK_ROOT / "android-device.lock"


def _lock_contention(exc: OSError) -> bool:
    return (isinstance(exc, BlockingIOError)
            or exc.errno in {errno.EACCES, errno.EAGAIN}
            or getattr(exc, "winerror", None) in {33, 36})


def _acquire_platform_file_lock(fh) -> str:
    """Acquire one non-blocking OS lock and return the backend used."""
    fh.seek(0)
    if fcntl is not None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return "fcntl"
    if msvcrt is not None:
        # msvcrt.locking locks from the current file position.  acquire() guarantees the
        # file contains at least one byte, and both lock and unlock always seek to byte 0.
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        return "msvcrt"
    raise RuntimeError(
        "Android device locking is unavailable: this platform has neither fcntl nor msvcrt")


def _release_platform_file_lock(fh, backend: str) -> None:
    fh.seek(0)
    if backend == "fcntl":
        if fcntl is None:  # defensive: backend availability cannot change during a real run
            raise RuntimeError("fcntl disappeared while releasing Android device lock")
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return
    if backend == "msvcrt":
        if msvcrt is None:  # defensive: see fcntl branch above
            raise RuntimeError("msvcrt disappeared while releasing Android device lock")
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        return
    raise RuntimeError(f"unknown Android device lock backend: {backend}")


class _AndroidDeviceLock:
    """Advisory cross-process lock so two Android runs (Hinge, Bumble-once-calibrated, or
    two copies of the same one) can never overlap on the one physical Pixel — Android shows
    a single app in the foreground and `adb exec-out screencap`/the UHID touchscreen both
    act on whatever currently holds it. platforms.check_runnable() already stops two Android
    platforms being requested in the SAME run; this covers the case check_runnable can't see:
    two separate processes (e.g. the hub plus a manually launched CLI run).

    Uses flock(2) on POSIX and msvcrt byte-range locking on Windows.  Both tie ownership to
    this process's open file handle, so a crash releases the OS lock automatically.  The
    small PID file deliberately remains in place: deleting a lock path during release creates
    a race where another process can lock the old inode while a third locks a newly-created
    one.  A stale PID is harmless because only the live OS lock decides ownership.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self._backend: str | None = None

    def acquire(self) -> None:
        if self._fh is not None:
            return
        ensure_private_dir(self.path.parent)
        fd = open_private_rw(self.path, parent=self.path.parent)
        fh = os.fdopen(fd, "r+b")
        # Windows cannot lock a zero-length range.  Establish one byte before competing for
        # byte 0; simultaneous creators may both write this placeholder, which is harmless.
        fh.seek(0, os.SEEK_END)
        if fh.tell() == 0:
            fh.write(b"0")
            fh.flush()
        try:
            backend = _acquire_platform_file_lock(fh)
        except OSError as exc:
            fh.seek(0)
            holder = fh.read(128).decode("ascii", errors="replace").strip() or "an unknown process"
            fh.close()
            if not _lock_contention(exc):
                raise RuntimeError(
                    f"Could not acquire Android device lock at {self.path}: {exc}") from exc
            raise RuntimeError(
                f"Android device is already in use by another Operation Love run (lock held "
                f"by pid {holder}, {self.path}). Android shows one app in the foreground at "
                "a time, so two Android runs can never share the phone -- stop that run "
                "first.") from None
        except BaseException:
            fh.close()
            raise
        try:
            fh.seek(0)
            fh.write(str(os.getpid()).encode("ascii"))
            fh.truncate()
            fh.flush()
        except BaseException:
            try:
                _release_platform_file_lock(fh, backend)
            finally:
                fh.close()
            raise
        self._fh = fh
        self._backend = backend

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            _release_platform_file_lock(self._fh, self._backend or "")
        except (OSError, RuntimeError):
            pass
        try:
            self._fh.close()
        except OSError:
            pass
        self._fh = None
        self._backend = None


@contextmanager
def exclusive_android_device(cfg, app: str = "hinge"):
    """Hold the SAME cross-process Android lock a run holds, for the duration of the block.

    ADDED 2026-08-22. The lock existed only inside run(), so every offline tool that drives the
    phone -- the calibration capture, the video-bound campaigns, the scroll/inspect probes --
    competed with a hub run on the honour system. The failure it invites is not hypothetical:
    run `a01fbcd1e9a0`'s false Malaika Pass came from two drivers on one phone, and it took a
    hash-bound retraction plan to undo. A capture campaign is exactly the case the honour system
    loses, because it runs for many minutes beside an idle hub the owner may Start at any moment.

    Exported rather than reimplemented so there is one lock file, one contention message, and one
    place where the policy can change. Tools call it around their device session; `run()` keeps
    its own acquire/release because it must hand the lock to a wedged-worker reaper, which a
    context manager cannot express.
    """
    lock = _AndroidDeviceLock(_android_lock_path(cfg, app))
    lock.acquire()      # raises RuntimeError naming the holding pid if contended
    try:
        yield lock
    finally:
        lock.release()


_RETAINED_DEVICE_LOCKS: set[_AndroidDeviceLock] = set()
_RETAINED_DEVICE_LOCKS_GUARD = threading.Lock()


def _retain_device_lock_until_worker_exits(
        device_lock: _AndroidDeviceLock, worker: Worker) -> threading.Thread | None:
    """Transfer a wedged Android worker's lock to a daemon reaper.

    Releasing merely because the bounded shutdown wait expired reopens the exact overlapping
    input race the lock prevents: the old worker is still live and may still touch the phone.
    The process-wide set also keeps the lock alive if thread creation itself fails.
    """
    with _RETAINED_DEVICE_LOCKS_GUARD:
        _RETAINED_DEVICE_LOCKS.add(device_lock)

    def reap() -> None:
        try:
            worker.join()
        finally:
            device_lock.release()
            with _RETAINED_DEVICE_LOCKS_GUARD:
                _RETAINED_DEVICE_LOCKS.discard(device_lock)

    reaper = threading.Thread(
        target=reap,
        name=f"android-lock-reaper-{worker.app}",
        daemon=True,
    )
    try:
        reaper.start()
    except Exception as exc:  # noqa: BLE001 -- shutdown/store cleanup must still run
        # The global strong reference deliberately keeps the lock held for the remainder of
        # this process. Releasing here would let a second run overlap the still-live worker.
        print(
            "Supervisor: CRITICAL — could not start the Android-lock reaper "
            f"({type(exc).__name__}: {exc}); retaining the device lock for this process's "
            "lifetime. Restart Operation Love only after the wedged worker/device action ends.")
        return None
    return reaper


MAX_PER_RUN_OVERRIDE = 1_000_000


def normalize_max_per_run(value: object) -> int | None:
    """Validate a direct or Hub per-run cap without truthy/coercion surprises.

    ``None`` delegates to config, zero explicitly clears a configured cap for this run,
    and a positive integer supplies the cap. Keep this invariant in the supervisor so
    non-HTTP callers cannot bypass the Hub's input validation.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_per_run must be null, 0, or a positive integer")
    if value > MAX_PER_RUN_OVERRIDE:
        raise ValueError(f"max_per_run must not exceed {MAX_PER_RUN_OVERRIDE}")
    return value


def _resolve_run_cap(config_cap: int | None, override: int | None) -> int | None:
    """Per-run override of the max-swipes-per-run cap (set from the hub, auto mode).

    override is None -> no override, use the config value.
    override == 0    -> UNLIMITED for this run (no per-run cap).
    override > 0     -> cap this run at that many swipes.
    No cap is set by default in config.yaml. A positive hub/CLI override can add a
    temporary per-run cap; zero can explicitly clear a configured per-run cap.
    """
    override = normalize_max_per_run(override)
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
        except Exception as exc:  # noqa: BLE001 — best-effort; nothing was buffered yet
            print(f"Run {run_id}: warning flushing store during startup abort: {exc}")
        try:
            store.close()
        except Exception as exc:  # noqa: BLE001 — closing must run even when flush failed
            print(f"Run {run_id}: warning closing store during startup abort: {exc}")
    for app in cfg.enabled_apps:
        status.set_app(app, state="stopped")
    status.set_global(running=False, phase="stopped")


def _validated_daily_spend(value: object) -> float:
    """Validate the persisted day-budget seed before it can loosen a money cap."""
    try:
        finite = math.isfinite(value)
    except (TypeError, OverflowError):
        finite = False
    if isinstance(value, bool) or not isinstance(value, Real) or not finite or value < 0:
        raise ValueError(
            "Store spend_today() must return a finite nonnegative number; the daily budget "
            "cannot be enforced from malformed persisted spend")
    return float(value)


def _close_store_after_startup_failure(store, run_id: str) -> None:
    """Release a store opened during startup without ever replacing the original failure."""
    try:
        store.close()
    except BaseException as close_exc:  # noqa: BLE001 — original startup error owns propagation
        print(f"Run {run_id}: warning closing store after startup failure: "
              f"{type(close_exc).__name__}: {close_exc}")


def load_effective_config(config_path: str = "config.yaml", *, mode: str | None = None,
                          enabled_apps=None) -> cfg_mod.Config:
    """Load the exact config a run would use and enforce every start-time gate.

    Hub starts need the same answer synchronously, before they create a worker thread or
    timed-stop timer, while direct CLI callers need the identical check inside ``run``.
    ``None`` alone means "use config.yaml". Any explicit override is applied, including an
    empty app list or empty mode string, so invalid API/CLI input fails closed instead of
    silently falling back to a potentially live configured run. The Hub separately gives an
    explicit empty app selection a friendlier message before calling this helper.
    """
    cfg = cfg_mod.load(config_path)
    if mode is not None:                      # explicit hub/CLI override of config.yaml
        cfg.mode = mode
    if enabled_apps is not None:
        cfg.enabled_apps = enabled_apps

    # Shape-check explicit overrides before registry lookups. In particular, a direct
    # ``enabled_apps=[{}]`` must produce the config contract's clean ValueError instead of
    # reaching set/dict membership in the availability gate as an unhashable TypeError.
    cfg_mod._validate_config_shape(cfg)

    # Training is a mode with a mandatory Hub decision boundary, not an advisory spelling
    # of AUTO. Per-app overrides normally win, but allowing one to turn this explicit request
    # back into ``auto`` would silently use ranker decisions without the promised checkpoint. This is
    # intentionally after the shape check: malformed caller input must retain Config's clean
    # ValueError rather than leaking a mapping/list TypeError from this special-case scan.
    if cfg.mode == "training":
        overridden = [
            app for app in cfg.enabled_apps
            if ((cfg.apps or {}).get(app, {}) or {}).get("mode") not in (None, "training")
        ]
        if overridden:
            raise ValueError(
                "Training cannot run while apps override its mode for "
                f"{sorted(overridden)}; remove the per-app mode override or set it to "
                "training")

    # Validate before consulting dynamic platform readiness. Config validation installs the
    # still-photo licence and verifies Hinge AUTO's exact release artifact; checking the
    # registry first in a fresh process would see neither installed and reject a valid AUTO
    # config before its gate had a chance to prove itself. This remains before any Worker,
    # driver, touch transport, or external store is constructed.
    cfg_mod.validate(cfg)

    requested_modes = {
        app: (((cfg.apps or {}).get(app, {}) or {}).get("mode", cfg.mode))
        for app in cfg.enabled_apps
    }
    # Registry readiness is intentionally after full config validation but still before all
    # construction: an uncalibrated target such as Bumble is refused with no driver/touch.
    unrunnable = platforms.check_runnable(cfg.enabled_apps, modes=requested_modes)
    if unrunnable:
        raise ValueError(unrunnable)
    return cfg


def run(config_path: str = "config.yaml", *, stop_event: threading.Event | None = None,
        on_status=None, on_store=None, on_opener_service=None, mode: str | None = None,
        enabled_apps=None, max_per_run: int | None = None, on_worker=None) -> None:
    """Run every enabled app until stop_event is set (or the queue/rate-limit/error runs out
    the run on its own), then flush + close the store.

    on_status/on_store/on_opener_service are hub-only callbacks (unused by the plain CLI
    path): each is invoked exactly once, immediately after the object it hands over is
    constructed, so HubState can capture a live reference for the running hub page and the
    one-click bug report to read from a different thread while the run is in progress —
    on_status gets the RunStatus (phase/labels/per-app state), on_store gets the live label
    store (so the hub's model-quality card reads in-memory labels, not a lagging committed
    read), and on_opener_service gets the OpenerService (so a bug report can show what each
    opener actually said this run, not just RunStatus's raw `openers` call count — see the
    call site below for why that distinction matters).
    """
    from ._warnings import configure_warnings
    configure_warnings()

    # Validate the public override before loading providers, stores, models, or device state.
    # _resolve_run_cap repeats the check as a defense-in-depth backstop for direct callers.
    max_per_run = normalize_max_per_run(max_per_run)

    # Shared with HubState.start(): the Hub rejects a bad effective config before it creates
    # background state, and this remains the backstop for direct CLI/test callers.
    cfg = load_effective_config(config_path, mode=mode, enabled_apps=enabled_apps)
    effective_modes = {
        app: ((cfg.apps or {}).get(app, {}) or {}).get("mode", cfg.mode)
        for app in cfg.enabled_apps
    }
    if "training" in effective_modes.values() and on_worker is None:
        raise ValueError(
            "Training requires the local Hub decision bridge; start it from the Hub rather "
            "than a direct CLI/supervisor run")

    # Publish a cancellable startup state before any credential, provider, capability, or
    # device preflight.  A Stop can arrive immediately after the Hub starts this background
    # thread; doing a Gemini ListModels request or invoking adb after that point is needless
    # latency at best and can keep the shutdown UI stuck in "starting" for a network timeout.
    # Config validation above still runs first so malformed input remains a clear refusal rather
    # than looking like a successful cancelled run.
    run_id = uuid.uuid4().hex[:12]
    status = RunStatus(run_id, cfg.enabled_apps, min_labels=cfg.ranker.min_labels_to_engage,
                       mode=cfg.mode, budget_cap=cfg.budget.run_budget_usd)
    # AppStatus defaults to training so legacy/direct callers have a safe shape, but the Hub
    # reads this snapshot before any Worker exists. Publish the already-resolved per-app mode
    # first: otherwise an AUTO app briefly renders as Training until its worker thread starts.
    for app, app_mode in effective_modes.items():
        status.set_app(app, mode=app_mode)
    if on_status:
        on_status(status)
    if _stop_requested(stop_event):
        _abort_startup(run_id, status, cfg)
        return

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

    def _prepare_runtime():
        if on_store:
            # Publish the live store so the hub eval reads live in-memory labels.
            on_store(store)
        labels = store.load_labels()
        status.set_global(labels=len(labels))
        print(f"Store: backend={cfg.storage.backend} labels={len(labels)}  "
              f"apps={cfg.enabled_apps}")

        if _stop_requested(stop_event):
            _abort_startup(run_id, status, cfg, store)
            return None

        effective_budget = cfg.budget.run_budget_usd
        if cfg.budget.day_budget_usd is not None:
            # Daily spend is a safety input, not optional analytics. A missing method, failed
            # query, or malformed persisted scalar aborts while the startup ownership guard
            # below still owns (and closes) the store.
            today_spend = _validated_daily_spend(store.spend_today())
            remaining_today = max(0.0, cfg.budget.day_budget_usd - today_spend)
            print(f"Daily budget: ${cfg.budget.day_budget_usd:.2f}  "
                  f"spent today: ${today_spend:.4f}  remaining: ${remaining_today:.4f}")
            if effective_budget is None:
                effective_budget = remaining_today
            else:
                effective_budget = min(effective_budget, remaining_today)
        # The hub must show the EFFECTIVE cap, not the raw config value.
        status.set_global(budget_cap=effective_budget)
        tracker = CostTracker(cfg.budget.pricing, effective_budget)
        # opener_client was constructed/preflighted before the store. It is either a working
        # Gemini client or None (opener.enabled=false); no legacy provider branch remains.
        opener_service = OpenerService(
            opener_client, tracker, store, cfg.opener.style,
            max_attempts=cfg.opener.max_attempts,
            advisory_max_attempts=cfg.opener.advisory_max_attempts,
            advisory_deadline_s=cfg.opener.advisory_deadline_s)
        if on_opener_service:
            # Publish the same live service workers receive, for run-scoped bug telemetry.
            on_opener_service(opener_service)

        if _stop_requested(stop_event):
            _abort_startup(run_id, status, cfg, store)
            return None

        status.set_global(phase="training ranker")
        model = PreferenceModel(
            min_labels=cfg.ranker.min_labels_to_engage,
            threshold=cfg.ranker.like_threshold,
            min_per_class=cfg.ranker.min_per_class)
        ready = model.train(labels)
        print(f"Ranker: labels={len(labels)} ready={ready} "
              f"(min={cfg.ranker.min_labels_to_engage}, "
              f"threshold={cfg.ranker.like_threshold})")
        quality = QualityFilter(
            cfg.quality_filter.enabled, cfg.quality_filter.min_score,
            cfg.quality_filter.metric)
        if not cfg.quality_filter.enabled:
            print("Quality filter: DISABLED — all photos are scored as passing quality threshold")
        embedder = Embedder()
        decider = RankerDecider(quality, embedder, model)

        if _stop_requested(stop_event):
            _abort_startup(run_id, status, cfg, store)
            return None

        status.set_global(ranker_ready=ready, phase="loading ML models")
        print("Warming up embedder and quality filter "
              "(avoids first-profile delay and init races)…")
        embedder.warmup()
        quality.warmup()

        if _stop_requested(stop_event):
            _abort_startup(run_id, status, cfg, store)
            return None

        active_stop_event = stop_event if stop_event is not None else threading.Event()
        _install_signal_handlers(active_stop_event)
        return tracker, opener_service, decider, active_stop_event

    try:
        prepared = _prepare_runtime()
    except BaseException:
        _close_store_after_startup_failure(store, run_id)
        raise
    if prepared is None:  # stop was requested and _abort_startup already closed the store
        return
    tracker, opener_service, decider, stop_event = prepared

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
            try:
                # Hub binding is part of launching a worker, not a precondition outside its
                # cleanup boundary.  A bridge may reject a replacement while registering; the
                # driver already exists then and must be released exactly like a failed
                # Thread.start().
                if on_worker:
                    on_worker(w)  # hub-only binding; must happen before this Worker thread starts
                print(f"{app.title()} worker mode={mode} limits={limiter.describe()}")
                w.start()
            except BaseException:
                # A Thread cannot be joined until start() succeeds, and a Hub binding can fail
                # after the driver was constructed but before start().  In either case keeping
                # this worker in ``workers`` would make shutdown join an unstarted thread,
                # masking the actual launch error and skipping the store cleanup below.
                bridge = getattr(w, "training_action_bridge", None)
                if bridge is not None:
                    try:
                        bridge.unregister(w)
                    except BaseException as bridge_exc:  # noqa: BLE001 — preserve start error
                        print(
                            f"Run {run_id}: warning unregistering {app} worker after launch "
                            f"failed: {type(bridge_exc).__name__}: {bridge_exc}")
                try:
                    driver.close()
                except BaseException as close_exc:  # noqa: BLE001 — never mask start failure
                    print(
                        f"Run {run_id}: warning closing {app} driver after worker launch "
                        f"failed: {type(close_exc).__name__}: {close_exc}")
                raise
            workers.append(w)

        status.set_global(phase="live")
        # Break on stop_event too — not only when every worker has died. Otherwise a
        # worker that's slow to exit (mid-capture/embed) makes Ctrl-C busy-spin here
        # forever and never reach the flush/close below. With this, shutdown always
        # proceeds to join(timeout) -> flush -> close.
        while not stop_event.is_set() and any(w.is_alive() for w in workers):
            status.set_global(budget_spent=tracker.run_spend_usd, openers=tracker.calls)
            stop_event.wait(_STATUS_POLL_INTERVAL_S)
    finally:
        # Publish the "stopping" tail -- not "saving data" -- the instant shutdown begins,
        # at the same moment stop_event.set() runs (right before it, same statement group,
        # so no reader can observe stop_event set while status still claims "live"). Audit
        # fix: this used to stamp phase="saving data" here, ~40 lines before the actual
        # store.flush() call below, so the hub showed "saving data…" for the ENTIRE
        # worker-join window (up to join_timeout_s -- 105s with the shipped opener config)
        # while a worker could still be mid-profile-read -- and kept rendering a live run cue
        # even though no new decision should begin after stop_event is set. See status.py's
        # `stopping` docstring.
        status.set_global(stopping=True, phase="stopping",
                          budget_spent=tracker.run_spend_usd, openers=tracker.calls)
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
        if workers:
            # Named + bounded up front, before any worker has had a chance to report back,
            # so an operator watching the terminal or the hub's live-log panel (which tees
            # this same stdout -- see bugreport.install_log_capture) sees WHY nothing else
            # happens for a while, instead of the silent gap this whole fix addresses.
            print(f"Supervisor: stopping — waiting up to {join_timeout_s:.0f}s for "
                  f"{len(workers)} worker(s) to finish what they are doing…")
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
                # Capture its precise Python location while it is still known alive. These
                # lines flow through the hub's stdout tee into bugreport's Recent logs.
                for line in _wedged_worker_stack_lines(w):
                    print(line)
        wedged = [w for w in workers if w.is_alive()]
        if device_lock is not None:
            wedged_android = next(
                (worker for worker in wedged if worker.app == android_app), None)
            if wedged_android is None:
                device_lock.release()
            else:
                reaper = _retain_device_lock_until_worker_exits(device_lock, wedged_android)
                if reaper is not None:
                    print(
                        f"Supervisor: retaining Android device lock until wedged worker "
                        f"'{wedged_android.app}' actually exits; a daemon reaper will release it.")
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
        # Only NOW is "saving data" true: every worker has been joined or accepted as
        # wedged (both branches above have already run), so nothing still mid-swipe can
        # surprise the flush below. This replaces the old stamp at the top of this block --
        # see this function's "stopping" comment above for what was wrong with that.
        status.set_global(phase="saving data")
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
            # Preserve the earlier flush failure when present, but a close-only failure is also
            # a failed final persistence boundary (for example, an image archive that completed
            # after the first flush and failed during close's drain). Reporting success in that
            # case would falsely claim every buffered row made it to the system of record.
            if save_err is None:
                save_err = close_exc
        if save_err is not None:
            phase = "save_failed"
        elif wedged:
            phase = "wedged"
        else:
            phase = "stopped"
        status.set_global(running=False, phase=phase, stopping=False,
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
                    "error", "out_of_profiles", "rate_limited", "stopped", "blocked"
                } else "stopped"
            status.set_app(app, state=app_state)
        # ``tracker.calls`` is a compatibility name for model results whose usage reached the
        # billing tracker, including a Training draft that may be passed or abandoned. It is NOT
        # an HTTP-attempt count: Gemini's internal 503/429/404/transport fallbacks do not carry
        # usage into CostTracker and are logged separately at the point of failure.
        tail = (f"accounted_provider_results={tracker.calls} "
                f"(HTTP fallback failures logged separately and excluded) "
                f"provider_spend=${tracker.run_spend_usd:.4f}")
        if save_err is not None:
            print(f"Run {run_id}: ❌ SAVE FAILED to {cfg.storage.backend} "
                  f"({type(save_err).__name__}: {save_err}) — buffered data may be incomplete; {tail}")
            raise save_err
        saved = getattr(store, "saved_summary", lambda: "")()
        detail = f" [{saved}]" if saved else ""
        if wedged:
            names = ", ".join(w.app for w in wedged)
            # join_timeout_s, not a hardcoded "30s": that flat number predates
            # _worker_join_timeout_s (see its own comment) and is wrong -- confusingly so --
            # on every run with openers enabled, where the real bound is 105s by default.
            print(f"Run {run_id}: saved to {cfg.storage.backend}{detail}, but {len(wedged)} "
                  f"worker(s) did not stop within {join_timeout_s:.0f}s ({names}) — NOT an "
                  f"unqualified success, a late write from a wedged worker could still land "
                  f"after this save; {tail}")
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
        result = subprocess.run(
            [adb, "devices"], capture_output=True, text=True, timeout=5, check=False)
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
