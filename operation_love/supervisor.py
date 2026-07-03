"""Supervisor — one process runs all enabled apps concurrently.

Loads the shared resources once (ranker, BigQuery store, GLOBAL budget tracker),
launches one Worker per enabled app, and supervises them: clean shutdown on
Ctrl-C / SIGTERM (so it runs as an always-on service on any OS), flush + close
on exit. This replaces the single-app loop.
"""
from __future__ import annotations

import signal
import threading
import uuid

from . import config as cfg_mod
from .costing import CostTracker
from .drivers import make_driver
from .opener.opener import AnthropicOpener
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
_WORKER_JOIN_TIMEOUT_S = 30.0


def _resolve_run_cap(config_cap: int | None, override: int | None) -> int | None:
    """Per-run override of the max-swipes-per-run cap (set from the hub, auto mode).

    override is None -> no override, use the config value.
    override == 0    -> UNLIMITED for this run (no per-run cap).
    override > 0     -> cap this run at that many swipes.
    (The per-DAY cap still applies regardless — it's the standing safety floor.)
    """
    if override is None:
        return config_cap
    return override or None


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
    cfg_mod.validate(cfg)
    run_id = uuid.uuid4().hex[:12]

    # Create + publish status up front (before the slow store/model setup) so the
    # hub shows a live "phase" immediately rather than appearing to hang on Start.
    status = RunStatus(run_id, cfg.enabled_apps, min_labels=cfg.ranker.min_labels_to_engage,
                       mode=cfg.mode, budget_cap=cfg.budget.run_budget_usd)
    if on_status:
        on_status(status)

    caps = Capabilities.detect()
    print(caps.banner())
    missing_cloud = caps.missing("bigquery", "cloud_storage") if cfg.storage.backend == "bigquery" else []
    if missing_cloud:
        raise SystemExit("Storage.backend=bigquery but cloud storage dependencies are missing "
                         f"({', '.join(missing_cloud)}). Install `pip install -e '.[bq]'`, "
                         "or set storage.backend: sqlite.")

    if caps.missing("arcface", "clip"):
        print("Degrade: ml extra not installed -> ranking unavailable "
              "(`pip install -e '.[ml]'`). Workers will defer until it's present.")

    status.set_global(phase="loading saved data")     # BigQuery ensure-tables + label load
    store = make_store(cfg)
    if on_store:
        on_store(store)            # publish the live store so the hub eval reads live in-memory labels
    labels = store.load_labels()
    status.set_global(labels=len(labels))
    print(f"Store: backend={cfg.storage.backend} labels={len(labels)}  "
          f"apps={cfg.enabled_apps}")

    tracker = CostTracker(cfg.budget.pricing, cfg.budget.run_budget_usd)
    opener_client = None
    if cfg.opener.enabled and not caps.missing("anthropic"):
        opener_client = AnthropicOpener(cfg.opener.model, cfg.opener.max_tokens)
    elif cfg.opener.enabled:
        print("Degrade: anthropic SDK not installed -> swiping without openers")
    opener_service = OpenerService(opener_client, tracker, store, cfg.opener.style,
                                   cfg.budget.on_exhausted)

    status.set_global(phase="training ranker")
    model = PreferenceModel(min_labels=cfg.ranker.min_labels_to_engage,
                            threshold=cfg.ranker.like_threshold)
    ready = model.train(labels)
    status.set_global(ranker_ready=ready, phase="launching app")
    print(f"Ranker: labels={len(labels)} ready={ready} "
          f"(min={cfg.ranker.min_labels_to_engage}, threshold={cfg.ranker.like_threshold})")
    quality = QualityFilter(cfg.quality_filter.enabled, cfg.quality_filter.min_score,
                            cfg.quality_filter.metric)
    embedder = Embedder(cfg)
    decider = RankerDecider(quality, embedder, model)

    stop_event = stop_event if stop_event is not None else threading.Event()
    _install_signal_handlers(stop_event)

    workers = []
    for app in cfg.enabled_apps:
        app_cfg = cfg.apps.get(app, {}) or {}
        mode = app_cfg.get("mode", cfg.mode)                          # per-app override
        lim = {**cfg.limits, **(app_cfg.get("limits", {}))}
        run_cap = _resolve_run_cap(lim.get("max_per_run"), max_per_run)
        limiter = RateLimiter(run_cap, lim.get("max_per_day"),
                              lim.get("max_likes_per_run"))
        driver = make_driver(app, cfg)
        w = Worker(app, driver, decider, opener_service, store, run_id, cfg.pacing,
                   stop_event, mode=mode, retrain_every=cfg.ranker.retrain_every,
                   limiter=limiter, status=status)
        print(f"{app.title()} worker mode={mode} limits={limiter.describe()}")
        workers.append(w)
        w.start()

    status.set_global(phase="live")
    try:
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
        for w in workers:
            w.join(timeout=_WORKER_JOIN_TIMEOUT_S)
            if w.is_alive():
                # Proceeding anyway (below) rather than blocking forever: the worker's
                # own stop_event is set, but it's still stuck mid-capture/embed/API-call.
                # It may call store.add_label()/record_decision() after store.close()
                # runs — a store write racing a closed store is the tradeoff for not
                # hanging shutdown indefinitely on one wedged app.
                print(f"Supervisor: worker '{w.app}' did not stop within "
                      f"{_WORKER_JOIN_TIMEOUT_S:.0f}s; proceeding to save without it "
                      "(it may still be running in the background).")
        for app in cfg.enabled_apps:
            status.set_app(app, state="saving")
        save_err = None
        try:
            store.flush()                 # raises if any buffered insert was rejected
            store.close()
        except Exception as exc:  # noqa: BLE001 — report a clear save outcome, then re-raise
            save_err = exc
        finally:
            status.set_global(running=False, phase=("stopped" if save_err is None else "save_failed"),
                              budget_spent=tracker.run_spend_usd, openers=tracker.calls)
            for app in cfg.enabled_apps:
                status.set_app(app, state=("stopped" if save_err is None else "error"))
        tail = f"openers={tracker.calls} spend=${tracker.run_spend_usd:.4f}"
        if save_err is None:
            saved = getattr(store, "saved_summary", lambda: "")()
            detail = f" [{saved}]" if saved else ""
            print(f"Run {run_id}: ✅ all data saved to {cfg.storage.backend}{detail}; {tail}")
        else:
            print(f"Run {run_id}: ❌ SAVE FAILED to {cfg.storage.backend} "
                  f"({type(save_err).__name__}: {save_err}) — buffered data may be incomplete; {tail}")
            raise save_err


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _stop(*_):
        print("\nSupervisor: shutdown requested; stopping workers...")
        stop_event.set()
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):  # not in main thread / unsupported on this OS
            pass
