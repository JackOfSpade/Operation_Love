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
from .vision.embed import Embedder
from .vision.quality import QualityFilter
from .worker import Worker


def run(config_path: str = "config.yaml") -> None:
    cfg = cfg_mod.load(config_path)
    cfg_mod.validate(cfg)
    run_id = uuid.uuid4().hex[:12]

    caps = Capabilities.detect()
    print(caps.banner())
    if cfg.storage.backend == "bigquery" and caps.missing("bigquery"):
        raise SystemExit("storage.backend=bigquery but google-cloud-bigquery isn't installed "
                         "(`pip install -e '.[bq]'`, or set storage.backend: sqlite).")

    if caps.missing("arcface", "clip"):
        print("[degrade] ml extra not installed -> ranking unavailable "
              "(`pip install -e '.[ml]'`). Workers will defer until it's present.")

    store = make_store(cfg)
    labels = store.load_labels()
    print(f"[store] backend={cfg.storage.backend} labels={len(labels)}  "
          f"apps={cfg.enabled_apps}")

    tracker = CostTracker(cfg.budget.pricing, cfg.budget.run_budget_usd)
    opener_client = None
    if cfg.opener.enabled and not caps.missing("anthropic"):
        opener_client = AnthropicOpener(cfg.opener.model, cfg.opener.max_tokens)
    elif cfg.opener.enabled:
        print("[degrade] anthropic SDK not installed -> swiping without openers")
    opener_service = OpenerService(opener_client, tracker, store, cfg.opener.style,
                                   cfg.budget.on_exhausted)

    model = PreferenceModel(min_labels=cfg.ranker.min_labels_to_engage,
                            threshold=cfg.ranker.like_threshold)
    ready = model.train(labels)
    print(f"[ranker] labels={len(labels)} ready={ready} "
          f"(min={cfg.ranker.min_labels_to_engage}, threshold={cfg.ranker.like_threshold})")
    quality = QualityFilter(cfg.quality_filter.enabled, cfg.quality_filter.min_score,
                            cfg.quality_filter.metric)
    embedder = Embedder(cfg)
    decider = RankerDecider(quality, embedder, model)

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)

    workers = []
    for app in cfg.enabled_apps:
        app_cfg = cfg.apps.get(app, {}) or {}
        mode = app_cfg.get("mode", cfg.mode)                          # per-app override
        lim = {**cfg.limits, **(app_cfg.get("limits", {}))}
        limiter = RateLimiter(lim.get("max_per_run"), lim.get("max_per_day"))
        driver = make_driver(app, cfg)
        w = Worker(app, driver, decider, opener_service, store, run_id, cfg.pacing,
                   stop_event, mode=mode, retrain_every=cfg.ranker.retrain_every, limiter=limiter)
        print(f"[worker-{app}] mode={mode} limits={limiter.describe()}")
        workers.append(w)
        w.start()

    try:
        while any(w.is_alive() for w in workers):
            stop_event.wait(0.5)
    finally:
        stop_event.set()
        for w in workers:
            w.join(timeout=30)
        store.flush()
        store.close()
        print(f"[run {run_id}] openers={tracker.calls} spend=${tracker.run_spend_usd:.4f}")


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _stop(*_):
        print("\n[supervisor] shutdown requested; stopping workers...")
        stop_event.set()
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):  # not in main thread / unsupported on this OS
            pass
