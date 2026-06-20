"""Per-app worker thread, in one of two modes.

observe  — SHADOW LEARNING. You swipe manually on real profiles in the live app;
           the worker captures each profile, watches your like/pass, embeds it,
           stores it as a label, and retrains the ranker live. No autonomous
           swiping, no openers. This is how the model learns your taste — from
           your real usage, not stock images.
auto     — AUTONOMOUS. The worker captures, scores with the trained ranker,
           and likes/dislikes itself (sending openers where the app allows).

Multiple workers run concurrently in one process, sharing the ranker, store, and
global budget. Failures are isolated and auto-restarted with backoff.
"""
from __future__ import annotations

import threading
import traceback

from .drivers.base import DatingAppDriver
from .human import human_cooldown, human_delay
from .ranker.decider import Decider


class Worker(threading.Thread):
    def __init__(self, app, driver: DatingAppDriver, decider: Decider, opener_service,
                 store, run_id, pacing, stop_event: threading.Event, mode: str = "observe",
                 retrain_every: int = 10, limiter=None, max_restarts: int = 5):
        super().__init__(name=f"worker-{app}", daemon=True)
        self.app = app
        self.driver = driver
        self.decider = decider
        self.opener_service = opener_service
        self.store = store
        self.run_id = run_id
        self.pacing = pacing
        self.stop_event = stop_event
        self.mode = mode
        self.retrain_every = max(1, int(retrain_every))
        self.limiter = limiter
        self.max_restarts = max_restarts

    def run(self) -> None:
        backoff, restarts = 2.0, 0
        while not self.stop_event.is_set():
            try:
                self._observe_loop() if self.mode == "observe" else self._auto_loop()
                return
            except Exception:  # noqa: BLE001
                restarts += 1
                print(f"[worker-{self.app}] error (restart {restarts}/{self.max_restarts}):")
                traceback.print_exc()
                if restarts > self.max_restarts or self.stop_event.is_set():
                    print(f"[worker-{self.app}] giving up.")
                    return
                self.stop_event.wait(human_cooldown(min(backoff, 60)))
                backoff *= 2

    # --- shadow learning: you swipe, the bot learns ---------------------
    def _observe_loop(self) -> None:
        print(f"[worker-{self.app}] observe mode — swipe manually; I'll learn from each swipe.")
        self.driver.open_session()
        added = 0
        try:
            while not self.stop_event.is_set():
                if self.driver.out_of_profiles():
                    break
                profile = self.driver.current_profile()      # capture the card you're viewing
                if profile is None:
                    continue
                liked = self.driver.wait_for_decision()       # block until your manual like/pass
                if liked is None:                             # card changed / timeout -> skip
                    continue
                vec = self.decider.embed(profile)
                if vec is None:                               # no face -> not a useful label
                    continue
                self.store.add_label(self.run_id, self.app, liked, vec, source="manual")
                self.store.record_decision(self.run_id, self.app,
                                           "like" if liked else "dislike", 1.0 if liked else 0.0)
                added += 1
                if added % self.retrain_every == 0:
                    ready = self.decider.retrain(self.store)
                    print(f"[worker-{self.app}] learned {added} labels this run; ranker ready={ready}")
        finally:
            self.driver.close()

    # --- autonomous: the bot swipes ------------------------------------
    def _auto_loop(self) -> None:
        acted = 0
        today0 = self.store.count_today(self.app) if self.limiter else 0
        self.driver.open_session()
        try:
            while not self.stop_event.is_set():
                if self.driver.out_of_profiles():
                    break
                if self.limiter and not self.limiter.allow(acted, today0 + acted):
                    print(f"[worker-{self.app}] rate limit reached ({self.limiter.describe()}); "
                          f"stopping {self.app}.")
                    break
                profile = self.driver.next_profile()
                if profile is None:
                    break

                d = self.decider.decide(profile)
                if d.decision == "defer":
                    print(f"[worker-{self.app}] ranker not ready (cold-start) — run in observe "
                          f"mode and swipe manually to seed it. Stopping {self.app}.")
                    break

                self.store.record_decision(self.run_id, self.app, d.decision, d.score)
                if d.embedding:                               # the swipe becomes a label too
                    self.store.add_label(self.run_id, self.app, d.decision == "like",
                                         d.embedding, source=d.source)

                if d.decision == "like":
                    opener = self.opener_service.maybe_opener(self.run_id, self.app, profile)
                    self.driver.like(opener)
                else:
                    self.driver.dislike()
                acted += 1

                if self.opener_service.stop_requested:        # global budget/credit stop
                    self.stop_event.set()
                self._pace()
        finally:
            self.driver.close()

    def _pace(self) -> None:
        self.stop_event.wait(human_delay(self.pacing.swipe_delay_s))
