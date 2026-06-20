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
                 retrain_every: int = 10, limiter=None, max_restarts: int = 5, status=None):
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
        self.status = status              # optional RunStatus (live overlay/hub); None in tests

    # --- live status (overlay + hub); no-ops when status is unset ---------
    def _stat(self, **fields) -> None:
        if self.status:
            self.status.set_app(self.app, **fields)
        self._render()

    def _render(self) -> None:
        if self.status:
            self.driver.render_status(self.status.app_view(self.app))

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
        self._stat(mode="observe", state="waiting")
        added = 0
        last_retrained = 0
        try:
            while not self.stop_event.is_set():
                if self.driver.out_of_profiles():
                    self._stat(state="out_of_profiles")
                    break
                profile = self.driver.current_profile()      # capture the card you're viewing
                if profile is None:
                    continue
                self._stat(state="waiting")                   # overlay: "swipe — learning your taste"
                liked = self.driver.wait_for_decision()       # block until your manual like/pass
                if liked is None:                             # card changed / timeout -> skip
                    continue
                vec = self.decider.embed(profile)
                if vec is None:                               # no face -> not a useful label
                    self._stat(last_decision="no_face")
                    continue
                self.store.add_label(self.run_id, self.app, liked, vec, source="manual")
                self.store.record_decision(self.run_id, self.app,
                                           "like" if liked else "dislike", 1.0 if liked else 0.0)
                if self.status:
                    self.status.record_swipe(self.app, "like" if liked else "pass")
                    self.status.inc_labels(1)
                self._render()
                added += 1
                if added % self.retrain_every == 0:
                    self._retrain_after_observe_labels(added)
                    last_retrained = added
            if added and added != last_retrained:
                self._retrain_after_observe_labels(added)
        finally:
            self._stat(state="stopped")
            self.driver.close()

    def _retrain_after_observe_labels(self, added: int) -> None:
        ready = self.decider.retrain(self.store)
        if self.status:
            self.status.set_global(ranker_ready=ready)
        self._render()
        print(f"[worker-{self.app}] learned {added} labels this run; ranker ready={ready}")

    # --- autonomous: the bot swipes ------------------------------------
    def _auto_loop(self) -> None:
        acted = 0
        today0 = self.store.count_today(self.app) if self.limiter else 0
        self.driver.open_session()
        self._stat(mode="auto", state="scoring")
        try:
            while not self.stop_event.is_set():
                if self.driver.out_of_profiles():
                    self._stat(state="out_of_profiles")
                    break
                if self.limiter and not self.limiter.allow(acted, today0 + acted):
                    self._stat(state="rate_limited")
                    print(f"[worker-{self.app}] rate limit reached ({self.limiter.describe()}); "
                          f"stopping {self.app}.")
                    break
                profile = self.driver.next_profile()
                if profile is None:
                    break

                self._stat(state="scoring")
                d = self.decider.decide(profile)
                if d.decision == "defer":
                    self._stat(last_decision="defer", state="stopped")
                    print(f"[worker-{self.app}] ranker not ready (cold-start) — run in observe "
                          f"mode and swipe manually to seed it. Stopping {self.app}.")
                    break

                self.store.record_decision(self.run_id, self.app, d.decision, d.score)
                if d.embedding:                               # the swipe becomes a label too
                    self.store.add_label(self.run_id, self.app, d.decision == "like",
                                         d.embedding, source=d.source)
                    if self.status:
                        self.status.inc_labels(1)

                if d.decision == "like":
                    opener = self.opener_service.maybe_opener(self.run_id, self.app, profile)
                    self.driver.like(opener)
                else:
                    self.driver.dislike()
                if self.status:
                    self.status.record_swipe(self.app, d.decision, d.score)
                self._render()
                acted += 1

                if self.opener_service.stop_requested:        # global budget/credit stop
                    self.stop_event.set()
                self._pace()
        finally:
            self._stat(state="stopped")
            self.driver.close()

    def _pace(self) -> None:
        self.stop_event.wait(human_delay(self.pacing.swipe_delay_s))
