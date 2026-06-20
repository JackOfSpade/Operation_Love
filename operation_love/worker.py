"""Per-app worker thread.

One Worker drives one app (Bumble or Hinge) end-to-end. Multiple workers run
concurrently in a single process, sharing the ranker, store, and global budget.
Each worker owns its driver (a browser context / emulator connection) — failures
are isolated and auto-restarted with backoff, so a dead emulator can't take the
browser worker down.
"""
from __future__ import annotations

import random
import threading
import time
import traceback

from .drivers.base import DatingAppDriver
from .ranker.decider import Decider


class Worker(threading.Thread):
    def __init__(self, app, driver: DatingAppDriver, decider: Decider, opener_service,
                 store, run_id, pacing, stop_event: threading.Event, max_restarts: int = 5):
        super().__init__(name=f"worker-{app}", daemon=True)
        self.app = app
        self.driver = driver
        self.decider = decider
        self.opener_service = opener_service
        self.store = store
        self.run_id = run_id
        self.pacing = pacing
        self.stop_event = stop_event
        self.max_restarts = max_restarts

    def run(self) -> None:
        backoff, restarts = 2.0, 0
        while not self.stop_event.is_set():
            try:
                self._loop()
                return  # finished normally (deck empty or stop requested)
            except Exception:  # noqa: BLE001
                restarts += 1
                print(f"[worker-{self.app}] error (restart {restarts}/{self.max_restarts}):")
                traceback.print_exc()
                if restarts > self.max_restarts or self.stop_event.is_set():
                    print(f"[worker-{self.app}] giving up.")
                    return
                self.stop_event.wait(min(backoff, 60))
                backoff *= 2

    def _loop(self) -> None:
        self.driver.open_session()
        try:
            while not self.stop_event.is_set():
                if self.driver.out_of_profiles():
                    break
                profile = self.driver.next_profile()
                if profile is None:
                    break

                d = self.decider.decide(profile)
                if d.decision == "defer":
                    print(f"[worker-{self.app}] ranker not ready (cold-start) — seed labels with "
                          f"the labeling tool, then run autonomously. Stopping {self.app}.")
                    break

                self.store.record_decision(self.run_id, self.app, d.decision, d.score)
                if d.embedding:  # the swipe becomes a training label
                    self.store.add_label(self.run_id, self.app, d.decision == "like",
                                         d.embedding, source=d.source)

                if d.decision == "like":
                    opener = self.opener_service.maybe_opener(self.run_id, self.app, profile)
                    self.driver.like(opener)
                else:
                    self.driver.dislike()

                if self.opener_service.stop_requested:   # global budget/credit stop
                    self.stop_event.set()

                self._pace()
        finally:
            self.driver.close()

    def _pace(self) -> None:
        self.stop_event.wait(random.uniform(self.pacing.min_delay_s, self.pacing.max_delay_s))
