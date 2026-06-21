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
import uuid

from .drivers.base import DatingAppDriver, DriverClosed
from .human import human_cooldown, human_delay
from .ranker.decider import Decider

_OBSERVE_CAPTURE_BUSY = "Capturing profile — please wait before your next swipe"
_OBSERVE_PROCESSING_BUSY = "Processing — please wait before your next swipe"
_NO_PHOTO_RETRY_S = 0.5


class Worker(threading.Thread):
    def __init__(self, app, driver: DatingAppDriver, decider: Decider, opener_service,
                 store, run_id, pacing, stop_event: threading.Event, mode: str = "observe",
                 retrain_every: int = 1, limiter=None, max_restarts: int = 5, status=None):
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

    def _block_observe_capture(self, **status_fields) -> None:
        self.driver.render_busy(_OBSERVE_CAPTURE_BUSY)
        self._stat(state="capturing", **status_fields)

    def run(self) -> None:
        backoff, restarts = 2.0, 0
        while not self.stop_event.is_set():
            try:
                self._observe_loop() if self.mode == "observe" else self._auto_loop()
                return
            except DriverClosed as exc:
                print(f"[worker-{self.app}] {exc}; stopping run so buffered data can be saved.")
                self.stop_event.set()
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
        self._block_observe_capture(mode="observe")
        added = 0
        last_retrained = 0
        pending_error = False
        try:
            while not self.stop_event.is_set():
                if self.driver.out_of_profiles():
                    self._stat(state="out_of_profiles")
                    break
                profile = self.driver.current_profile()      # capture the card you're viewing
                if self.stop_event.is_set():
                    break
                if profile is None:
                    continue
                if not profile.photos:
                    print(f"[worker-{self.app}] captured 0 profile photos; waiting to recapture before learning.")
                    self._block_observe_capture(last_decision="no_photos")
                    self.stop_event.wait(_NO_PHOTO_RETRY_S)
                    continue
                self.driver.render_busy(None)                 # processing done -> OK to swipe now
                self._stat(state="waiting")                   # overlay: "swipe — learning your taste"
                print(f"[worker-{self.app}] ✅ READY — swipe this profile (like or pass).")
                liked = self.driver.wait_for_decision(should_stop=self.stop_event.is_set)
                if liked is None:                             # card changed / timeout / stop -> skip
                    self._block_observe_capture()
                    continue
                if self.stop_event.is_set():
                    break
                # block the next swipe while this one embeds (avoids mis-attribution)
                print(f"[worker-{self.app}] got {'LIKE' if liked else 'PASS'} — "
                      f"processing, don't swipe yet…")
                self.driver.render_busy(_OBSERVE_PROCESSING_BUSY)
                profile_id = uuid.uuid4().hex
                metadata = self._label_metadata(profile)
                archived = self.store.record_profile(self.run_id, self.app, profile_id, liked,
                                                     source="manual", photos=profile.photos, **metadata)
                vec = self.decider.embed(profile)
                if vec is None:                               # no face -> not a useful label
                    self._stat(last_decision="no_face")
                    continue
                if self.stop_event.is_set():
                    break
                if archived is False:                         # images couldn't be saved -> no label without them
                    self._stat(last_decision="archive_failed")
                    continue
                self.store.add_label(self.run_id, self.app, liked, vec, source="manual",
                                     profile_id=profile_id, **metadata)
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
        except BaseException:
            pending_error = True
            raise
        finally:
            try:
                if added and added != last_retrained:
                    try:
                        self._retrain_after_observe_labels(added)
                    except Exception:  # noqa: BLE001
                        if not pending_error:
                            raise
                        print(f"[worker-{self.app}] final retrain skipped after shutdown error:")
                        traceback.print_exc()
            finally:
                self.driver.render_busy(None)
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
        liked = 0
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
                if self.stop_event.is_set():
                    break
                if profile is None:
                    break

                self._stat(state="scoring")
                d = self.decider.decide(profile)
                if self.stop_event.is_set():
                    break
                if d.decision == "defer":
                    self._stat(last_decision="defer", state="stopped")
                    print(f"[worker-{self.app}] ranker not ready (cold-start) — run in observe "
                          f"mode and swipe manually to seed it. Stopping {self.app}.")
                    break

                # Per-run like budget: stop the run rather than mislabel a wanted
                # like as a pass (keeps the right-swipe ratio human; see limits.py).
                if d.decision == "like" and self.limiter and not self.limiter.allow_like(liked):
                    self._stat(state="rate_limited")
                    print(f"[worker-{self.app}] per-run like budget reached "
                          f"({self.limiter.describe()}); stopping {self.app}.")
                    break

                # AUTO mode is pure INFERENCE: log the decision (for stats + the daily
                # rate limit), but do NOT store it as a training label. An autonomous
                # swipe is the model's own PREDICTION, not ground truth — training on it
                # would create a self-reinforcing feedback loop that amplifies the
                # model's biases. Training labels come ONLY from manual/observe swipes.
                self.store.record_decision(self.run_id, self.app, d.decision, d.score)
                if self.stop_event.is_set():
                    break

                if d.decision == "like":
                    opener = self.opener_service.maybe_opener(self.run_id, self.app, profile)
                    self.driver.like(opener)
                    liked += 1
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

    @staticmethod
    def _label_metadata(profile) -> dict:
        return {"photo_count": len(profile.photos)}
