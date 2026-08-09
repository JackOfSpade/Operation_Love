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

import random
import threading
import traceback
import uuid
from datetime import date

from .config import PacingCfg
from .drivers.base import DatingAppDriver, DriverClosed
from .human import human_cooldown, human_delay
from .human_motion import think_time_s
from .interaction import AutoSessionPolicy
from .ranker.decider import Decider, Decision

_OBSERVE_CAPTURE_BUSY = "Capturing profile — please wait before your next swipe"
_OBSERVE_PROCESSING_BUSY = "Processing — please wait before your next swipe"
_NO_PHOTO_RETRY_S = 0.5
_PROFILE_LOG_WIDTH = 72
# think_time_s() is calibrated to real measured Hinge dwell data around this many
# seconds — pacing.swipe_delay_s scales it proportionally, so the config knob still
# speeds up/slows down pacing (and a test fixture's swipe_delay_s=0.0 still paces
# instantly) while think_time_s supplies the measured like-vs-pass asymmetry shape.
# Derived from PacingCfg's own default so the baseline can't drift from the config.
_THINK_TIME_BASELINE_S = PacingCfg().swipe_delay_s


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

    def _capture_failure(self, exc: BaseException) -> None:
        """Let the driver snapshot the on-screen failure state into its debug log. Called from
        inside the loop's except — WHILE the transport is still live — because the loop's finally
        closes the driver before run()'s handler sees the exception."""
        snap = getattr(self.driver, "snapshot_failure", None)
        if callable(snap):
            try:
                snap(exc)
            except Exception:  # noqa: BLE001 — debug capture must never mask the real error
                pass

    def _capture_failure_if_unexpected(self, exc: BaseException) -> None:
        """Shared except-clause body for _observe_loop/_auto_loop: snapshot the on-screen
        state for any failure that isn't a clean DriverClosed stop."""
        if not isinstance(exc, DriverClosed):
            self._capture_failure(exc)

    def _finish_session(self, state: str = "stopped") -> None:
        """Shared unconditional cleanup for _observe_loop/_auto_loop's finally: publish the
        loop's TERMINAL state (rate_limited / out_of_profiles / stopped — not a blanket
        "stopped", which would hide why the run ended) and release the driver."""
        self._stat(state=state)
        self.driver.close()

    def _profile_separator(self) -> None:
        print("-" * _PROFILE_LOG_WIDTH)

    def _block_observe_capture(self, **status_fields) -> None:
        self.driver.render_busy(_OBSERVE_CAPTURE_BUSY)
        self._stat(state="capturing", **status_fields)

    def _block_observe_processing(self) -> None:
        # Same "don't swipe yet" signal as capture, but for the embed/store window after a
        # manual swipe. Drives BOTH channels identically for every app: the in-page busy
        # modal (Bumble, when enabled) AND the shared status state the hub banner reads —
        # so apps with no on-screen overlay (Hinge) still show WAIT during the slow embed
        # instead of a stale SWIPE prompt that would mis-attribute the next swipe.
        self.driver.render_busy(_OBSERVE_PROCESSING_BUSY)
        self._stat(state="acting")

    def run(self) -> None:
        backoff, restarts = 2.0, 0
        # Publish the configured mode without touching the driver. In particular this must
        # happen before open_session(): if opening an auto session fails, the error banner's
        # per-app selector must not mistake the still-default AppStatus mode for "observe".
        if self.status:
            self.status.set_app(self.app, mode=self.mode)
        while not self.stop_event.is_set():
            try:
                self._observe_loop() if self.mode == "observe" else self._auto_loop()
                return
            except DriverClosed as exc:
                print(f"{exc}; Stopping run so buffered data can be saved.")
                self.stop_event.set()
                return
            except Exception:  # noqa: BLE001
                # NOTE: the on-screen failure screenshot is captured INSIDE the loop's except
                # (_capture_failure), BEFORE the loop's finally closes the driver — by here the
                # transport is already closed, so a snapshot would be blank.
                # HALT-on-unexpected: any AUTO run that hits an unexpected error must STOP, for
                # EVERY app — an autonomous loop that restart-and-continues would keep swiping
                # blindly (risky) and rotate the crucial failure logs away. The HALT line + the
                # traceback go to stdout/stderr, which the hub tees into the live-log panel, so
                # the stop is visible on the GUI terminal view (and lands in the bug report).
                # Observe mode keeps per-driver behavior via halt_on_error — which now
                # DEFAULTS TO TRUE on the driver ABC (see DatingAppDriver.halt_on_error).
                # The default used to be False here, so a driver that never declared the
                # attribute (PlaywrightDriver did not) got restart-with-backoff by omission
                # rather than by decision: the riskier path was what you got by forgetting.
                # Restarting is not free even in observe mode, where the bot only reads — the
                # worker re-attaches to whatever is on screen, and a driver that was confused
                # about which card it was on then mis-attributes the manual swipes it records,
                # corrupting the taste model permanently. Restart is now reachable only by
                # explicitly setting apps.<app>.halt_on_error: false, which config validation
                # allows in observe mode only.
                if self.mode == "auto" or getattr(self.driver, "halt_on_error", True):
                    print(f"{self.app.title()} unexpected error in {self.mode} mode; HALTING "
                          f"(no restart) so nothing swipes blindly and the debug logs survive:")
                    traceback.print_exc()
                    # Publish error to hub so the banner shows the failure (not a silent stop).
                    self._stat(state="error", error=traceback.format_exc().strip().splitlines()[-1])
                    self.stop_event.set()
                    return
                restarts += 1
                print(f"{self.app.title()} worker error (restart {restarts}/{self.max_restarts}):")
                traceback.print_exc()
                if restarts > self.max_restarts or self.stop_event.is_set():
                    print(f"{self.app.title()} worker giving up.")
                    return
                self.stop_event.wait(human_cooldown(min(backoff, 60)))
                backoff *= 2

    # --- shadow learning: you swipe, the bot learns ---------------------
    def _observe_loop(self) -> None:
        print(f"{self.app.title()} observe mode — swipe manually; I'll learn from each swipe.")
        self.driver.open_session()
        self._stat(mode="observe")             # set mode for the hub; the loop owns per-card WAIT/SWIPE
        added = 0
        last_retrained = 0
        pending_error = False
        terminal_state = "stopped"            # overwritten below when the loop ends for a known reason
        try:
            while not self.stop_event.is_set():
                if self.driver.out_of_profiles():
                    terminal_state = "out_of_profiles"
                    self._stat(state=terminal_state)
                    break
                self._profile_separator()
                self._block_observe_capture()                # WAIT cue for every card (capturing state)
                profile = self.driver.current_profile()      # capture the card you're viewing
                if self.stop_event.is_set():
                    break
                if profile is None:
                    continue
                if not profile.photos:
                    print("Captured 0 profile photos; waiting to recapture before learning.")
                    self._stat(last_decision="no_photos")     # stays WAIT (busy still up from this iteration)
                    self.stop_event.wait(_NO_PHOTO_RETRY_S)
                    continue
                self.driver.render_busy(None)                 # processing done -> OK to swipe now
                self._stat(state="waiting")                   # overlay: "swipe — learning your taste"
                print("✅ READY — swipe this profile (like or pass).")
                liked = self.driver.wait_for_decision(timeout=None,
                                                      should_stop=self.stop_event.is_set)
                if liked is None:                             # card changed / deck empty / stop -> recapture
                    continue                                  # next iteration re-blocks + recaptures the card
                if self.stop_event.is_set():
                    break
                # block the next swipe while this one embeds (avoids mis-attribution)
                decision = "LIKE" if liked else "PASS"
                print(f"Got {decision} — processing, don't swipe yet…")
                self._block_observe_processing()
                profile_id = uuid.uuid4().hex
                metadata = self._label_metadata(profile)
                archived = self.store.record_profile(self.run_id, self.app, profile_id, liked,
                                                     source="manual", photos=profile.photos, **metadata)
                vec = self.decider.embed(profile)
                if vec is None:                               # no face -> not a useful label
                    self._stat(last_decision="no_face")
                    continue
                if archived is False:                         # images couldn't be saved -> no label without them
                    self._stat(last_decision="archive_failed")
                    continue
                # NOTE: deliberately no stop_event check here. A manual swipe already
                # happened and its photos + embedding are already committed above — Stop
                # must end the loop AFTER this label is saved (the while-loop condition
                # below does that without starting a NEW capture), not discard work in hand.
                self.store.add_label(self.run_id, self.app, liked, vec, source="manual",
                                     profile_id=profile_id, **metadata)
                self.store.record_decision(self.run_id, self.app,
                                           "like" if liked else "dislike", 1.0 if liked else 0.0,
                                           source="manual")
                if self.status:
                    self.status.record_swipe(self.app, "like" if liked else "pass")
                    self.status.inc_labels(1)
                self._render()
                added += 1
                if added % self.retrain_every == 0:
                    self._retrain_after_observe_labels(added)
                    last_retrained = added
        except BaseException as exc:
            pending_error = True
            if isinstance(exc, Exception):
                self._capture_failure_if_unexpected(exc)      # snapshot WHILE the transport is live
            raise
        finally:
            try:
                if added and added != last_retrained:
                    try:
                        self._retrain_after_observe_labels(added)
                    except Exception:  # noqa: BLE001
                        if not pending_error:
                            raise
                        print(f"{self.app.title()} final retrain skipped after shutdown error:")
                        traceback.print_exc()
            finally:
                self.driver.render_busy(None)
                self._finish_session(terminal_state)

    def _retrain_after_observe_labels(self, added: int) -> None:
        ready = self.decider.retrain(self.store)
        if self.status:
            self.status.set_global(ranker_ready=ready)
        self._render()
        print(f"Learned {added} labels this run; ranker ready={ready}")

    # --- autonomous: the bot swipes ------------------------------------
    def _auto_loop(self) -> None:
        acted = 0
        liked = 0
        # This state belongs to one auto session, not to the learned preference
        # model.  It may only make a marginal model like more conservative; it
        # never manufactures a like.  Fakes/third-party deciders need not expose
        # the underlying model, hence the guarded configured-threshold lookup.
        configured_threshold = getattr(getattr(self.decider, "model", None), "threshold", 0.5)
        try:
            configured_threshold = float(configured_threshold)
        except (TypeError, ValueError):
            configured_threshold = 0.5
        if not 0.0 < configured_threshold < 1.0:
            configured_threshold = 0.5
        self._auto_policy = AutoSessionPolicy(base_threshold=configured_threshold)
        # AndroidDriver exposes the policy through a deliberately optional
        # hook.  Other drivers can ignore it, while lightweight fakes and older
        # plugins still receive it through a harmless instance attribute.
        install_policy = getattr(self.driver, "set_auto_session_policy", None)
        if callable(install_policy):
            install_policy(self._auto_policy)
        else:
            try:
                setattr(self.driver, "_auto_policy", self._auto_policy)
            except (AttributeError, TypeError):
                pass
        # Session micro-break fatigue model: re-rolled each stretch so a long run's
        # break pattern isn't governed by one fixed hazard rate for its whole duration
        # (see _maybe_session_break).
        self._actions_since_break = 0
        self._break_hazard = random.uniform(0.04, 0.14)
        self._break_due_after = random.uniform(12, 35)
        # A no-op RateLimiter (all fields None, the shipped default) is still truthy.
        # Daily history is needed only for max_per_day; querying it for an uncapped run
        # adds startup latency and can turn an irrelevant store read failure into a halt.
        has_daily_limit = self.limiter is not None and self.limiter.max_per_day is not None
        today0 = self.store.count_today(self.app) if has_daily_limit else 0
        today_acted = 0            # actions this worker made since today0 was last measured
        today_date = date.today()  # LOCAL day — matches the store's count_today() day boundary
        self.driver.open_session()
        self._stat(mode="auto", state="scoring")
        terminal_state = "stopped"            # overwritten below when the loop ends for a known reason
        try:
            while not self.stop_event.is_set():
                if has_daily_limit:
                    today = date.today()
                    if today != today_date:
                        # Local midnight crossed mid-run — re-baseline against the new day
                        # instead of forever charging new-day actions against a day that's
                        # already over. Only queries the store on the rollover itself, not
                        # every profile.
                        today0 = self.store.count_today(self.app)
                        today_acted = 0
                        today_date = today
                if self.driver.out_of_profiles():
                    terminal_state = "out_of_profiles"
                    self._stat(state=terminal_state)
                    break
                if self.limiter and not self.limiter.allow(acted, today0 + today_acted):
                    terminal_state = "rate_limited"
                    self._stat(state=terminal_state)
                    print(f"{self.app.title()} rate limit reached ({self.limiter.describe()}); "
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
                    print(f"{self.app.title()} ranker not ready (cold-start) — run in observe "
                          f"mode and swipe manually to seed it. Stopping {self.app}.")
                    break

                # A session can decline a marginal model like according to its
                # bounded fatigue/context state.  This happens before every
                # limit check so a policy-demoted like neither consumes a like
                # budget nor trips the running like-ratio guard.
                d = self._auto_policy.apply_decision(d, profile).decision

                # Per-run like budget: stop the run rather than mislabel a wanted
                # like as a pass (keeps the right-swipe ratio human; see limits.py).
                if d.decision == "like" and self.limiter and not self.limiter.allow_like(liked):
                    terminal_state = "rate_limited"
                    self._stat(state=terminal_state)
                    print(f"{self.app.title()} per-run like budget reached "
                          f"({self.limiter.describe()}); stopping {self.app}.")
                    break

                # Ratio shape: demote this like to a pass when the running like-rate
                # is at the ceiling (soft cap — does NOT halt the run).
                if d.decision == "like" and self.limiter and not self.limiter.allow_like_ratio(liked, acted):
                    assert acted > 0  # allow_like_ratio returns True when acted==0, so we can't be here
                    ratio_pct = f"{liked / acted:.0%}"
                    print(f"{self.app.title()} like-ratio ceiling "
                          f"({ratio_pct} ≥ {self.limiter.target_like_ratio:.0%}) — demoting to pass")
                    d = Decision("dislike", d.score, d.embedding, d.source)

                if self.stop_event.is_set():
                    break

                if d.decision == "like":
                    # Only generate a Claude opener for apps that can actually send one
                    # at swipe time (Hinge). On Bumble we'd just discard it — wasted credits.
                    pick = (self.opener_service.maybe_opener(self.run_id, self.app, profile)
                            if getattr(self.driver, "accepts_opener", True) else None)
                    # item_index lets the driver attach the comment to the photo/prompt the
                    # opener is actually about, not blindly the first one.
                    self.driver.like(pick.text if pick else None,
                                     item_index=pick.index if pick else 0)
                    liked += 1
                else:
                    self.driver.dislike()

                # AUTO mode is pure INFERENCE: log the decision (for stats + the daily rate
                # limit) AFTER the action actually landed — do NOT store it as a training label
                # (training on the model's own prediction would create a self-reinforcing feedback
                # loop), and do NOT record a phantom if like()/dislike() raised (e.g. a
                # HingeActionError halt on an unsent like) and so never landed.
                self.store.record_decision(self.run_id, self.app, d.decision, d.score, source="auto")
                if self.status:
                    self.status.record_swipe(self.app, d.decision, d.score)
                self._render()
                acted += 1
                today_acted += 1

                # no_face still lands as a conservative pass in the pre-existing
                # worker flow, even though its audit decision stays "no_face".
                # The policy must count the physical action only after it landed.
                landed_action = "like" if d.decision == "like" else "dislike"
                self._auto_policy.record_landed_action(landed_action)

                if self.opener_service.stop_requested:        # global budget/credit stop
                    self.stop_event.set()
                self._pace(landed_action, profile=profile, score=d.score)
                self._maybe_session_break()
        except Exception as exc:  # noqa: BLE001
            self._capture_failure_if_unexpected(exc)          # snapshot WHILE the transport is live
            raise                                             # (the finally below closes it)
        finally:
            self._finish_session(terminal_state)

    def _pace(self, decision: str, *, profile=None, score: float | None = None) -> None:
        # ``0`` is the documented test/no-pacing setting.  Do not consume the
        # session policy's random state merely to wait for zero seconds.
        if self.pacing.swipe_delay_s == 0 and profile is not None:
            return
        if not getattr(self.driver, "think_time_calibrated", False):
            # No app-specific calibration for this driver -> the flat, decision-agnostic
            # anchor (unchanged pre-existing behavior for e.g. Bumble).
            self.stop_event.wait(human_delay(self.pacing.swipe_delay_s))
            return
        # Decision-aware "think time" (measured like/pass dwell asymmetry), scaled by the
        # configured anchor so pacing.swipe_delay_s is a real knob rather than an on/off
        # switch. config.validate() bounds it: an unbounded scale lets a negative or
        # near-zero value collapse the wait to ~0 (Event.wait treats a negative timeout as
        # "return now"), i.e. machine-speed swiping on a live account.
        scale = self.pacing.swipe_delay_s / _THINK_TIME_BASELINE_S
        policy = getattr(self, "_auto_policy", None)
        if policy is not None and profile is not None:
            # Keep the old direct-call behavior for diagnostics and legacy tests;
            # actual auto-loop calls carry the real captured profile and score.
            delay = policy.post_action_delay_s(decision, profile,
                                                0.5 if score is None else score,
                                                scale=scale)
            self.stop_event.wait(delay)
            return
        self.stop_event.wait(think_time_s("like" if decision == "like" else "pass") * scale)

    def _maybe_session_break(self) -> None:
        """Randomized micro-break between profiles — mimics stepping away.

        Break likelihood ramps up the longer it's been since the last break (real
        attention fatigue isn't memoryless), and both the hazard rate and the typical
        interval are re-rolled after every break. A constant per-swipe probability
        would instead produce an exactly geometric gap distribution — a detectable,
        single-parameter bot signature no real human's break pattern has.
        """
        if self.pacing.swipe_delay_s == 0:
            return
        self._actions_since_break += 1
        fatigue = self._actions_since_break / self._break_due_after
        p = self._break_hazard * min(fatigue, 2.5)
        if random.random() < p:
            pause = human_delay(45.0, sigma=0.5)
            print(f"{self.app.title()} session micro-break: {pause:.0f}s")
            self._actions_since_break = 0
            self._break_hazard = random.uniform(0.04, 0.14)
            self._break_due_after = random.uniform(12, 35)
            self.stop_event.wait(pause)

    @staticmethod
    def _label_metadata(profile) -> dict:
        return {"photo_count": len(profile.photos)}
