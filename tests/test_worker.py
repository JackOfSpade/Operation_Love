"""Worker + OpenerService tests with fakes — no Playwright/emulator/SDK/network."""
import math
import threading
import time
from types import SimpleNamespace

import pytest

from operation_love.costing import CostTracker, ModelPricing, Usage
from operation_love.drivers.base import (DatingAppDriver, DeckBlockedError, DriverClosed,
                                        ItemTargetingError)
from operation_love.drivers.adb import AdbError
from operation_love.opener.opener import (
    FIRST_ITEM_INDEX,
    INDEX_SPACE_MODEL_ITEMS,
    INDEX_SPACE_PROFILE_PHOTOS,
    ITEM_INDEX_ABSENT,
    GeminiAPIError,
    OpenerError,
    OpenerParseError,
    OpenerResult,
    REASON_PROMPT_BLOCKED,
)
from operation_love.opener.service import OpenerPick, OpenerService
from operation_love.limits import RateLimiter
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import Decision
from operation_love.ranker.profile_key import profile_key_from_identity
from operation_love.worker import Worker

PRICING = {"gemini-test-model": ModelPricing(input=5.0, output=25.0)}

# Liveness bound, not a performance bound: it exists only so a genuine hang fails these tests
# instead of hanging the whole suite forever. Widened 2026-08-22 when `python -m pytest` moved
# to one worker per core (pyproject.toml addopts `-n auto --dist loadgroup`), which measured a
# ~15x slowdown (0.33s idle vs 5.06s under load) on tests/test_concurrency.py's positive
# liveness waits of the same shape. Nothing about the property under test (did the background
# thread reach the expected state / finish?) depends on the exact number, so widening it loses
# nothing.
_LIVENESS_TIMEOUT_S = 15.0


def test_terminal_error_summary_keeps_multiline_adb_cause_and_retry_notes():
    """RunStatus must not reduce an AdbError to its trailing ``stderr:`` line."""
    from operation_love.worker import _terminal_error_summary

    exc = AdbError(
        ["adb", "-s", "pixel", "exec-out", "screencap", "-p"],
        "ADB command timed out after 10s",
        "error: device offline",
    )
    exc.add_note("screencap retry also timed out")

    summary = _terminal_error_summary(exc)

    assert summary.startswith("AdbError: ADB command timed out after 10s")
    assert "adb -s pixel exec-out screencap -p" in summary
    assert "stderr: error: device offline" in summary
    assert "notes: screencap retry also timed out" in summary
    assert "\n" not in summary


def test_terminal_error_summary_caps_untrusted_device_output():
    from operation_love.worker import _STATUS_ERROR_MAX_CHARS, _terminal_error_summary

    summary = _terminal_error_summary(RuntimeError("device replied " + "x" * 2_000))

    assert len(summary) == _STATUS_ERROR_MAX_CHARS
    assert summary.endswith("…")


def test_staged_opener_generation_context_uses_only_the_existing_pick_and_stage():
    from operation_love.worker import _staged_opener_generation_context

    pick = SimpleNamespace(
        _staged_record=SimpleNamespace(model="gemini-test-model"),
        index_space="model_items", referenced="two puppies", angle="a playful question",
        item_description="photo of two puppies")

    assert _staged_opener_generation_context(pick) == {
        "model": "gemini-test-model", "index_space": "model_items",
        "referenced": "two puppies", "angle": "a playful question",
        "item_description": "photo of two puppies",
    }
    assert _staged_opener_generation_context(None) is None


def test_worker_run_publishes_the_complete_adb_timeout_summary():
    """The generic terminal handler must use the helper, not revive traceback's last line."""
    from operation_love.status import RunStatus

    class TimeoutDriver(FakeDriver):
        def next_profile(self):
            exc = AdbError(
                ["adb", "-s", "pixel", "exec-out", "screencap", "-p"],
                "ADB command timed out after 10s", "transport stalled")
            exc.add_note("read-only screencap retry was exhausted")
            raise exc

    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", TimeoutDriver(1), FakeDecider("dislike"), None, FakeStore(), "run1",
           _Pacing(), threading.Event(), mode="auto", status=status).run()

    error = status.app_view("bumble")["app"]["error"]
    assert error.startswith("AdbError: ADB command timed out after 10s")
    assert "adb -s pixel exec-out screencap -p" in error
    assert "stderr: transport stalled" in error
    assert "notes: read-only screencap retry was exhausted" in error
    assert "\n" not in error


# --- fakes ---------------------------------------------------------------
class FakeDriver(DatingAppDriver):
    supports_interruptible_dislike = True

    def __init__(self, n):
        self.cards = [Profile(photos=[b"x"], bio=f"bio{i}") for i in range(n)]
        self.i = 0
        self.likes, self.dislikes, self.opened, self.closed = [], 0, False, False
        # Recorded on every like() call, alongside self.likes -- see like()'s own comment
        # for why these need their own history rather than folding into self.likes (which
        # stays a bare list of opener texts so every pre-existing `driver.likes == [...]`
        # assertion in this file keeps working unchanged).
        self.like_item_indexes = []
        self.like_model_item_indexes = []

    def open_session(self): self.opened = True
    def next_profile(self):
        if self.i >= len(self.cards):
            return None
        p = self.cards[self.i]
        self.i += 1
        return p
    # No `anchored_opener` keyword, deliberately: the driver's repair hatch -- a callback that
    # re-asked the model for text about whatever item the tap had actually landed on -- was
    # removed on 2026-08-12 (ops/OPENER-REDESIGN.md 5.6: rewriting the opener to match whatever
    # we hit is a substitution, and a targeting miss is now a stop). This signature tracks
    # base.Driver.like exactly, so a worker that started passing one again would fail here.
    def like(self, opener=None, item_index=None, *, model_item_index=None):
        self.likes.append(opener)
        self.like_item_indexes.append(item_index)
        # The OTHER index space, recorded separately for the same reason `like_item_indexes` is
        # separate from `likes`: since doc 5.5's counting navigation was wired the worker hands
        # a model item number straight through as `model_item_index` rather than converting it,
        # and a test that cannot see which of the two arguments was used cannot tell a navigated
        # like from a capture-order one.
        self.like_model_item_indexes.append(model_item_index)
    def dislike(self, *, should_stop=None):
        if should_stop is not None and should_stop():
            from operation_love.drivers.base import ActionCancelled
            raise ActionCancelled("test pass cancelled")
        self.dislikes += 1
    def out_of_profiles(self): return self.i >= len(self.cards)
    def close(self): self.closed = True


_PAYWALL_REASON = "Hinge is out of free likes for today — the Hinge+ upgrade screen is up"


class _BlockedDriver(FakeDriver):
    """An AUTO deck that is unavailable before any profile can be captured."""
    def __init__(self, n, reason=_PAYWALL_REASON):
        super().__init__(n)
        self.reason = reason
        self.blocked_calls = 0

    def blocked_reason(self):
        self.blocked_calls += 1
        return self.reason


class _BlockedAfterLikeDriver(FakeDriver):
    """The deck is healthy until the app rejects Send Like with a paywall."""
    accepts_opener = False

    def __init__(self):
        super().__init__(1)
        self.like_calls = 0

    def blocked_reason(self):
        return None

    def like(self, opener=None, item_index=None, *, model_item_index=None):
        self.like_calls += 1
        raise DeckBlockedError(_PAYWALL_REASON)


class _BlockedAndEmptyDriver(FakeDriver):
    """A blocked deck can also look empty; the block reason must take precedence."""
    def __init__(self):
        super().__init__(0)

    def blocked_reason(self):
        return _PAYWALL_REASON


class InterruptibleLikeDriver(FakeDriver):
    supports_interruptible_like_navigation = True

    def __init__(self, n):
        super().__init__(n)
        self.like_stop_callbacks = []

    def like(self, opener=None, item_index=None, *, model_item_index=None, should_stop=None):
        self.like_stop_callbacks.append(should_stop)
        return super().like(opener, item_index, model_item_index=model_item_index)


class ClosedDriver(FakeDriver):
    def __init__(self):
        super().__init__(1)
    def next_profile(self):
        raise DriverClosed("browser closed")


class RaisingLikeDriver(FakeDriver):
    """Hinge-style: halts on unexpected, and its like() raises (e.g. an unsent like)."""
    halt_on_error = True

    def __init__(self, n):
        super().__init__(n)
        self.snapshotted = []

    def like(self, opener=None, item_index=None, *, model_item_index=None):
        raise RuntimeError("like did not land")

    def snapshot_failure(self, exc):
        self.snapshotted.append((self._adb_live(), exc))

    def _adb_live(self):
        return not self.closed          # snapshot must run while the transport is still open


# A minimal duck-typed stand-in for drivers.item_identity.ProfileIdentity -- profile_key.
# profile_key_from_identity reads only `.known`/`.fingerprint`/`.grid` off whatever it is given
# (deliberately, see that module's own docstring), so a SimpleNamespace is exactly as valid an
# input as the real dataclass and keeps this file free of a cv2/numpy-carrying import.
_FAKE_IDENTITY = SimpleNamespace(known=True, fingerprint=(10, 20, 30), grid=(64, 16))
_FAKE_PROFILE_KEY = profile_key_from_identity(_FAKE_IDENTITY)
_UNKNOWN_IDENTITY = SimpleNamespace(known=False, fingerprint=None, grid=(64, 16))


class _IdentityDriver(FakeDriver):
    """A FakeDriver that also exposes current_profile_identity() (drivers/hinge.py's own
    optional hook), so worker.py's `_current_profile_key` has something to read.

    `identity_after_action`, when given, is what this driver answers AFTER like()/dislike() is
    called -- mirroring the real HingeDriver, whose own item index (and the identity it
    carries) is invalidated in EACH of those methods' `finally` clause the instant the action
    completes (doc 5.3). Defaulting it to a SENTINEL distinct from None lets a test assert the
    worker read the identity BEFORE the action, not after: if the worker read it late, the
    profile_key recorded would reflect this post-action value instead.
    """
    _POST_ACTION_UNSET = object()

    def __init__(self, n, identity, *, identity_after_action=_POST_ACTION_UNSET):
        super().__init__(n)
        self._identity = identity
        self._identity_after_action = (
            identity if identity_after_action is self._POST_ACTION_UNSET
            else identity_after_action)
        self._action_taken = False

    def current_profile_identity(self):
        return self._identity_after_action if self._action_taken else self._identity

    def like(self, opener=None, item_index=None, *, model_item_index=None):
        result = super().like(opener, item_index, model_item_index=model_item_index)
        self._action_taken = True
        return result

    def dislike(self, *, should_stop=None):
        result = super().dislike(should_stop=should_stop)
        self._action_taken = True
        return result


class FakeDecider:
    def __init__(self, decision="dislike"):
        self.decision = decision
    def decide(self, profile):
        return Decision(decision=self.decision, score=0.9, embedding=[0.1, 0.2], source="ranker")


class StopAfterDecide(FakeDecider):
    def __init__(self, stop_event):
        super().__init__("dislike")
        self.stop_event = stop_event
    def decide(self, profile):
        self.stop_event.set()
        return super().decide(profile)


class FakeOpenerClient:
    def __init__(self, cost_tokens=400):
        self.calls = 0
        self.cost_tokens = cost_tokens
        self.should_stops = []   # records should_stop from every call -- see BUG 1's tests
        self.items = []          # records items from every call -- doc 5.2's crop request shape
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        self.should_stops.append(should_stop)
        self.items.append(items)
        # item_index/index_space: a REALISTIC pick (the legacy frame shape's first item, in
        # the translatable profile-photos space), not the OpenerResult dataclass's own
        # ITEM_INDEX_ABSENT/INDEX_SPACE_MODEL_ITEMS defaults. Those defaults exist so a
        # careless construction fails safe (see OpenerResult.index_space's own docstring),
        # but "fails safe" now means the AUTO-loop refuses to call driver.like() at all (see
        # the INDEX SPACES section below) -- which would silently turn every test in this
        # file that just wants "a like happens, with an opener" into an untargeted-item stop
        # unrelated to what it's actually testing. Tests that DO want an absent/untranslatable
        # index construct their own OpenerPick directly (see _SequencedOpenerService) instead
        # of going through this fake.
        return OpenerResult(opener=f"hi {self.calls}", referenced="r",
                            usage=Usage(input_tokens=self.cost_tokens), model="gemini-test-model",
                            item_index=FIRST_ITEM_INDEX, index_space=INDEX_SPACE_PROFILE_PHOTOS)


class SlowOpenerClient(FakeOpenerClient):
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        time.sleep(0.05)
        return super().generate(profile, style, retry_hint, items=items,
                                should_stop=should_stop)


# --- opener clients that fail in the PER-CALL (not global-exhaustion) ways
# maybe_opener() can return None -- see service.py's maybe_opener docstring. OpenerError,
# a single sub-latch 400, and a single sub-latch transient failure each leave
# OpenerService.disabled False (a lone occurrence is below every latch threshold), which
# is exactly the case the worker's no-bare-like guard exists for. ParseErrorOpenerClient
# is different: OpenerParseError is now RETRIED internally by maybe_opener() (see its
# docstring), so a client that keeps failing every call, like this one, drives the retry
# loop all the way to exhaustion (disabled True) rather than leaving the service enabled.
class ParseErrorOpenerClient:
    """Model returned a billed response that didn't parse into a usable opener, every
    single attempt -- exercises maybe_opener()'s retry-until-exhausted path."""
    def __init__(self):
        self.calls = 0
        self.retry_hints = []
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        self.retry_hints.append(retry_hint)
        raise OpenerParseError("bad JSON in response body", Usage(input_tokens=10),
                               "gemini-test-model")


class OpenerErrorOpenerClient:
    """Per-profile content problem raised before/without a billed round trip (e.g. a
    corrupt captured photo that couldn't be decoded). NOT retried by maybe_opener()."""
    def __init__(self):
        self.calls = 0
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise OpenerError("Gemini opener: photo index 0 could not be decoded")


class SafetyBlockedOpenerClient:
    """Gemini accepted the call but blocked this profile's content before generation."""
    def __init__(self):
        self.calls = 0
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise OpenerParseError(
            "Gemini blocked the opener prompt before generating content "
            "(promptFeedback.blockReason=PROHIBITED_CONTENT)",
            Usage(input_tokens=10), "gemini-test-model",
            reason_code=REASON_PROMPT_BLOCKED)


class BadRequestOpenerClient:
    """A single HTTP 400 -- below _BAD_REQUEST_LATCH_THRESHOLD, so this alone must not
    disable the service, only skip this one profile's opener."""
    def __init__(self):
        self.calls = 0
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise GeminiAPIError(400, "INVALID_ARGUMENT", "bad image or request")


class TransientOpenerClient:
    """An unclassified exception (timeout/connection blip) -- below
    _TRANSIENT_LATCH_THRESHOLD, so this alone must not disable the service."""
    def __init__(self):
        self.calls = 0
    def generate(self, profile, style, retry_hint="", *, items=None,
                 should_stop=None,
                 skip_models=frozenset()):
        self.calls += 1
        raise RuntimeError("connection reset")


class FakeStore:
    def __init__(self):
        self.decisions, self.labels, self.profiles, self.spend, self.openers = [], [], [], [], []
        self.rejections = []
        self.opener_stamps, self.rejection_stamps = [], []
        # Full per-call keyword lineage (profile_id/decision/decision_source/
        # decision_created_at/model_item_index), captured separately from `openers` (text-only)
        # for the same reason `opener_stamps` is separate: a discard_opener test needs to read
        # back the `decision` a committed-vs-abandoned row was written with, without changing
        # what every pre-existing opener-count assertion reads off `openers`/`opener_stamps`.
        self.opener_rows = []
    def load_labels(self): return []
    def record_profile(self, run_id, app, profile_id, liked, source="manual", **k):
        self.profiles.append((app, profile_id, liked, k))
        return True
    def add_label(self, run_id, app, liked, embedding, source="manual", profile_id="", **k):
        self.labels.append((app, profile_id, liked, k))
    def record_decision(self, run_id, app, decision, score, source="auto", **_):
        self.decisions.append((app, decision, source))
    # Signature mirrors ranker/store.py's real record_opener EXACTLY, both trailing
    # parameters included: angle="" (doc 3.4/3.5: the model's free-text account of what its
    # opener is doing) and item_description="" (doc 5.7: the model's description of the ITEM
    # it picked to write about and to like). Deliberately spelled out rather than *a:
    # the service persists inside a non-fatal try/except, so a fake that drifts from the
    # real signature does not raise here -- it prints "Warning: failed to persist opener
    # spend record" and leaves self.openers empty, which surfaces as an unrelated-looking
    # count assertion failing further down. Keeping the parameters explicit means the
    # mismatch is at least visible in the traceback when it happens again.
    def record_opener(self, run_id, app, model, opener, referenced, angle="",
                      item_description="", *, prompt_sha256=None, **kw):
        self.openers.append(opener)
        # Captured separately from `openers` so a test can assert the prompt-era stamp reached
        # the sink without changing what every existing opener-count assertion reads.
        self.opener_stamps.append(prompt_sha256)
        self.opener_rows.append({"model": model, "opener": opener, "referenced": referenced,
                                 "angle": angle, "item_description": item_description,
                                 "prompt_sha256": prompt_sha256, **kw})
    # Signature mirrors ranker/store.py's real record_opener_rejection EXACTLY, the trailing
    # keyword-only prompt_sha256 included (2026-09-05 (b): the digest of the prompt era that
    # produced the rejected attempt). Spelled out rather than **kw for the same reason as
    # record_opener above: the service persists inside a non-fatal try/except, so a fake that
    # drifts from the real signature prints a warning and records nothing rather than raising.
    def record_opener_rejection(self, run_id, app, model, attempt, reason_code, reason, raw_opener,
                                *, prompt_sha256=None):
        self.rejections.append((app, model, attempt, reason_code, reason, raw_opener))
        self.rejection_stamps.append(prompt_sha256)
    def record_spend(self, run_id, model, usage, cost): self.spend.append(cost)
    def count_today(self, app):
        return sum(1 for row in self.decisions if row[0] == app and row[2] == "auto")
    def flush(self): pass
    def close(self): pass


class _Pacing:
    swipe_delay_s = 0.0


@pytest.mark.parametrize("mode", ["auto_testing", "", "other"])
def test_worker_rejects_retired_or_unknown_modes(mode):
    with pytest.raises(ValueError, match="mode"):
        Worker("bumble", FakeDriver(0), FakeDecider(), None, FakeStore(), "run1", _Pacing(),
               threading.Event(), mode=mode)


def _worker(driver, decider, service, store):
    return Worker("bumble", driver, decider, service, store, "run1", _Pacing(),
                  threading.Event(), mode="auto")


def test_capture_progress_callback_publishes_hinge_capture_detail_and_is_cleared():
    """A driver's long capture can refresh the Hub without gaining a control channel."""
    from operation_love.status import RunStatus

    class ProgressDriver(FakeDriver):
        supports_interruptible_capture = True

        def __init__(self):
            super().__init__(1)
            self.callbacks = []
            self.progress = None

        def set_capture_progress_callback(self, callback):
            self.callbacks.append(callback)
            self.progress = callback

        def next_profile(self, *, should_stop=None):
            assert should_stop is not None
            assert self.progress is not None
            self.progress("verifying photo item 2 of 3 for motion")
            return super().next_profile()

    status = RunStatus("run1", ["hinge"], min_labels=1, mode="training")
    driver = ProgressDriver()
    worker = Worker("hinge", driver, FakeDecider(), None, FakeStore(), "run1", _Pacing(),
                    threading.Event(), mode="training", status=status)

    assert worker._capture_profile("next_profile") is not None
    app = status.app_view("hinge")["app"]
    assert app["state"] == "scoring"
    assert app["detail"] is None
    assert callable(driver.callbacks[0])
    assert driver.callbacks[-1] is None


def test_training_replaces_capture_progress_before_the_gemini_opener_call():
    """A finished photo read must not masquerade as the active opener request."""
    from operation_love.status import RunStatus
    from operation_love.training_actions import TrainingActionBridge

    class ProgressDriver(FakeDriver):
        accepts_opener = True
        supports_training_decision = True

        def __init__(self):
            super().__init__(1)
            self.progress = None
            self.capture_updated_at = None

        def set_capture_progress_callback(self, callback):
            self.progress = callback

        def next_profile(self):
            assert self.progress is not None
            self.progress("verifying photo item 3 of 3 for motion")
            self.capture_updated_at = status.app_view("hinge")["app"]["updated_at"]
            return super().next_profile()

        def set_training_decision(self, _approval):
            pytest.fail("an empty opener must stop before any device action")

    class EmptyOpenerService:
        disabled = False
        stop_requested = False

        def __init__(self):
            self.during_request = None

        def maybe_opener(self, *_args, **_kwargs):
            self.during_request = status.app_view("hinge")["app"]
            return SimpleNamespace(text="")

    status = RunStatus("run1", ["hinge"], min_labels=1, mode="training")
    driver, service, stop = ProgressDriver(), EmptyOpenerService(), threading.Event()
    Worker("hinge", driver, FakeDecider("like"), service, FakeStore(), "run1", _Pacing(), stop,
           mode="training", status=status,
           training_action_bridge=TrainingActionBridge()).run()

    assert service.during_request is not None
    assert service.during_request["state"] == "scoring"
    assert service.during_request["detail"] == "generating a targeted opener"
    assert service.during_request["updated_at"] > driver.capture_updated_at
    # A terminal status never retains an in-flight capture or Gemini description.
    assert status.app_view("hinge")["app"]["detail"] is None


def test_worker_binds_opt_in_driver_debug_to_its_exact_run_id_before_opening():
    class BindingDriver(FakeDriver):
        def __init__(self):
            super().__init__(0)
            self.bound_run_id = None

        def bind_debug_run(self, run_id):
            self.bound_run_id = run_id

    driver = BindingDriver()
    _worker(driver, FakeDecider(), None, FakeStore())._bind_debug_run()
    assert driver.bound_run_id == "run1"


def test_worker_binds_opt_in_driver_to_the_service_prompt_era_before_opening():
    """The other half of the hinge.py actions.jsonl threading this discard path also fixes: a
    driver's LOCAL debug rows should be era-attributable exactly like the durable `openers`
    table already is (see Worker._bind_opener_prompt_stamp and
    HingeDriver.set_opener_prompt_sha256's docstrings for why a one-time, run-level bind is
    exactly as fresh as threading the value through every like() call would have been)."""
    class BindingDriver(FakeDriver):
        def __init__(self):
            super().__init__(0)
            self.bound_prompt_sha256 = "unset"

        def set_opener_prompt_sha256(self, prompt_sha256):
            self.bound_prompt_sha256 = prompt_sha256

    driver = BindingDriver()
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider(), svc, store)._bind_opener_prompt_stamp()
    assert driver.bound_prompt_sha256 == svc.prompt_sha256
    assert driver.bound_prompt_sha256 is not None


def test_worker_binds_none_prompt_era_when_there_is_no_opener_service():
    class BindingDriver(FakeDriver):
        def __init__(self):
            super().__init__(0)
            self.bound_prompt_sha256 = "unset"

        def set_opener_prompt_sha256(self, prompt_sha256):
            self.bound_prompt_sha256 = prompt_sha256

    driver = BindingDriver()
    _worker(driver, FakeDecider(), None, FakeStore())._bind_opener_prompt_stamp()
    assert driver.bound_prompt_sha256 is None


def test_worker_run_binds_the_opener_prompt_era_before_any_profile_is_read(monkeypatch):
    """Integration pin: run() itself calls the bind (mirroring bind_debug_run/set_opener_enabled
    right beside it), not just some inner loop -- so a driver lacking a captured profile still
    gets bound before the session closes."""
    class BindingDriver(FakeDriver):
        def __init__(self):
            super().__init__(0)
            self.bound_prompt_sha256 = "unset"

        def set_opener_prompt_sha256(self, prompt_sha256):
            self.bound_prompt_sha256 = prompt_sha256

    driver = BindingDriver()
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("dislike"), svc, store).run()
    assert driver.bound_prompt_sha256 == svc.prompt_sha256


def test_worker_never_calls_the_prompt_era_hook_on_a_driver_that_lacks_it():
    """A driver (or legacy/third-party opener service) without this optional hook must be
    completely unaffected -- exactly like bind_debug_run/set_opener_enabled above it."""
    driver = FakeDriver(1)   # no set_opener_prompt_sha256 method at all
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("dislike"), svc, store).run()   # must not raise


# --- OpenerService (global budget) --------------------------------------
def test_budget_caps_openers_globally():
    tracker = CostTracker(PRICING, run_budget_usd=0.001)   # 400 input tok = $0.002 > cap
    client = FakeOpenerClient(cost_tokens=400)
    svc = OpenerService(client, tracker, FakeStore(), "style")
    assert svc.maybe_opener("r", "bumble", Profile()).text == "hi 1"   # first allowed
    assert svc.maybe_opener("r", "hinge", Profile()) is None           # second over budget
    assert client.calls == 1 and svc.stop_requested is True


def test_budget_caps_openers_across_concurrent_workers():
    tracker = CostTracker(PRICING, run_budget_usd=0.001)   # first call spends past cap
    client = SlowOpenerClient(cost_tokens=400)
    store = FakeStore()
    svc = OpenerService(client, tracker, store, "style")
    results = []

    def call(app):
        pick = svc.maybe_opener("r", app, Profile())
        results.append(pick.text if pick else None)

    threads = [threading.Thread(target=call, args=(app,)) for app in ("bumble", "hinge")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_LIVENESS_TIMEOUT_S)

    assert all(not t.is_alive() for t in threads)
    assert client.calls == 1
    assert sorted(results, key=lambda x: x or "") == [None, "hi 1"]
    assert len(store.spend) == 1 and len(store.openers) == 1
    assert svc.stop_requested is True


class _CalibratedDriver(FakeDriver):
    """A driver whose app has a real, measured think_time_s() calibration (Hinge)."""
    think_time_calibrated = True


def test_pace_scales_think_time_by_configured_anchor(monkeypatch):
    from operation_love import worker as worker_mod

    w = _worker(_CalibratedDriver(0), FakeDecider("like"), FakeOpenerClient(), FakeStore())
    monkeypatch.setattr(worker_mod, "think_time_s", lambda decision: 10.0)

    class _Scaled:
        swipe_delay_s = worker_mod._THINK_TIME_BASELINE_S * 0.5   # -> scale 0.5

    w.pacing = _Scaled()
    waited = []
    monkeypatch.setattr(w.stop_event, "wait", lambda s: waited.append(s))
    w._pace("like")
    assert waited == [5.0]                    # 10.0 * 0.5


def test_pace_uses_flat_anchor_for_a_driver_without_calibrated_pacing(monkeypatch):
    # e.g. Bumble: no measured think-time model for this app, so pacing stays the
    # original decision-agnostic anchor + log-normal spread, not a borrowed one.
    from operation_love import worker as worker_mod

    w = _worker(FakeDriver(0), FakeDecider("like"), FakeOpenerClient(), FakeStore())
    monkeypatch.setattr(worker_mod, "think_time_s",
                        lambda decision: (_ for _ in ()).throw(AssertionError("must not be called")))
    monkeypatch.setattr(worker_mod, "human_delay", lambda s: s * 2)
    waited = []
    monkeypatch.setattr(w.stop_event, "wait", lambda s: waited.append(s))
    w._pace("like")
    assert waited == [w.pacing.swipe_delay_s * 2]


@pytest.mark.parametrize("value", [-1.0, 0.5, math.nan, math.inf, 1e300, True, "3.5"])
def test_pace_rejects_invalid_direct_pacing_for_both_driver_branches(value):
    """Direct Worker construction must enforce the same finite 0-or-[1,3600] contract."""
    class _InvalidPacing:
        swipe_delay_s = value

    flat_worker = _worker(FakeDriver(0), FakeDecider("like"), FakeOpenerClient(), FakeStore())
    flat_worker.pacing = _InvalidPacing()
    with pytest.raises(ValueError, match="swipe_delay_s"):
        flat_worker._pace("like")

    calibrated_worker = _worker(_CalibratedDriver(0), FakeDecider("like"),
                                FakeOpenerClient(), FakeStore())
    calibrated_worker.pacing = _InvalidPacing()
    with pytest.raises(ValueError, match="swipe_delay_s"):
        calibrated_worker._pace("like")


def test_pace_maps_dislike_to_the_pass_think_time_bucket(monkeypatch):
    from operation_love import worker as worker_mod

    w = _worker(_CalibratedDriver(0), FakeDecider("like"), FakeOpenerClient(), FakeStore())
    seen = []
    monkeypatch.setattr(worker_mod, "think_time_s", lambda decision: seen.append(decision) or 10.0)

    class _Scaled:
        swipe_delay_s = worker_mod._THINK_TIME_BASELINE_S   # -> scale 1.0

    w.pacing = _Scaled()
    monkeypatch.setattr(w.stop_event, "wait", lambda s: None)
    w._pace("dislike")           # decider decisions are "like"/"dislike", never "pass"
    assert seen == ["pass"]


def test_pace_uses_session_context_when_the_auto_loop_supplies_profile():
    class _Policy:
        def __init__(self):
            self.calls = []

        def post_action_delay_s(self, decision, profile, score, *, scale):
            self.calls.append((decision, profile, score, scale))
            return 8.25

    class _Pacing2:
        swipe_delay_s = 7.0

    class _Stop:
        def __init__(self): self.waits = []
        def wait(self, seconds): self.waits.append(seconds)

    w = Worker.__new__(Worker)
    w.driver = _CalibratedDriver(0)
    w.pacing = _Pacing2()
    w.stop_event = _Stop()
    w._auto_policy = policy = _Policy()
    profile = Profile(photos=[b"x"], bio="hello")

    w._pace("like", profile=profile, score=0.73)

    assert policy.calls == [("like", profile, 0.73, 2.0)]
    assert w.stop_event.waits == [8.25]


def test_auto_session_policy_is_installed_and_counts_only_landed_final_action(monkeypatch):
    """Worker owns session-policy lifecycle; the driver only receives its optional hook."""
    from operation_love import worker as worker_mod

    made = []

    class _Adjusted:
        def __init__(self, decision): self.decision = decision

    class _Policy:
        def __init__(self, *, base_threshold):
            self.base_threshold = base_threshold
            self.applied, self.landed = [], []
            made.append(self)

        def apply_decision(self, decision, profile):
            self.applied.append((decision, profile))
            # Simulate the real policy's only permitted decision change: a marginal
            # model like becomes a pass before the action and rate-limit checks.
            return _Adjusted(Decision("dislike", decision.score, decision.embedding,
                                      "ranker_contextual"))

        def record_landed_action(self, decision):
            self.landed.append(decision)

    class _PolicyDriver(FakeDriver):
        def set_auto_session_policy(self, policy):
            self.installed_policy = policy

    monkeypatch.setattr(worker_mod, "AutoSessionPolicy", _Policy)
    driver = _PolicyDriver(1)
    decider = FakeDecider("like")
    decider.model = type("Model", (), {"threshold": 0.61})()
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")

    Worker("bumble", driver, decider, svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    policy = made[0]
    assert policy.base_threshold == 0.61
    assert driver.installed_policy is policy
    assert len(policy.applied) == 1
    assert policy.landed == ["dislike"]
    assert driver.likes == [] and driver.dislikes == 1
    assert store.decisions == [("bumble", "dislike", "auto")]


# --- Session micro-break: fatigue-hazard model --------------------------
class _NonZeroPacing:
    swipe_delay_s = 3.5


def _break_worker():
    """A worker with nonzero pacing (breaks are a no-op at swipe_delay_s == 0) and
    hand-set fatigue state, so each test controls fatigue directly rather than
    depending on how many swipes happened to run first."""
    w = _worker(FakeDriver(0), FakeDecider("like"), FakeOpenerClient(), FakeStore())
    w.pacing = _NonZeroPacing()
    w._actions_since_break = 0
    w._break_hazard = 0.08
    w._break_due_after = 20.0
    return w


def test_session_break_low_fatigue_effectively_never_fires(monkeypatch):
    """Right after a reset (fatigue ~ 1/20 = 0.05 -> p = 0.08*0.05 = 0.004), even a
    fairly low random() draw must not trigger a break -- a fixed 8%-per-swipe roll
    (the old behavior) would have fired here."""
    from operation_love import worker as worker_mod
    w = _break_worker()
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.01)
    waited = []
    monkeypatch.setattr(w.stop_event, "wait", lambda s: waited.append(s))
    w._maybe_session_break()
    assert waited == []                    # no break fired
    assert w._actions_since_break == 1      # fatigue counter still advances


def test_session_break_fires_once_fatigue_builds(monkeypatch):
    """The SAME random() draw that doesn't fire at low fatigue must fire once
    actions-since-break has run well past _break_due_after (fatigue capped at 2.5x
    -> p = 0.08*2.0 = 0.16 here) -- proving the hazard actually ramps with fatigue
    rather than staying at one constant rate."""
    from operation_love import worker as worker_mod
    w = _break_worker()
    w._actions_since_break = 39            # -> 40 after increment; fatigue = 40/20 = 2.0
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.1)   # between 0.004 and 0.16
    monkeypatch.setattr(worker_mod, "human_delay", lambda anchor, sigma=None: 42.0)
    waited = []
    monkeypatch.setattr(w.stop_event, "wait", lambda s: waited.append(s))
    w._maybe_session_break()
    assert waited == [42.0]
    assert w._actions_since_break == 0      # reset after firing


def test_session_break_hazard_is_capped_after_extreme_fatigue(monkeypatch):
    from operation_love import worker as worker_mod
    w = _break_worker()
    w._actions_since_break = 999
    # Capped p = .08 * 2.5 = .20, so .21 must not fire. Without the cap the raw
    # fatigue multiplier makes p > 1 and every sufficiently old stretch breaks.
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.21)
    waited = []
    monkeypatch.setattr(w.stop_event, "wait", lambda s: waited.append(s))
    w._maybe_session_break()
    assert waited == []
    assert w._actions_since_break == 1000
    # The ceiling is still a fatigue hazard, not the old flat 8% roll: a draw between
    # .08 and the capped .20 must fire once this stretch is extremely old.
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.15)
    monkeypatch.setattr(worker_mod, "human_delay", lambda anchor, sigma=None: 42.0)
    w._maybe_session_break()
    assert waited == [42.0]


def test_session_break_rerolls_hazard_and_interval_after_firing(monkeypatch):
    """After a break fires, both _break_hazard and _break_due_after must be redrawn
    (not left at the same value for the whole run) -- distinguishable sentinels from
    random.uniform prove the specific new attrs get the specific new draws, in the
    hazard-then-interval order the implementation calls them."""
    from operation_love import worker as worker_mod
    w = _break_worker()
    w._actions_since_break = 39
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.0)   # always fires
    monkeypatch.setattr(worker_mod, "human_delay", lambda anchor, sigma=None: 1.0)
    monkeypatch.setattr(w.stop_event, "wait", lambda s: None)
    draws = iter([0.1234, 27.5])           # first call -> hazard, second -> due_after
    monkeypatch.setattr(worker_mod.random, "uniform", lambda a, b: next(draws))
    w._maybe_session_break()
    assert w._break_hazard == 0.1234
    assert w._break_due_after == 27.5


def test_session_break_pause_is_lognormal_not_flat_uniform(monkeypatch):
    """The pause duration comes from human_delay(45.0, sigma=0.5) -- a log-normal
    draw -- not the old flat random.uniform(20, 90)."""
    from operation_love import worker as worker_mod
    w = _break_worker()
    w._actions_since_break = 39
    monkeypatch.setattr(worker_mod.random, "random", lambda: 0.0)   # always fires
    seen = {}

    def _fake_human_delay(anchor, sigma=None):
        seen["args"] = (anchor, sigma)
        return 7.0

    monkeypatch.setattr(worker_mod, "human_delay", _fake_human_delay)
    monkeypatch.setattr(worker_mod.random, "uniform", lambda a, b: 0.09)
    waited = []
    monkeypatch.setattr(w.stop_event, "wait", lambda s: waited.append(s))
    w._maybe_session_break()
    assert seen["args"] == (45.0, 0.5)
    assert waited == [7.0]


def test_auto_loop_initializes_session_break_fatigue_state(monkeypatch):
    """_auto_loop must set up the fatigue instance state (not just _maybe_session_break
    assuming it exists) so a fresh run's first micro-break check doesn't crash on a
    missing attribute."""
    driver = FakeDriver(2)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    stop_event = threading.Event()
    monkeypatch.setattr(stop_event, "wait", lambda s=None: None)   # keep the test instant
    w = Worker("bumble", driver, FakeDecider("dislike"), svc, store, "run1", _NonZeroPacing(),
               stop_event, mode="auto")
    w.run()
    assert w._actions_since_break >= 0
    assert 0.04 <= w._break_hazard <= 0.14
    assert 12 <= w._break_due_after <= 35
    assert driver.dislikes == 2
    assert stop_event.is_set() is False


def test_unlimited_limiter_does_not_query_irrelevant_daily_history():
    class _NoDailyReadStore(FakeStore):
        def count_today(self, app):
            raise AssertionError("count_today must not run without max_per_day")

    driver = FakeDriver(2)
    store = _NoDailyReadStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider("dislike"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=RateLimiter()).run()
    assert driver.dislikes == 2


# --- Worker loop ---------------------------------------------------------
def test_worker_dislikes_whole_deck():
    driver = FakeDriver(3)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")  # openers disabled
    _worker(driver, FakeDecider("dislike"), svc, store).run()
    assert driver.dislikes == 3 and driver.likes == []
    # AUTO mode is pure inference: decisions are logged, but NO training labels/profiles
    # are saved (training data comes only from Hub-reviewed Training decisions).
    assert len(store.decisions) == 3
    assert all(source == "auto" for _, _, source in store.decisions)
    assert store.labels == [] and store.profiles == []
    assert driver.opened and driver.closed


def test_worker_likes_with_openers():
    driver = FakeDriver(3)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert driver.likes == ["hi 1", "hi 2", "hi 3"]
    assert len(store.openers) == 3 and len(store.spend) == 3


def test_auto_stages_then_commits_an_opener_only_after_like_returns():
    driver = FakeDriver(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert driver.likes == ["hi 1"]
    assert len(store.openers) == 1 and len(svc.recent_openers_snapshot()) == 1


def test_auto_forwards_driver_presend_evidence_only_after_like_lands():
    evidence = {"frame": b"pre-send", "evidence_id": "evidence-id"}

    class EvidenceDriver(FakeDriver):
        def landed_auto_opener_evidence(self):
            assert self.likes == ["hi 1"]
            return evidence

    class EvidenceStore(FakeStore):
        def __init__(self):
            super().__init__()
            self.opener_evidence = []

        def record_opener_send_evidence(self, *args, **kwargs):
            self.opener_evidence.append((args, kwargs))

    store = EvidenceStore()
    service = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(EvidenceDriver(1), FakeDecider("like"), service, store).run()

    assert len(store.opener_evidence) == 1
    assert store.opener_evidence[0][1]["evidence"] == evidence


def test_auto_persists_landed_decision_before_committing_its_staged_opener():
    """A durable opener must always have a preceding durable Like decision to join to.

    This ordering is the cleanup/reporting invariant: generation is only a draft, a returned
    driver.like is the physical boundary, and decision storage establishes the acted-on fact
    before the draft becomes an opener row.
    """
    events = []

    class OrderedStore(FakeStore):
        def record_decision(self, *args, **kwargs):
            events.append("decision")
            kwargs.pop("profile_id", None)
            kwargs.pop("created_at", None)
            return super().record_decision(*args, **kwargs)

        def record_opener(self, *args, **kwargs):
            events.append("opener")
            for key in ("profile_id", "decision", "decision_source", "decision_created_at",
                        "model_item_index"):
                kwargs.pop(key, None)
            return super().record_opener(*args, **kwargs)

    driver = FakeDriver(1)
    store = OrderedStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert events == ["decision", "opener"]
    assert store.decisions == [("bumble", "like", "auto")]
    assert store.openers == ["hi 1"]
    # OrderedStore strips only the action-lineage keywords, so the prompt-era stamp added on
    # 2026-09-05 (b) still reaches the sink through this wrapper: an intermediate store that
    # forwards **kwargs must not need editing every time a column is added, and the staged
    # commit path is the one that writes every AUTO opener row.
    assert store.opener_stamps == [svc.prompt_sha256]


def test_auto_keeps_legacy_opener_service_without_stage_keyword_compatible():
    """The stage/commit protocol is additive; an old injected service must not crash AUTO."""
    class LegacyService:
        disabled = False
        stop_requested = False

        def maybe_opener(self, run_id, app, profile, *, items=None, should_stop=None):
            return OpenerPick("legacy opener", index=1,
                              index_space=INDEX_SPACE_PROFILE_PHOTOS)

        def commit_opener(self, pick, **_):
            raise AssertionError("legacy service must not receive a staged commit")

    driver = FakeDriver(1)
    store = FakeStore()
    Worker("bumble", driver, FakeDecider("like"), LegacyService(), store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.likes == ["legacy opener"]
    assert store.decisions == [("bumble", "like", "auto")]


def test_auto_targeting_failure_discards_the_staged_opener_as_never_sent():
    """A targeting refusal must not leave the survivorship-biased gap this discard path fixes:
    the draft is still not COMMITTED as a Like (recent_openers_snapshot, the live diagnostic
    view, stays empty -- unchanged from before this discard path existed), but the durable
    `openers` table now gets a `decision="never_sent"` row instead of nothing at all."""
    driver = _TargetingMissDriver(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert svc.recent_openers_snapshot() == []
    assert store.openers == ["hi 1"]
    assert len(store.opener_rows) == 1
    row = store.opener_rows[0]
    assert row["decision"] == "never_sent"
    assert row["decision_source"] == "auto"
    assert row["prompt_sha256"] == svc.prompt_sha256


def test_auto_post_send_paywall_discards_the_staged_opener_as_never_sent():
    class _PaywallWithOpener(_BlockedAfterLikeDriver):
        accepts_opener = True
    driver = _PaywallWithOpener()
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert svc.recent_openers_snapshot() == []
    assert store.openers == ["hi 1"]
    assert store.opener_rows[0]["decision"] == "never_sent"
    assert store.opener_rows[0]["decision_source"] == "auto"


def test_auto_generic_like_failure_discards_the_staged_opener_as_never_sent():
    driver = RaisingLikeDriver(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert svc.recent_openers_snapshot() == []
    assert store.openers == ["hi 1"]
    assert store.opener_rows[0]["decision"] == "never_sent"
    assert store.opener_rows[0]["decision_source"] == "auto"


def test_auto_never_sent_discard_when_the_model_names_no_item():
    """The "not targeted" refusal (opener generated no item number to target) is a clean stop,
    not a device action -- and, exactly like the targeting/exception abandonment paths above,
    must not leave this profile's real, billed draft silently missing from the durable table."""
    class _NoItemClient:
        def __init__(self):
            self.calls = 0

        def generate(self, profile, style, retry_hint="", *, items=None, should_stop=None,
                     skip_models=frozenset()):
            self.calls += 1
            return OpenerResult(opener=f"hi {self.calls}", referenced="r",
                                usage=Usage(input_tokens=400), model="gemini-test-model",
                                item_index=ITEM_INDEX_ABSENT, index_space=INDEX_SPACE_MODEL_ITEMS)

    driver = FakeDriver(1)
    store = FakeStore()
    svc = OpenerService(_NoItemClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert driver.likes == []
    assert svc.recent_openers_snapshot() == []
    assert store.openers == ["hi 1"]
    assert store.opener_rows[0]["decision"] == "never_sent"
    assert store.opener_rows[0]["decision_source"] == "auto"
    assert store.opener_rows[0]["prompt_sha256"] == svc.prompt_sha256


def test_auto_never_sent_discard_on_run_budget_exhaustion():
    """Same profile as test_worker_stops_before_bare_like_when_opener_budget_exhausts: the call
    that pushes the run over budget can still return a fully-formed staged draft for THIS
    profile (see maybe_opener's docstring -- exhaustion is checked after persisting spend, not
    before returning). AUTO correctly refuses to act on it (stop_requested wins before like()
    is ever called), but the draft itself was real, billed model output and must not vanish."""
    driver = FakeDriver(5)
    store = FakeStore()
    client = FakeOpenerClient(cost_tokens=400)  # $0.002/call
    svc = OpenerService(client, CostTracker(PRICING, run_budget_usd=0.001), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert client.calls == 1
    assert driver.likes == []
    assert store.opener_rows[0]["decision"] == "never_sent"
    assert store.opener_rows[0]["decision_source"] == "auto"
    assert store.opener_rows[0]["prompt_sha256"] == svc.prompt_sha256


def test_auto_dislike_never_generates_or_discards_an_opener_draft():
    """AUTO only ever asks the provider for an opener inside the `if d.decision == "like":`
    gate (see the module's opener-generation block) -- so a plain Dislike never has a staged
    draft to write down at all. discard_opener/commit_opener are both simply never called, and
    this is not a survivorship gap: unlike the never-sent cases above (a Like decision whose
    draft could not be sent), there is no draft here for this profile to have discarded."""
    driver = FakeDriver(1)
    store = FakeStore()
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("dislike"), svc, store).run()
    assert client.calls == 0
    assert store.openers == []
    assert store.decisions == [("bumble", "dislike", "auto")]


def test_auto_committed_like_decision_and_source_are_unchanged():
    """Pin: this discard path is purely additive. A landed Like's durable row still carries
    exactly decision="like"/decision_source="auto" -- byte-identical to before discard_opener
    existed -- with a real profile_id/decision_created_at lineage and the generation-time
    prompt_sha256, none of which this change touches."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert driver.likes == ["hi 1"]
    assert len(store.opener_rows) == 1
    row = store.opener_rows[0]
    assert row["decision"] == "like"
    assert row["decision_source"] == "auto"
    assert isinstance(row["profile_id"], str) and row["profile_id"]
    assert isinstance(row["decision_created_at"], float)
    assert row["prompt_sha256"] == svc.prompt_sha256


# ---------------------------------------------------------------------------------------
# profile_key (2026-09-06): the stable, cross-time attribution key (ranker/profile_key.py),
# read off the driver's optional current_profile_identity() hook and carried through to
# whichever of commit_opener/discard_opener this profile's outcome reaches. Exercised through
# the REAL OpenerService + FakeStore, exactly like the lineage test just above.
# ---------------------------------------------------------------------------------------

def test_auto_committed_opener_carries_a_profile_key_from_the_driver():
    driver = _IdentityDriver(1, _FAKE_IDENTITY)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert driver.likes == ["hi 1"]
    assert store.opener_rows[0]["profile_key"] == _FAKE_PROFILE_KEY
    assert _FAKE_PROFILE_KEY  # sanity: a known identity really does hash to something


def test_auto_discarded_draft_carries_the_same_profile_key_a_commit_would_have():
    """A never_sent discard (here: a targeting refusal) must be attributable to exactly the
    same profile a landed Like would have been -- the whole point of threading profile_key
    onto BOTH halves of the staged-opener lifecycle (see OpenerService.discard_opener's own
    docstring)."""
    driver = _TargetingMissDriver(1, identity=_FAKE_IDENTITY)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert store.opener_rows[0]["decision"] == "never_sent"
    assert store.opener_rows[0]["profile_key"] == _FAKE_PROFILE_KEY


def test_auto_committed_opener_profile_key_is_empty_when_the_driver_has_no_identity_hook():
    """Every non-Hinge driver today (and any driver predating this hook): current_profile_identity
    is simply absent, and the row must honestly say "no key could be derived" ("") rather than
    fabricate one."""
    driver = FakeDriver(1)
    assert not hasattr(driver, "current_profile_identity")
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert driver.likes == ["hi 1"]
    assert store.opener_rows[0]["profile_key"] == ""


def test_auto_committed_opener_profile_key_is_empty_when_identity_is_unknown():
    """A driver that DOES enumerate but could not fingerprint this particular profile (no
    identity band, an unreadable header, ...) reports `known=False` -- also "", never a
    placeholder hash of nothing."""
    driver = _IdentityDriver(1, _UNKNOWN_IDENTITY)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert driver.likes == ["hi 1"]
    assert store.opener_rows[0]["profile_key"] == ""


def test_auto_profile_key_is_captured_before_the_like_action_invalidates_it():
    """THE CRITICAL TIMING PROPERTY: HingeDriver's own item index (and the identity fingerprint
    it carries) is invalidated in like()'s own `finally` clause the instant the physical action
    completes (doc 5.3) -- so a worker that read identity AFTER driver.like() returns would
    always see None/"" instead of the real key. This driver answers _FAKE_IDENTITY before the
    action and None after; a passing test proves the worker captured the key on the correct
    side of that boundary."""
    driver = _IdentityDriver(1, _FAKE_IDENTITY, identity_after_action=None)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert driver.likes == ["hi 1"]
    assert store.opener_rows[0]["profile_key"] == _FAKE_PROFILE_KEY


def test_auto_never_sent_discard_store_failure_does_not_change_the_targeting_stop_outcome():
    """The exact requirement this telemetry was built under, exercised through the REAL
    Worker + OpenerService (not a fake service that might not replicate the real swallow): a
    store outage while discarding an abandoned draft must not turn a clean, correctly-diagnosed
    targeting refusal into a crash/error state, and must not stop the durable row from simply
    being absent (the write genuinely failed) rather than corrupting anything else."""
    from operation_love.status import RunStatus

    class _RaisingOnDiscardStore(FakeStore):
        def record_opener(self, *args, **kwargs):
            if kwargs.get("decision") == "never_sent":
                raise RuntimeError("simulated store outage")
            return super().record_opener(*args, **kwargs)

    driver = _TargetingMissDriver(1)
    store = _RaisingOnDiscardStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "stopped"
    assert app["stop_kind"] == "targeting"
    assert app["error"] is None
    assert store.openers == []          # the write itself failed -- nothing durable landed


def test_auto_commits_staged_opener_when_stop_arrives_after_like_lands():
    stop = threading.Event()
    class _LandedThenStop(FakeDriver):
        def like(self, *args, **kwargs):
            super().like(*args, **kwargs)
            stop.set()
    driver = _LandedThenStop(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(), stop,
           mode="auto").run()
    assert driver.likes == ["hi 1"]
    assert len(store.openers) == 1 and len(svc.recent_openers_snapshot()) == 1


def test_auto_stop_during_opener_generation_never_starts_like_but_discards_the_draft():
    """A provider result that returns after Stop is billed but cannot start a device action.

    The service deliberately lets an already-on-the-wire request finish.  This pins the Worker
    boundary that distinguishes that unavoidable provider completion from a Like: the staged
    draft must remain absent from the COMMITTED diagnostics buffer (recent_openers_snapshot,
    which is reserved for landed Likes), but the durable opener table now gets a
    `decision="never_sent"` discard row rather than nothing at all -- this profile's draft did
    real, billed work and would otherwise vanish from every measurement of opener quality.
    """
    stop = threading.Event()

    class _StopsAfterGenerating(FakeOpenerClient):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            stop.set()
            return result

    driver = FakeDriver(1)
    store = FakeStore()
    service = OpenerService(_StopsAfterGenerating(), CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider("like"), service, store, "run1", _Pacing(), stop,
           mode="auto").run()

    assert driver.likes == []
    assert store.decisions == []
    assert service.recent_openers_snapshot() == []
    assert store.openers == ["hi 1"]
    assert store.opener_rows[0]["decision"] == "never_sent"
    assert store.opener_rows[0]["decision_source"] == "auto"
    # The request had already reached the provider, so its spend remains accountable.
    assert len(store.spend) == 1


def test_worker_passes_stop_callback_only_to_interruptible_like_navigation():
    driver = InterruptibleLikeDriver(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, 5), store, "style")
    worker = _worker(driver, FakeDecider("like"), svc, store)
    worker.run()
    assert len(driver.like_stop_callbacks) == 1
    assert driver.like_stop_callbacks[0].__self__ is worker.stop_event
    assert driver.like_stop_callbacks[0]() is False


def test_worker_skips_openers_when_driver_declines():
    # Bumble-style driver: no swipe-time opener -> never call Gemini (no wasted spend).
    driver = FakeDriver(3)
    driver.accepts_opener = False
    store = FakeStore()
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    assert driver.likes == [None, None, None]      # liked, but with no opener
    assert client.calls == 0 and store.openers == [] and store.spend == []


def test_worker_stops_before_bare_like_when_opener_budget_exhausts():
    driver = FakeDriver(5)
    store = FakeStore()
    client = FakeOpenerClient(cost_tokens=400)  # $0.002/call
    svc = OpenerService(client, CostTracker(PRICING, run_budget_usd=0.001), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()
    # The call itself can push the global cap over budget. AUTO must stop before
    # sending a bare like for that same profile, rather than substituting a like
    # with no opener after the provider service requested a stop.
    assert client.calls == 1
    assert driver.likes == []
    assert store.decisions == []
    assert driver.closed


def test_auto_mode_stop_reason_is_visible_in_status_when_opener_budget_exhausts():
    """WS-opener-reason: the owner rule is that the hub must show WHY an auto run stopped,
    not just that it did. An opener-exhaustion stop used to leave AppStatus at the default
    state='stopped' with no reason anywhere -- rendering identically to an operator clicking
    Stop. This pins the whole path: OpenerService records the cause (see
    test_opener_service.py) and the worker publishes it into the shared status the hub
    reads, on the SAME budget-exhaustion trigger as
    test_worker_stops_before_bare_like_when_opener_budget_exhausts above."""
    from operation_love.status import RunStatus

    driver = FakeDriver(5)
    store = FakeStore()
    client = FakeOpenerClient(cost_tokens=400)  # $0.002/call
    svc = OpenerService(client, CostTracker(PRICING, run_budget_usd=0.001), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "stopped"
    assert app["stop_reason"] == "run budget reached"
    # stop_kind disambiguates stop_reason's SOURCE now that the deck-blocked check (added
    # 2026-08-11, see the DECK_BLOCKED section below) also populates stop_reason -- an
    # OpenerService-triggered stop must publish "opener", not the new "deck_blocked".
    assert app["stop_kind"] == "opener"


# ---------------------------------------------------------------------------------------
# COMPLETED RULE: in AUTO mode on an opener-capable app, a like is normally sent WITH its
# opener or not sent at all. A provider safety block is the narrow exception: AUTO already
# chose Like, so it proceeds without a comment. There is no configuration that broadens that
# exception (the old
# budget.on_exhausted="swipe_without_opener" mode was removed outright; see service.py's
# module docstring). The tests above cover the pre-existing GLOBAL-exhaustion guard
# (stop_requested); these cover every other way maybe_opener() can return None -- a
# per-call failure that leaves OpenerService still enabled (see service.py's maybe_opener
# docstring and last_skip_reason), plus the case where a bad AI response keeps failing
# through every retry attempt and the SERVICE itself ends up exhausted. Each must halt
# before driver.like(), record no decision for the abandoned profile, and publish a
# stop_reason naming the actual cause (not a generic line). The dedicated safety-block test
# pins the opposite result for that one structured outcome.
# ---------------------------------------------------------------------------------------

def test_worker_halts_before_bare_like_when_the_opener_ultimately_fails_every_retry():
    """THE OWNER'S RULE: a bad AI response is retried, never sent bare, and if it is STILL
    bad after max_attempts (5, the OpenerService default) that means something is wrong and
    the whole run stops. ParseErrorOpenerClient fails every single call, so the worker's one
    maybe_opener() call drives OpenerService's retry loop out to full exhaustion -- this is
    the "ultimately fails" case, distinct from the single-failure-but-still-enabled cases
    below (OpenerError / a single sub-latch 400 / a single sub-latch transient failure)."""
    driver = FakeDriver(3)
    store = FakeStore()
    client = ParseErrorOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert client.calls == svc.max_attempts == 5   # every retry attempt was actually made
    assert driver.likes == []                       # no bare like sent
    assert store.decisions == []                     # no decision recorded for the abandoned profile
    assert svc.disabled is True and svc.stop_requested is True   # the service gave up for the run
    assert svc.exhausted_reason and "bad JSON in response body" in svc.exhausted_reason
    assert driver.closed


def test_worker_stops_before_bare_like_on_opener_error():
    driver = FakeDriver(3)
    store = FakeStore()
    client = OpenerErrorOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert client.calls == 1
    assert driver.likes == []
    assert store.decisions == []
    assert svc.disabled is False and svc.stop_requested is False
    assert svc.last_skip_reason and "could not be decoded" in svc.last_skip_reason
    assert driver.closed


def test_worker_auto_likes_without_comment_when_profile_content_is_safety_blocked():
    """The ranker already chose Like; an uncontrollable profile-content block removes only
    the optional comment and must not discard that decision or poison later profiles."""
    driver = FakeDriver(1)
    store = FakeStore()
    client = SafetyBlockedOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")

    Worker("hinge", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert client.calls == 1
    assert driver.likes == [None]
    assert driver.like_item_indexes == [None]
    assert driver.like_model_item_indexes == [None]
    assert store.decisions == [("hinge", "like", "auto")]
    assert store.openers == []
    assert svc.disabled is False and svc.stop_requested is False
    assert svc.last_skip_allows_commentless_like is True
    assert driver.closed


def test_training_stops_before_a_commentless_safety_block_like():
    """Training checkpoints always carry the actual complete opener text for Hub review."""
    from operation_love.training_actions import TrainingActionBridge

    class _TestingDriver(FakeDriver):
        accepts_opener = True
        supports_training_decision = True
        def set_training_decision(self, _approval):
            pass

    driver = _TestingDriver(1)
    store = FakeStore()
    service = OpenerService(SafetyBlockedOpenerClient(), CostTracker(PRICING, None), store, "s")
    stop = threading.Event()
    Worker("hinge", driver, FakeDecider("like"), service, store, "run1", _Pacing(), stop,
           mode="training", training_action_bridge=TrainingActionBridge()).run()

    assert stop.is_set()
    assert driver.likes == []
    assert store.decisions == []
    assert service.last_skip_allows_commentless_like is True
    assert driver.closed


def test_training_stale_hinge_calibration_stops_with_recalibration_action():
    """A runtime calibration mismatch is a targeting setup problem, not opener exhaustion."""
    from operation_love.status import RunStatus
    from operation_love.training_actions import TrainingActionBridge

    class _TestingDriver(FakeDriver):
        accepts_opener = True
        supports_training_decision = True

        def targeted_suggestion_blocker(self):
            return mismatch

        def next_profile(self):
            pytest.fail("a stale live calibration must stop before the first profile capture")

        def set_training_decision(self, _approval):
            pass

    class _UnreachableOpenerService:
        disabled = False
        stop_requested = False
        calls = 0

        def maybe_opener(self, *_args, **_kwargs):
            self.calls += 1
            pytest.fail("a stale targeting calibration must stop before opener generation")

    mismatch = ("apps.hinge.targeting_calibration is unavailable (the live app build/frame "
                "geometry does not exactly match schema-v3 calibration "
                "('10.1.0'/(1080, 2400) != '10.0.1'/(1080, 2400)))")
    driver = _TestingDriver(1)
    service, store, stop = _UnreachableOpenerService(), FakeStore(), threading.Event()
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="training")

    Worker("hinge", driver, FakeDecider("like"), service, store, "run1", _Pacing(), stop,
           mode="training", status=status,
           training_action_bridge=TrainingActionBridge()).run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "stopped"
    assert app["stop_kind"] == "targeting_calibration"
    assert "Capture and validate a fresh targeting calibration" in app["stop_reason"]
    assert mismatch in app["stop_reason"]
    assert service.calls == 0
    assert driver.i == 0
    assert driver.likes == []
    assert store.decisions == []
    assert stop.is_set() and driver.closed


def test_training_rechecks_calibration_before_requesting_an_opener():
    """A build change during capture must not spend a provider request on stale crops."""
    from operation_love.status import RunStatus
    from operation_love.training_actions import TrainingActionBridge

    mismatch = ("apps.hinge.targeting_calibration is unavailable (the live app build/frame "
                "geometry does not exactly match schema-v3 calibration "
                "('10.1.0'/(1080, 2400) != '10.0.1'/(1080, 2400)))")

    class _TestingDriver(FakeDriver):
        accepts_opener = True
        supports_training_decision = True

        def __init__(self):
            super().__init__(1)
            self.blocker = ""

        def targeted_suggestion_blocker(self):
            return self.blocker

        def next_profile(self):
            self.blocker = mismatch
            return super().next_profile()

        def set_training_decision(self, _approval):
            pass

    class _UnreachableOpenerService:
        disabled = False
        stop_requested = False

        def __init__(self):
            self.calls = 0

        def maybe_opener(self, *_args, **_kwargs):
            self.calls += 1
            pytest.fail("a calibration invalidated during capture must stop before opener generation")

    driver, service, store, stop = (
        _TestingDriver(), _UnreachableOpenerService(), FakeStore(), threading.Event())
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="training")

    Worker("hinge", driver, FakeDecider("like"), service, store, "run1", _Pacing(), stop,
           mode="training", status=status,
           training_action_bridge=TrainingActionBridge()).run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "stopped"
    assert app["stop_kind"] == "targeting_calibration"
    assert mismatch in app["stop_reason"]
    assert driver.i == 1, "the profile was captured before its runtime calibration changed"
    assert service.calls == 0
    assert driver.likes == []
    assert store.decisions == []
    assert stop.is_set() and driver.closed


@pytest.mark.parametrize("text", ["", " \n\t ", None, 17])
def test_training_stops_before_navigation_for_an_empty_opener_object(text):
    """A malformed opener object is not a reviewable typed draft."""
    from operation_love.training_actions import TrainingActionBridge

    class _TestingDriver(FakeDriver):
        accepts_opener = True
        supports_training_decision = True
        def set_training_decision(self, _approval):
            pass

    class _BlankPickService:
        disabled = False
        stop_requested = False
        last_skip_reason = None
        last_skip_allows_commentless_like = False

        def maybe_opener(self, *_args, **_kwargs):
            return SimpleNamespace(text=text)

    driver, store, stop = _TestingDriver(1), FakeStore(), threading.Event()
    Worker("hinge", driver, FakeDecider("like"), _BlankPickService(), store, "run1",
           _Pacing(), stop, mode="training",
           training_action_bridge=TrainingActionBridge()).run()

    assert stop.is_set()
    assert driver.likes == []
    assert store.decisions == []
    assert driver.closed


def test_training_refuses_a_noop_decision_setter_before_any_like():
    """A Hinge-named adapter cannot opt into testing by exposing a setter it never calls."""
    from operation_love.training_actions import TrainingActionBridge

    class _NoopApprovalDriver(FakeDriver):
        accepts_opener = True
        supports_training_decision = True
        def set_training_decision(self, _approval):
            pass

    class _Service:
        disabled = False
        stop_requested = False
        last_skip_reason = None
        last_skip_allows_commentless_like = False

        def maybe_opener(self, *_args, **_kwargs):
            return SimpleNamespace(
                text="A complete opener", index=1,
                index_space=INDEX_SPACE_PROFILE_PHOTOS, capture_order_index=0)

    driver, store, stop = _NoopApprovalDriver(1), FakeStore(), threading.Event()
    Worker("hinge", driver, FakeDecider("like"), _Service(), store, "run1", _Pacing(), stop,
           mode="training", training_action_bridge=TrainingActionBridge()).run()

    assert stop.is_set()
    assert driver.likes == []
    assert store.decisions == []
    assert driver.closed


def test_training_generation_context_clears_when_decision_hook_installation_fails():
    """A failed checkpoint callback installation cannot leak one draft into the next card."""
    from operation_love.training_actions import TrainingActionBridge

    class _FailingDecisionDriver(FakeDriver):
        accepts_opener = True
        supports_training_decision = True
        supports_capture_order_training_target = True

        def __init__(self):
            super().__init__(1)
            self.context_calls = []
            self.decision_calls = []

        def set_staged_opener_generation_context(self, context):
            self.context_calls.append(context)

        def set_training_decision(self, decision):
            self.decision_calls.append(decision)
            if decision is not None:
                raise RuntimeError("checkpoint callback installation failed")

    class _Service:
        disabled = False
        stop_requested = False
        last_skip_reason = None
        last_skip_allows_commentless_like = False

        def maybe_opener(self, *_args, **_kwargs):
            return SimpleNamespace(
                text="A complete opener", index=1,
                index_space=INDEX_SPACE_PROFILE_PHOTOS, capture_order_index=0,
                referenced="a visible detail", angle="a question",
                item_description="a photo",
                _staged_record=SimpleNamespace(model="gemini-test-model"))

    driver, store, stop = _FailingDecisionDriver(), FakeStore(), threading.Event()
    Worker("hinge", driver, FakeDecider("like"), _Service(), store, "run1", _Pacing(), stop,
           mode="training", training_action_bridge=TrainingActionBridge()).run()

    assert driver.context_calls == [{
        "model": "gemini-test-model", "index_space": INDEX_SPACE_PROFILE_PHOTOS,
        "referenced": "a visible detail", "angle": "a question",
        "item_description": "a photo",
    }, None]
    # Preserve the old decision-cleanup semantics: a setter that raises while installing is not
    # called again with None, because the driver's callback state is unknowable.
    assert len(driver.decision_calls) == 1 and driver.decision_calls[0] is not None
    assert driver.likes == [] and stop.is_set() and driver.closed


def test_worker_stops_before_bare_like_on_single_bad_request():
    driver = FakeDriver(3)
    store = FakeStore()
    client = BadRequestOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert client.calls == 1                       # below _BAD_REQUEST_LATCH_THRESHOLD -- didn't latch
    assert driver.likes == []
    assert store.decisions == []
    assert svc.disabled is False and svc.stop_requested is False
    assert svc.last_skip_reason and "HTTP 400" in svc.last_skip_reason
    assert driver.closed


def test_worker_stops_before_bare_like_on_single_transient_failure():
    driver = FakeDriver(3)
    store = FakeStore()
    client = TransientOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert client.calls == 1                       # below _TRANSIENT_LATCH_THRESHOLD -- didn't latch
    assert driver.likes == []
    assert store.decisions == []
    assert svc.disabled is False and svc.stop_requested is False
    assert svc.last_skip_reason and "connection reset" in svc.last_skip_reason
    assert driver.closed


def test_auto_mode_stop_reason_names_the_cause_when_every_retry_attempt_fails():
    """Companion to test_auto_mode_stop_reason_is_visible_in_status_when_opener_budget_exhausts
    above: when OpenerService's retry loop exhausts (see
    test_worker_halts_before_bare_like_when_the_opener_ultimately_fails_every_retry), the hub
    must show the actual cause -- the last attempt's real failure message -- not a generic
    'stopped'."""
    from operation_love.status import RunStatus

    driver = FakeDriver(3)
    store = FakeStore()
    client = ParseErrorOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "stopped"
    assert "bad JSON in response body" in app["stop_reason"]
    # See the matching stop_kind comment on
    # test_auto_mode_stop_reason_is_visible_in_status_when_opener_budget_exhausts above --
    # this is the OTHER opener-triggered stop path (retry exhaustion, not budget), and it
    # must publish the same "opener" stop_kind, not the new "deck_blocked".
    assert app["stop_kind"] == "opener"


def test_worker_skips_openers_when_driver_declines_even_through_opener_failures():
    """A non-opener-capable app (Bumble-style) must be COMPLETELY unaffected by the new
    per-call guard -- it never calls the opener service at all in auto mode, so a failing
    opener client (that would halt an opener-capable app) must not even be reached."""
    driver = FakeDriver(3)
    driver.accepts_opener = False
    store = FakeStore()
    client = ParseErrorOpenerClient()          # would halt an opener-capable app every time
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("like"), svc, store).run()

    assert driver.likes == [None, None, None]  # liked all 3, normally, with no opener
    assert client.calls == 0                   # opener service never even called
    assert len(store.decisions) == 3
    assert svc.last_skip_reason is None         # never touched


def test_worker_with_opener_disabled_by_config_still_likes_normally_in_auto_mode():
    """opener.enabled=false in config produces an OpenerService(client=None, ...), which is
    `disabled` from construction (see OpenerService.__init__). That is a DELIBERATE
    "openers don't exist this run" choice, not a per-call failure of an otherwise-live
    service -- the no-bare-like guard must not mistake it for one and halt the very first
    like of every run. `disabled` is exactly what tells the two apart (see the guard's
    comment in worker.py's _auto_loop)."""
    driver = FakeDriver(3)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    assert svc.disabled is True                # opener.enabled=false -- no client configured

    _worker(driver, FakeDecider("like"), svc, store).run()

    assert driver.likes == [None, None, None]   # liked all 3, normally, with no opener
    assert len(store.decisions) == 3
    assert svc.stop_requested is False           # never asked anyone to stop


def test_auto_mode_stops_as_blocked_and_never_acts_when_driver_reports_a_blocked_deck():
    """Same incident, AUTO mode: a driver reporting a blocked deck must stop the same way
    (state="blocked", stop_kind="deck_blocked") AND -- the extra AUTO-specific requirement
    -- must never issue a single like/dislike. An autonomous loop that kept swiping past an
    unrecognized paywall would be the worse failure mode; this pins that it doesn't."""
    from operation_love.status import RunStatus

    driver = _BlockedDriver(3)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "blocked"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert app["stop_kind"] == "deck_blocked"
    assert driver.likes == [] and driver.dislikes == 0    # no autonomous action into a paywall
    assert store.decisions == []
    assert driver.closed


def test_auto_none_capture_publishes_the_driver_latched_entry_refusal_as_blocked():
    """The post-capture blocked_reason probe must explain a conservative entry refusal."""
    from operation_love.status import RunStatus

    reason = ("the capture entry could not be proven at scroll top after a rewind: "
              "confirmed_not_top")

    class EntryRefusalDriver(FakeDriver):
        def __init__(self):
            super().__init__(1)
            self.refused = False

        def blocked_reason(self):
            return reason if self.refused else None

        def next_profile(self):
            self.refused = True
            return None

    driver = EntryRefusalDriver()
    store = FakeStore()
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="auto")
    Worker("hinge", driver, FakeDecider("like"), None, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "blocked"
    assert app["stop_reason"] == reason and app["stop_kind"] == "deck_blocked"
    assert driver.likes == [] and driver.dislikes == 0
    assert store.decisions == [] and driver.closed


def test_auto_mode_post_send_blocked_error_records_no_rejected_like():
    """The worker's between-profile probe cannot see a screen that only appears *after*
    Send Like.  A driver therefore raises DeckBlockedError at that exact point; it must produce
    the same graceful status as an already-blocked deck, without increasing counters or saving a
    phantom decision."""
    from operation_love.status import RunStatus

    driver = _BlockedAfterLikeDriver()
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert driver.like_calls == 1
    assert driver.likes == []
    assert store.decisions == [] and store.labels == []
    assert app["state"] == "blocked"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert app["stop_kind"] == "deck_blocked"
    assert driver.closed


def test_worker_handles_concrete_hinge_blocked_error_without_count_store_or_failure_snapshot():
    """The concrete Hinge error must take Worker's graceful DeckBlockedError branch.

    Its HingeActionError base is intentionally retained for direct callers, so this protects the
    multiple-inheritance order from quietly turning a known paywall into the generic error path.
    """
    from operation_love.drivers.hinge import HingeActionError, HingeDeckBlockedError
    from operation_love.status import RunStatus

    class ConcreteHingePaywallDriver(FakeDriver):
        accepts_opener = False

        def __init__(self):
            super().__init__(1)
            self.failure_snapshots = []

        def blocked_reason(self):
            return None

        def like(self, opener=None, item_index=None, *, model_item_index=None):
            raise HingeDeckBlockedError(_PAYWALL_REASON)

        def snapshot_failure(self, exc):
            self.failure_snapshots.append(exc)

    assert issubclass(HingeDeckBlockedError, DeckBlockedError)
    assert issubclass(HingeDeckBlockedError, HingeActionError)
    assert HingeDeckBlockedError.__mro__[:3] == (
        HingeDeckBlockedError, HingeActionError, DeckBlockedError)

    driver = ConcreteHingePaywallDriver()
    store = FakeStore()
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="auto")
    Worker("hinge", driver, FakeDecider("like"), None, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "blocked" and app["stop_kind"] == "deck_blocked"
    assert app["swipes_run"] == 0
    assert store.decisions == [] and driver.failure_snapshots == []
    assert driver.likes == [] and driver.closed


def test_auto_mode_handles_concrete_hinge_blocked_error_from_dislike_boundary():
    """A foreground race on Pass is the same blocked state as one on Send Like.

    The dislike call historically sat outside the Like branch's DeckBlockedError handler, so
    the identical Hinge error became a red unexpected failure depending only on the decision.
    """
    from operation_love.drivers.hinge import HingeDeckBlockedError
    from operation_love.status import RunStatus

    class ForegroundLostOnDislike(FakeDriver):
        def __init__(self):
            super().__init__(1)
            self.failure_snapshots = []

        def blocked_reason(self):
            return None

        def dislike(self, *, should_stop=None):
            raise HingeDeckBlockedError(_PAYWALL_REASON)

        def snapshot_failure(self, exc):
            self.failure_snapshots.append(exc)

    driver = ForegroundLostOnDislike()
    store = FakeStore()
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="auto")
    Worker("hinge", driver, FakeDecider("dislike"), None, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "blocked" and app["stop_kind"] == "deck_blocked"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert app["swipes_run"] == 0
    assert store.decisions == [] and driver.failure_snapshots == []
    assert driver.dislikes == 0 and driver.closed


def test_auto_mode_surfaces_block_latched_during_capture_that_returns_no_profile():
    """A package switch between the loop probe and capture must not look like a clean stop."""
    from operation_love.status import RunStatus

    class ForegroundLostDuringCapture(FakeDriver):
        def __init__(self):
            super().__init__(1)
            self.blocked_calls = 0

        def blocked_reason(self):
            self.blocked_calls += 1
            return None if self.blocked_calls == 1 else _PAYWALL_REASON

        def out_of_profiles(self):
            return False

        def next_profile(self):
            return None

    driver = ForegroundLostDuringCapture()
    store = FakeStore()
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="auto")
    Worker("hinge", driver, FakeDecider("dislike"), None, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("hinge")["app"]
    assert driver.blocked_calls == 2
    assert app["state"] == "blocked" and app["stop_kind"] == "deck_blocked"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert store.decisions == [] and driver.likes == [] and driver.dislikes == 0
    assert driver.closed


def test_auto_mode_prefers_blocked_over_out_of_profiles_when_driver_reports_both():
    from operation_love.status import RunStatus

    driver = _BlockedAndEmptyDriver()
    assert driver.out_of_profiles() is True          # sanity: the empty-deck condition really holds
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "blocked"                  # NOT "out_of_profiles"
    assert app["stop_reason"] == _PAYWALL_REASON
    assert app["stop_kind"] == "deck_blocked"
    assert driver.likes == [] and driver.dislikes == 0
    assert driver.closed


def test_auto_mode_unaffected_when_driver_explicitly_reports_no_blocked_deck():
    """Guards against the new check accidentally halting a normal run: a driver whose
    blocked_reason() is explicitly present (not merely inherited-and-never-called) but
    always answers None -- "nothing standing between us and the deck" -- must swipe the
    whole deck exactly as before, with blocked_reason() polled once per loop iteration
    (proving the check genuinely runs every time) but never once tripping it."""
    class ExplicitlyUnblockedDriver(FakeDriver):
        def __init__(self, n):
            super().__init__(n)
            self.blocked_calls = 0

        def blocked_reason(self):
            self.blocked_calls += 1
            return None

    driver = ExplicitlyUnblockedDriver(3)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    _worker(driver, FakeDecider("dislike"), svc, store).run()

    assert driver.dislikes == 3 and driver.likes == []
    # Polled every iteration: 3 cards processed + 1 final iteration where out_of_profiles()
    # finally reads True -- 4 calls, none of which ever returned anything but None.
    assert driver.blocked_calls == 4
    assert driver.closed


def test_worker_stops_at_per_run_like_budget():
    driver = FakeDriver(5)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    limiter = RateLimiter(max_likes_per_run=2)
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=limiter).run()
    # Likes up to the budget, then stops the run before the over-budget like.
    assert driver.likes == ["hi 1", "hi 2"]
    # AUTO is inference-only: 2 decisions logged, no training labels saved.
    assert store.labels == [] and len(store.decisions) == 2
    assert driver.closed


def test_hinge_normal_auto_uses_ranker_and_continues_past_the_first_card(monkeypatch):
    from operation_love import worker as worker_mod

    class IdentityPolicy:
        def __init__(self, **_):
            self.landed = []

        def apply_decision(self, decision, _profile):
            return SimpleNamespace(decision=decision)

        def record_landed_action(self, decision):
            self.landed.append(decision)

        def post_action_delay_s(self, *_args, **_kwargs):
            return 0.0

    class SequencedDecider:
        def __init__(self):
            self.decisions = iter(("like", "dislike", "like"))
            self.calls = 0

        def decide(self, _profile):
            self.calls += 1
            decision = next(self.decisions)
            return Decision(decision, 0.9 if decision == "like" else 0.1,
                            [0.1, 0.2], "ranker")

    monkeypatch.setattr(worker_mod, "AutoSessionPolicy", IdentityPolicy)
    driver = FakeDriver(3)
    decider = SequencedDecider()
    store = FakeStore()
    service = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")

    Worker("hinge", driver, decider, service, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=RateLimiter()).run()

    assert decider.calls == 3
    assert driver.likes == ["hi 1", "hi 2"] and driver.dislikes == 1
    assert [decision for _app, decision, _source in store.decisions] == [
        "like", "dislike", "like"]


class ScriptedScoreDecider:
    """Always returns 'like', with a scripted per-call score in order -- lets a test
    drive the ratio-ceiling shaper (worker.py's _auto_loop, via RateLimiter.
    allow_like_ratio in limits.py) through a specific marginal/strong score sequence,
    which the old constant-score FakeDecider can't do."""
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def decide(self, profile):
        score = self.scores[self.calls]
        self.calls += 1
        return Decision(decision="like", score=score, embedding=[0.1, 0.2], source="ranker")


def test_like_ratio_ceiling_demotes_marginal_likes_but_lets_strong_ones_through():
    """The ratio ceiling is now SCORE-AWARE (see limits.py's allow_like_ratio
    docstring): once the running rate is at/above target_like_ratio, only a MARGINAL
    like (within 0.15 of the ranker's own threshold) is demoted to a pass -- a
    clearly strong like (>= threshold + 0.15) is let through even though that briefly
    overshoots the ratio. This replaces the earlier score-blind version: an audit ran
    the OLD shaper against real scores [0.80, 0.99, 0.81, 0.82] at
    target_like_ratio=0.5 and found it demoted the model's SINGLE MOST CONFIDENT call
    (0.99) purely because of arrival order, while keeping a weaker 0.80 liked -- the
    most anti-"purely model-driven" outcome the shaper could produce. It still
    demotes rather than halts (a soft shaper, not a hard cap -- max_likes_per_run is
    the separate hard cap, see test_worker_stops_at_per_run_like_budget above, which
    this change does not touch).

    With target_like_ratio=0.5 and threshold=0.5 (FakeDecider/ScriptedScoreDecider's
    implicit default, since neither exposes .model.threshold):
    card 1: score=0.55 (marginal), acted=0 -> no history yet    -> like.  liked=1 acted=1
    card 2: score=0.55 (marginal), 1/1=100% >= 50% -> MARGINAL  -> demoted to pass.
    card 3: score=0.95 (strong),   1/2=50%  >= 50% -> STRONG    -> survives, stays like.
    card 4: score=0.55 (marginal), 2/3=67%  >= 50% -> MARGINAL  -> demoted to pass.

    0.55 (not e.g. 0.51) is deliberately chosen so this test exercises ONLY the
    ratio-ceiling logic under test, not AutoSessionPolicy's own separate, independent
    contextual-threshold demotion that runs earlier in _auto_loop (d =
    self._auto_policy.apply_decision(d, profile).decision, BEFORE the ratio check):
    that policy can itself lift the effective bar up to base_threshold +
    max_threshold_lift = 0.5 + 0.045 = 0.545 (interaction.py's _THRESHOLD_MAX_LIFT),
    so a score of 0.51 could occasionally get demoted right there by pure chance
    (Worker doesn't inject a seeded rng into AutoSessionPolicy), producing a flaky
    test that has nothing to do with the ratio ceiling. 0.55 is always >= 0.545, so
    it always survives AutoSessionPolicy unconditionally, while still being < 0.65
    (threshold + _STRONG_LIKE_MARGIN) so the ratio ceiling itself still calls it
    marginal.
    """
    driver = FakeDriver(4)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    limiter = RateLimiter(target_like_ratio=0.5)
    decider = ScriptedScoreDecider([0.55, 0.55, 0.95, 0.55])
    Worker("bumble", driver, decider, svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=limiter).run()

    # All 4 cards processed — NOT halted.
    assert len(store.decisions) == 4
    assert driver.closed
    assert [d for _, d, _ in store.decisions] == ["like", "dislike", "like", "dislike"]
    # The strong 0.95 actually landed as a driver.like() call (with its opener), not
    # silently swallowed by the ceiling the way a marginal one is.
    assert driver.likes == ["hi 1", "hi 2"]


def test_ratio_ceiling_demotion_tags_a_distinguishable_source(monkeypatch):
    """Consistency with interaction.py's own contextual demotion (which tags
    f"{source}_contextual", see AutoSessionPolicy.apply_decision): the ratio-ceiling
    demotion must also retag .source instead of leaving it as the raw ranker source,
    so a future consumer of Decision.source can tell a shaped decision apart from a
    raw model one. No functional effect today -- Decision.source isn't read anywhere
    downstream and the stored row hardcodes source="auto" (see FakeStore.
    record_decision / the real store.record_decision call in worker.py) -- so this
    spies on the Decision objects the loop actually constructs rather than on the
    store, which cannot see .source at all."""
    from operation_love import worker as worker_mod
    from operation_love.ranker.decider import Decision as RealDecision

    created = []

    def spying_decision(decision, score, embedding, source):
        d = RealDecision(decision, score, embedding, source)
        created.append(d)
        return d

    monkeypatch.setattr(worker_mod, "Decision", spying_decision)

    driver = FakeDriver(2)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    limiter = RateLimiter(target_like_ratio=0.5)
    # 0.55, not 0.51: stays clear of AutoSessionPolicy's own independent demotion --
    # see the worked-numbers comment in
    # test_like_ratio_ceiling_demotes_marginal_likes_but_lets_strong_ones_through above.
    decider = ScriptedScoreDecider([0.55, 0.55])   # card 2: marginal, ratio at ceiling -> demoted
    Worker("bumble", driver, decider, svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=limiter).run()

    demoted = [d for d in created if d.decision == "dislike"]
    assert len(demoted) == 1
    assert demoted[0].source == "ranker_ratio_ceiling"


class _RatioNoZeroGuardLimiter:
    """HOLE 3 (mutation audit): a DELIBERATELY minimal RateLimiter double. The real
    RateLimiter.allow_like_ratio (limits.py) already returns True whenever acted == 0, so
    _auto_loop's OWN `acted > 0` clause (right before the ratio check) is redundant against
    today's real limiter -- nothing breaks if that clause is deleted, and no test built
    against the real RateLimiter would notice. The clause is deliberate defense in depth
    (see _auto_loop's comment above it): it only matters if RateLimiter's internal guard is
    ever removed, or -- exactly what this double is for -- a limiter implementation is used
    that never had the guard in the first place. This double's allow_like_ratio divides
    liked/acted with NO acted == 0 check of its own, so calling it with acted == 0 raises
    ZeroDivisionError; only the worker's own `acted > 0` short-circuit can prevent that
    call from ever happening on the very first 'like' decision of a run (acted is still 0
    at the moment of that check -- see _auto_loop, `acted` only increments AFTER it)."""
    max_per_day = None
    target_like_ratio = 0.5

    def allow(self, acted_this_run, acted_today):
        return True

    def allow_like(self, liked_this_run):
        return True

    def allow_like_ratio(self, liked, acted):
        return liked / acted < self.target_like_ratio   # no acted == 0 guard, unlike limits.py

    def describe(self):
        return "test double (no internal zero guard)"


def test_first_like_survives_a_limiter_double_with_no_internal_zero_guard():
    """Confirms _auto_loop's `acted > 0` clause is load-bearing against a limiter double
    that (unlike the real RateLimiter) does not itself guard acted == 0: the very first
    'like' decision of the run must land normally, with no ZeroDivisionError, because the
    worker's own guard keeps allow_like_ratio(liked=0, acted=0) from ever being called."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = OpenerService(FakeOpenerClient(), CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=_RatioNoZeroGuardLimiter()).run()

    assert driver.likes == ["hi 1"]                         # the like landed, not swallowed
    assert store.decisions == [("bumble", "like", "auto")]


def test_worker_daily_limit_ignores_manual_decisions():
    driver = FakeDriver(2)
    store = FakeStore()
    store.decisions = [("bumble", "like", "manual")] * 83
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider("dislike"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", limiter=RateLimiter(max_per_day=1)).run()

    assert driver.dislikes == 1
    assert store.count_today("bumble") == 1
    assert store.decisions[-1] == ("bumble", "dislike", "auto")
    assert all(source == "manual" for _, _, source in store.decisions[:-1])


def test_no_auto_decision_recorded_when_like_raises_and_snapshot_runs_live():
    driver = RaisingLikeDriver(2)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")   # openers disabled -> pick None
    _worker(driver, FakeDecider("like"), svc, store).run()
    # like() raised -> halt_on_error -> the phantom 'like' must NOT be recorded (record is post-action)
    assert store.decisions == []
    assert driver.closed
    # the failure snapshot must have run while the transport was still live (before close())
    assert driver.snapshotted and driver.snapshotted[0][0] is True


def test_worker_treats_browser_close_as_graceful_stop():
    driver = ClosedDriver()
    store = FakeStore()
    stop_event = threading.Event()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           stop_event, mode="auto").run()

    assert stop_event.is_set()
    assert driver.closed
    assert store.labels == []
    assert store.profiles == []


def test_worker_stops_before_writing_auto_decision_after_stop():
    driver = FakeDriver(1)
    store = FakeStore()
    stop_event = threading.Event()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    Worker("bumble", driver, StopAfterDecide(stop_event), svc, store, "run1", _Pacing(),
           stop_event, mode="auto").run()

    assert store.decisions == []
    assert store.labels == [] and store.profiles == []
    assert driver.dislikes == 0 and driver.likes == []
    assert driver.closed


def test_auto_pass_stop_at_driver_input_boundary_is_not_landed_or_recorded():
    """A Stop after Worker has chosen Pass still wins until the driver issues input.

    This is the precise scheduling gap a worker-side `is_set()` check cannot close:
    the driver flips Stop only after its caller entered dislike(), then observes the
    callback before it increments the fake physical-action counter.
    """
    from operation_love.status import RunStatus

    stop_event = threading.Event()

    class _StopAtPassBoundary(FakeDriver):
        def dislike(self, *, should_stop=None):
            stop_event.set()
            return super().dislike(should_stop=should_stop)

    driver = _StopAtPassBoundary(1)
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    service = OpenerService(None, CostTracker(PRICING, None), store, "style")
    Worker("bumble", driver, FakeDecider("dislike"), service, store, "run1", _Pacing(),
           stop_event, mode="auto", status=status).run()

    assert driver.dislikes == 0
    assert store.decisions == []
    assert status.snapshot()["apps"]["bumble"]["swipes_run"] == 0
    assert driver.closed


def test_auto_refuses_to_pass_with_a_driver_that_cannot_honour_stop():
    """Legacy no-argument dislike() implementations cannot silently reopen the race."""
    from operation_love.status import RunStatus

    class _LegacyPassDriver(FakeDriver):
        supports_interruptible_dislike = False

        def dislike(self):
            self.dislikes += 1

    driver = _LegacyPassDriver(1)
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    service = OpenerService(None, CostTracker(PRICING, None), store, "style")
    Worker("bumble", driver, FakeDecider("dislike"), service, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    assert driver.dislikes == 0
    assert store.decisions == []
    assert "Stop-safe AUTO pass" in str(status.snapshot()["apps"]["bumble"]["error"])


def test_worker_closes_open_session_when_post_open_setup_raises():
    """The outer worker finalizer owns a driver after open_session() returns."""
    class _PostOpenFailure(FakeDriver):
        accepts_opener = True

        def __init__(self):
            super().__init__(1)
            self.close_calls = 0

        def targeted_suggestion_blocker(self):
            raise RuntimeError("live calibration probe failed")

        def close(self):
            self.close_calls += 1
            super().close()

    driver = _PostOpenFailure()
    store = FakeStore()
    from operation_love.status import RunStatus
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    service = OpenerService(None, CostTracker(PRICING, None), store, "style")
    Worker("bumble", driver, FakeDecider("dislike"), service, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    assert driver.opened and driver.closed
    assert driver.close_calls == 1
    assert store.decisions == []
    assert status.snapshot()["apps"]["bumble"]["state"] == "error"


def test_auto_mode_halts_on_unexpected_even_when_a_driver_opts_out_of_halting():
    """AUTO mode must STOP on any unexpected error for EVERY app, EVEN one that explicitly sets
    halt_on_error=False, so an autonomous run never keeps swiping blindly. It must halt on the
    FIRST error (no restart retries) and set the stop so buffered data is saved.

    The opt-out is set explicitly here rather than relied on as a default: halt_on_error now
    defaults to True on DatingAppDriver, so "a driver that never mentions it" no longer means
    "a driver that restarts". Setting it False makes this the strongest version of the claim —
    auto mode overrides the opt-out."""
    class BoomDriver(FakeDriver):
        halt_on_error = False           # explicitly opted out — auto must override it anyway
        def __init__(self, n):
            super().__init__(n)
            self.calls = 0
        def next_profile(self):
            self.calls += 1
            raise RuntimeError("unexpected boom")

    driver = BoomDriver(3)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    stop = threading.Event()
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           stop, mode="auto", max_restarts=5).run()

    assert driver.calls == 1        # halted on the first error — did NOT restart-and-retry
    assert stop.is_set()            # stop set (halt path), so the supervisor saves buffered data
    assert driver.closed


def test_pace_scales_wait_by_configured_swipe_delay(monkeypatch):
    """Every existing pacing test uses _Pacing.swipe_delay_s == 0.0 (the early-return path)
    -- the scaling branch itself (`scale = swipe_delay_s / _THINK_TIME_BASELINE_S`) was
    entirely unexercised. Pin the formula deterministically: stub think_time_s() to a fixed
    base and assert stop_event.wait() is called with base * scale, not the raw base.

    Needs a driver declaring think_time_calibrated: _pace() only uses the measured
    like/pass dwell shape for apps it was actually calibrated on (Hinge), and falls back
    to a flat anchor otherwise -- so without one this exercises the wrong branch."""
    import operation_love.worker as wmod
    monkeypatch.setattr(wmod, "think_time_s", lambda key: 2.0)   # fixed, deterministic base

    class _Pacing2:
        swipe_delay_s = 7.0   # 2x the 3.5s default anchor -> scale == 2.0

    class _RecordingStopEvent:
        def __init__(self):
            self.waits = []
        def wait(self, t):
            self.waits.append(t)
        def is_set(self):
            return False

    class _CalibratedDriver:
        think_time_calibrated = True

    w = Worker.__new__(Worker)
    w.driver = _CalibratedDriver()
    w.pacing = _Pacing2()
    w.stop_event = _RecordingStopEvent()
    w._pace("like")
    assert w.stop_event.waits == [4.0]   # 2.0 * (7.0 / 3.5)


def test_auto_defer_cold_start_stops_without_action_or_record():
    """AUTO cold-start: a not-ready ranker returns 'defer' -> the loop STOPS after pulling the
    first card, takes no swipe, records no decision/label, and still closes the driver."""
    class DeferDecider(FakeDecider):
        def decide(self, profile):
            return Decision(decision="defer", score=0.0, embedding=[0.1], source="cold_start")

    driver = FakeDriver(3)
    store = FakeStore()
    svc = OpenerService(None, CostTracker(PRICING, None), store, "s")
    _worker(driver, DeferDecider(), svc, store).run()

    # (a) no action: nothing liked, nothing disliked
    assert driver.likes == [] and driver.dislikes == 0
    # (b) auto 'defer' records neither a decision nor a label/profile
    assert store.decisions == [] and store.labels == [] and store.profiles == []
    # (c) the finally ran -> driver closed
    assert driver.closed
    # (d) exactly one profile pulled, then the loop broke immediately
    assert driver.i == 1


def test_auto_loop_passes_stop_event_is_set_as_should_stop_to_maybe_opener():
    """Pins the AUTO-loop call site: worker.py must forward self.stop_event.is_set (not some
    other callable, and not omit it) so a Stop click can abort an in-flight opener call."""
    driver = FakeDriver(1)
    store = FakeStore()
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), store, "s")
    w = _worker(driver, FakeDecider("like"), svc, store)
    w.run()

    assert client.calls == 1
    assert client.should_stops == [w.stop_event.is_set]


def test_auto_like_with_opener_service_none_does_not_crash():
    from operation_love.status import RunStatus

    driver = FakeDriver(1)
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), None, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] != "error"          # not the HALT-on-unexpected path (an AttributeError)
    assert app.get("error") is None
    assert driver.likes == [None]           # no opener service -> liked, but with no opener
    assert driver.closed


def test_auto_dislike_with_opener_service_none_does_not_crash():
    """The post-action stop_requested check runs after EVERY action -- a dislike-only run
    (which never calls maybe_opener() at all) must not crash on it either."""
    from operation_love.status import RunStatus

    driver = FakeDriver(2)
    store = FakeStore()
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("dislike"), None, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] != "error"
    assert app.get("error") is None
    assert driver.dislikes == 2
    assert driver.closed


# ---------------------------------------------------------------------------------------
# WHAT THE AUTO LOOP HANDS THE DRIVER, AND WHAT IT DOES WHEN THE DRIVER CANNOT HONOUR IT.
# This section used to be headed ANCHORED_OPENER and was about the driver's repair hatch: a
# callback the worker handed over so the driver could re-ask for text about whatever item its
# tap actually landed on. That hatch was removed on 2026-08-12 -- it repaired the TEXT while
# the LIKE still went to an item the model never chose (ops/OPENER-REDESIGN.md 5.6). What is
# left is the index the worker converts and passes, the stops it makes before calling the
# driver at all, and doc 5.6's stop for when the driver reports the miss itself (bottom of
# this file).
# ---------------------------------------------------------------------------------------

class _SequencedOpenerService:
    """Records every maybe_opener() call's kwargs and returns a SCRIPTED pick per call, in
    call order -- so a test can drive several opener calls through DIFFERENT results and tell
    which call produced which text. A scripted entry of None models a call that produced no
    opener at all."""
    stop_requested = False
    disabled = False

    def __init__(self, picks):
        self.picks = list(picks)
        self.calls = []

    def maybe_opener(self, run_id, app, profile, *, anchor=None, items=None,
                     should_stop=None):
        self.calls.append({"run_id": run_id, "app": app, "anchor": anchor, "items": items,
                           "should_stop": should_stop})
        return self.picks[len(self.calls) - 1]


def test_auto_like_passes_the_translated_item_index_and_no_repair_callback():
    """Pins item_index: it must be whatever OpenerPick.capture_order_index says, never the raw
    pick.index and never a hardcoded 0 -- this pick names model item 3 in the (translatable)
    profile-photos space, so the driver must receive 2, not 3 and not 0.

    And pins what is NO LONGER passed. This test used to assert that the worker also handed over
    a callable `anchored_opener`, the driver's repair hatch for a targeting miss discovered at
    swipe time. That hatch is removed (ops/OPENER-REDESIGN.md 5.6): it repaired the TEXT while the
    LIKE still landed on an item the model never chose. FakeDriver.like's signature no longer
    accepts the keyword, so a worker that started passing one again would raise a TypeError here
    rather than silently reintroducing the substitution path."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = _SequencedOpenerService([
        OpenerPick("hi 1", index=3, referenced="r", index_space=INDEX_SPACE_PROFILE_PHOTOS),
    ])
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.likes == ["hi 1"]
    assert driver.like_item_indexes == [2]


# --- INDEX SPACES AT THE WORKER/DRIVER SEAM (ops/OPENER-REDESIGN.md 5.1/5.3/5.7) ------------
# `pick.index` counts the numbered images the MODEL was sent; driver.like(item_index=...)
# counts the frames the DRIVER captured. Between 2026-08-12 and this regression the worker
# handed one straight to the other, and because the driver's only bounds check is against its
# own frame count, an in-range 1-based item number resolved to a real, adjacent, WRONG frame
# and was reported on-target -- the comment landed one card down with full confidence.
#
# A LATER regression narrowed but did not close the gap: the worker started converting via
# `capture_order_index`, but when that conversion came back None (no sound conversion exists)
# it still called driver.like(item_index=None) and let the driver tap the first item and
# repair the message against whatever it landed on -- which still spent a like on an item the
# model never chose, just with text that matched it. The worker now refuses to call the driver
# AT ALL in that case (ops/OPENER-REDESIGN.md 5.3: "treat a missing table as a hard stop, never
# as a reason to fall back to a fixed coordinate") and stops the run instead. These tests pin
# the crossing itself, at the seam, in the worker, where both bugs lived.

def test_auto_like_translates_a_profile_photo_index_into_the_drivers_capture_order():
    """The legacy frame shape numbers profile.photos 1..N, and the driver's capture order is
    that same list 0-based, so model item 3 is capture-order index 2. The worker must send the
    CONVERTED value -- sending 3 would target the next frame down."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = _SequencedOpenerService([
        OpenerPick("hi 1", index=3, referenced="r", index_space=INDEX_SPACE_PROFILE_PHOTOS),
    ])
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.like_item_indexes == [2]


def test_auto_like_hands_a_model_item_number_to_the_driver_rather_than_converting_it():
    """REWRITTEN IN PLACE, 2026-08-12, when doc 5.5's counting navigation was wired. This test
    used to pin the opposite behaviour -- that a model-item pick was UNTRANSLATABLE and stopped
    the run -- and the reason it stopped was true at the time: nothing could turn that number
    into a tapped heart. `item_nav.navigate_to_item` now can, by walking UP from where the
    profile read left the card, so the number goes to the driver AS a model item number.

    What must NOT happen is a conversion. `capture_order_index` still returns None for this
    space, correctly: the model's number resolves to a HEART ORDINAL, not to a captured frame,
    and inventing a frame index for it is the exact bug the two index spaces exist to prevent.
    So the worker passes `model_item_index=3` and leaves `item_index` at None -- which is the one
    combination `hinge._like_comment_sheet` routes to counting navigation."""
    driver = FakeDriver(1)
    store = FakeStore()
    svc = _SequencedOpenerService([
        OpenerPick("hi 1", index=3, referenced="r", index_space=INDEX_SPACE_MODEL_ITEMS),
    ])
    Worker("bumble", driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.likes == ["hi 1"]
    assert driver.like_model_item_indexes == [3]      # the model's own number, unconverted
    assert driver.like_item_indexes == [None]         # ...and NO capture-order index beside it
    assert len(store.decisions) == 1


def test_auto_like_never_turns_an_absent_item_index_into_the_first_item():
    """ITEM_INDEX_ABSENT means the model could not pick an item. It is 0, and 0 is a perfectly
    legal FIRST FRAME in the driver's space, so passing it through would turn "no choice" into
    a confident like of item 1 -- which is exactly what happened before this regression. It
    must never even reach the driver -- let alone `_locate_target_heart` -- in any index
    space: the worker refuses to call driver.like() and stops the run instead."""
    for space in (INDEX_SPACE_PROFILE_PHOTOS, INDEX_SPACE_MODEL_ITEMS):
        driver = FakeDriver(1)
        svc = _SequencedOpenerService([
            OpenerPick("hi 1", index=ITEM_INDEX_ABSENT, referenced="r", index_space=space),
        ])
        Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
               threading.Event(), mode="auto").run()

        assert driver.likes == [], space
        assert driver.like_item_indexes == [], space


def test_auto_mode_stop_reason_names_absent_item_index_distinctly_from_an_unknown_space():
    """The operator's next move differs between the two refusal causes above: "the model never
    named an item" is an opener-layer problem (a prompt/model issue -- retrying may help),
    while "the model named one in a space this worker does not act on" is a wiring problem. A
    single stop_reason reading "item None" for both would hide that distinction from the
    operator, so the two must produce recognizably different text -- and both must publish
    stop_kind="opener", the same channel every other opener-triggered stop uses.

    REWRITTEN IN PLACE, 2026-08-12: the second half used to use INDEX_SPACE_MODEL_ITEMS, which
    was untranslatable before doc 5.5's counting navigation was wired and is now the space the
    worker acts on directly. What is left in the "cannot act on it" class is a space this build
    has never heard of, which is what a stale or future producer would emit -- and which must
    still degrade to a stop rather than to a guess."""
    from operation_love.status import RunStatus

    driver = FakeDriver(1)
    svc = _SequencedOpenerService([OpenerPick("hi 1", index=ITEM_INDEX_ABSENT, referenced="r")])
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "stopped"
    assert app["stop_kind"] == "opener"
    assert "no item number" in app["stop_reason"]
    assert driver.likes == []

    driver2 = FakeDriver(1)
    svc2 = _SequencedOpenerService([
        OpenerPick("hi 1", index=3, referenced="r", index_space="some_future_space"),
    ])
    status2 = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    Worker("bumble", driver2, FakeDecider("like"), svc2, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status2).run()

    app2 = status2.app_view("bumble")["app"]
    assert app2["state"] == "stopped"
    assert app2["stop_kind"] == "opener"
    assert "some_future_space" in app2["stop_reason"]
    assert "neither" in app2["stop_reason"]
    assert driver2.likes == []


# =====================================================================================
# THE REQUEST SHAPE: numbered crops, or a stop -- never the raw frames
# (ops/OPENER-REDESIGN.md 5.2/5.7)
# =====================================================================================

def test_auto_sends_the_enumerated_item_crops_as_the_request_not_the_scroll_frames():
    """Doc 5.2's whole point: with crops, image k IS item k, so the number the model answers
    with means something. When the capture enumerated the profile, the AUTO loop must build an
    ItemRequest from it and hand it to the opener call -- numbered items in model order, the
    unnumbered context tier after them, her name as text, the truncation flag -- and the raw
    scroll frames must not be what the model is looking at."""
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), FakeStore(), "s")
    driver = FakeDriver(1)
    driver.cards[0] = Profile(
        photos=[b"frame-0", b"frame-1", b"frame-2"],
        name="Ada", items=(b"crop-1", b"crop-2"), item_context=(b"vitals",),
        items_truncated=True)

    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert len(client.items) == 1
    request = client.items[0]
    assert request is not None, "the crops the driver enumerated must reach the client"
    assert request.items == (b"crop-1", b"crop-2")
    assert request.context == (b"vitals",)
    assert request.name == "Ada"
    assert request.truncated is True
    assert request.item_count == 2
    # The frames are still on the Profile for the ranker; they are simply not the request.
    assert b"frame-0" not in request.images


def test_auto_keeps_the_frame_shape_for_a_driver_that_enumerates_nothing():
    """A driver with no item space at all -- every non-Hinge driver today -- must be completely
    unaffected: no ItemRequest, no refusal, exactly the pre-existing call. "This capture has no
    items" and "this capture could not produce its items" are different facts and only the
    second one stops a run."""
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), FakeStore(), "s")
    driver = FakeDriver(1)

    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert client.items == [None]
    assert driver.likes == ["hi 1"]


def test_auto_stops_when_the_capture_could_not_produce_numbered_items():
    """The refusal doc 5.2 requires, at the one place the substitution would otherwise happen.
    A driver that enumerates and failed says so on the Profile; the worker must stop with that
    sentence rather than fall back to sending `profile.photos`, which would hand the model a
    numbering nothing downstream can act on. No opener call is made and no like is sent."""
    from operation_love.status import RunStatus

    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), FakeStore(), "s")
    driver = FakeDriver(1)
    driver.cards[0] = Profile(
        photos=[b"frame-0", b"frame-1"],
        items_unavailable="the card is not confirmed to be at its scroll top (confirmed_not_top)")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")

    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "stopped"
    assert app["stop_kind"] == "opener"
    assert "scroll top" in app["stop_reason"]
    assert "numbered items" in app["stop_reason"]
    assert client.calls == 0, "nothing may be billed for a request that cannot be built"
    assert driver.likes == []


def test_auto_stale_hinge_calibration_stops_before_opener_or_like():
    from operation_love.status import RunStatus

    mismatch = ("apps.hinge.targeting_calibration is unavailable (the live app build/frame "
                "geometry does not exactly match schema-v3 calibration "
                "('10.1.0'/(1080, 2400) != '10.0.1'/(1080, 2400)))")
    client = FakeOpenerClient()
    svc = OpenerService(client, CostTracker(PRICING, None), FakeStore(), "s")
    driver = FakeDriver(1)
    driver.cards[0] = Profile(
        photos=[b"frame-0"], items_unavailable=mismatch,
        items_unavailable_kind="targeting_calibration")
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="auto")

    Worker("hinge", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "stopped"
    assert app["stop_kind"] == "targeting_calibration"
    assert "fresh targeting calibration" in app["stop_reason"]
    assert mismatch in app["stop_reason"]
    assert client.items == []
    assert driver.likes == []


def test_auto_stale_live_calibration_stops_before_profile_capture_or_action():
    """AUTO must not pass profiles while its live targeting licence is stale."""
    from operation_love.status import RunStatus

    mismatch = ("apps.hinge.targeting_calibration is unavailable (the live app build/frame "
                "geometry does not exactly match schema-v3 calibration "
                "('10.1.0'/(1080, 2400) != '10.0.1'/(1080, 2400)))")

    class _TestingDriver(FakeDriver):
        accepts_opener = True

        def targeted_suggestion_blocker(self):
            return mismatch

        def next_profile(self):
            pytest.fail("a stale live calibration must stop AUTO before the first capture")

    client, store = FakeOpenerClient(), FakeStore()
    service = OpenerService(client, CostTracker(PRICING, None), store, "s")
    driver = _TestingDriver(1)
    status = RunStatus("run1", ["hinge"], min_labels=1, mode="auto")

    Worker("hinge", driver, FakeDecider("dislike"), service, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "stopped"
    assert app["stop_kind"] == "targeting_calibration"
    assert mismatch in app["stop_reason"]
    assert driver.i == 0
    assert driver.likes == [] and driver.dislikes == 0
    assert client.calls == 0
    assert store.decisions == []
    assert driver.closed


def test_auto_does_not_stop_on_items_unavailable_when_openers_are_disabled():
    """Audit fix, "BUG 2" (2026-08-12): opener.enabled: false must not turn a driver's
    items_unavailable sentence into a hard stop -- there is no consumer for a numbered item list
    when openers are off, so its absence is not a failure. Mirrors
    test_auto_stops_when_the_capture_could_not_produce_numbered_items immediately above with the
    one variable that matters changed: OpenerService(client=None, ...), i.e. `disabled` from
    construction exactly like opener.enabled: false produces (see
    test_worker_with_opener_disabled_by_config_still_likes_normally_in_auto_mode). The driver-side
    fix (HingeDriver.set_opener_enabled) is what stops a real Hinge read from ever setting this
    field in the first place; this test pins the independent worker-side guard that must hold even
    if a driver enumerates anyway (a fake, an older driver, or a future one that forgets the hook)."""
    from operation_love.status import RunStatus

    svc = OpenerService(None, CostTracker(PRICING, None), FakeStore(), "s")   # openers disabled
    driver = FakeDriver(1)
    driver.cards[0] = Profile(
        photos=[b"frame-0", b"frame-1"],
        items_unavailable="the card is not confirmed to be at its scroll top (confirmed_not_top)")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")

    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] != "stopped"
    assert svc.stop_requested is False
    assert driver.likes == [None]     # liked normally, with no opener -- exactly as if openers
                                       # did not exist for this run at all


# --- THE items_unnumbered REGRESSION (found+fixed 2026-08-22, second pass, same day) -----------
# `items_unnumbered` is the third state `Profile` can be in: enumeration ran to completion and
# legitimately numbered nothing (a profile of all video, or a dwell burst that only ever covered
# one card of fifteen -- ops/STILL-PHOTO-DISCRIMINATOR.md 5d). The first pass that introduced it
# made a PASS decision sail past a zero-item profile, correctly. But it did that by leaving the
# whole state invisible to the AUTO loop's opener block -- which meant a LIKE decision fell
# through with `items` left `None`, all the way to `maybe_opener(..., items=None)`. Per that
# method's own docstring, `items=None` means "fall back to the raw scroll frames", exactly the
# ambiguity ops/OPENER-REDESIGN.md 5.2 exists to remove, and a live violation of the owner's
# never-substitute-liked-item rule: there is no verifiable item for that opener to have named.
# These tests pin the fix -- a LIKE decision now stops here, by call count, not just by message --
# and re-confirm the PASS side the first pass already got right.

def test_auto_stops_on_like_when_enumeration_numbered_nothing():
    """The regression itself. `maybe_opener` must never be reached -- not "reached and it
    returned nothing useful", literally never invoked -- and `driver.like` must never be called
    either. Checking call counts is the point: a test that only inspected the stop message would
    have passed on the old, buggy code path too (a stop can still fire later, from `pick is
    None`, after `maybe_opener` already ran on the wrong request)."""
    from operation_love.status import RunStatus

    svc = _SequencedOpenerService([])   # any call at all is a bug; nothing is scripted to return
    driver = FakeDriver(1)
    driver.cards[0] = Profile(
        photos=[b"frame-0"], items=(),
        items_unnumbered="15 selectable card(s) were considered; 15 could not be judged because "
                          "this capture's configured bounded dwell walk did not cover them.")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")

    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    app = status.app_view("bumble")["app"]
    assert app["state"] == "stopped"
    assert app["stop_kind"] == "opener"
    assert svc.calls == [], "maybe_opener must not be invoked when nothing was numbered"
    assert driver.likes == [], "no like may be sent with no verifiable item to attach it to"


def test_auto_does_not_stop_on_pass_when_enumeration_numbered_nothing():
    """The other half of the same profile: a PASS decision is exactly the improvement this
    whole state exists to preserve (a profile of videos is a normal outcome), and it must sail
    past the zero-item profile and keep running -- unaffected by the LIKE-path stop added above,
    because a PASS decision never reaches that opener code at all."""
    svc = _SequencedOpenerService([])
    driver = FakeDriver(2)
    driver.cards[0] = Profile(
        photos=[b"frame-0"], items=(),
        items_unnumbered="15 selectable card(s) were considered; 15 could not be judged because "
                          "this capture's configured bounded dwell walk did not cover them.")

    Worker("bumble", driver, FakeDecider("dislike"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto").run()

    assert driver.dislikes == 2, "both cards were reached -- the run did not stop on card one"
    assert svc.calls == []
    assert driver.out_of_profiles()


def test_items_unnumbered_stop_reason_is_distinguishable_from_items_unavailable():
    """The two stops share `stop_kind == "opener"` (the hub branches on that alone), so the
    `stop_reason` TEXT is the only thing that tells an operator which situation they are in: an
    `items_unavailable` stop means the capture itself failed, while an `items_unnumbered` stop
    means the capture worked and nothing survived policy -- different operator next-steps, so
    the wording must not collide. This also pins that the unnumbered reason quotes the profile's
    own sentence rather than a hardcoded cause (doc STILL-PHOTO-DISCRIMINATOR.md 5d's own
    standing rule: guidance must derive from the condition it describes)."""
    from operation_love.status import RunStatus

    unavailable_status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    unavailable_driver = FakeDriver(1)
    unavailable_driver.cards[0] = Profile(
        photos=[b"frame-0"],
        items_unavailable="the card is not confirmed to be at its scroll top (confirmed_not_top)")
    Worker("bumble", unavailable_driver, FakeDecider("like"), _SequencedOpenerService([]),
           FakeStore(), "run1", _Pacing(), threading.Event(),
           mode="auto", status=unavailable_status).run()
    unavailable_reason = unavailable_status.app_view("bumble")["app"]["stop_reason"]

    unnumbered_status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")
    unnumbered_driver = FakeDriver(1)
    unnumbered_driver.cards[0] = Profile(
        photos=[b"frame-0"], items=(),
        items_unnumbered="15 selectable card(s) were considered; 15 could not be judged because "
                          "this capture's configured bounded dwell walk did not cover them.")
    Worker("bumble", unnumbered_driver, FakeDecider("like"), _SequencedOpenerService([]),
           FakeStore(), "run1", _Pacing(), threading.Event(),
           mode="auto", status=unnumbered_status).run()
    unnumbered_reason = unnumbered_status.app_view("bumble")["app"]["stop_reason"]

    assert "15 selectable card(s) were considered" in unnumbered_reason
    assert unnumbered_reason != unavailable_reason
    assert "confirmed_not_top" not in unnumbered_reason
    assert "15 selectable card(s)" not in unavailable_reason


def test_all_video_profile_reaches_neither_the_opener_nor_the_targeting_path():
    """Re-pins the property `test_all_video_profile_fails_before_opener_or_targeting`
    (tests/test_hinge_item_capture.py, added 2026-08-16) used to encode at the DRIVER level,
    before that test was renamed to `test_all_video_profile_numbers_nothing_but_does_not_stop_the_run`
    on 2026-08-22 (an all-video profile is no longer a driver-level failure -- it is
    `items_unnumbered`, not `items_unavailable`, precisely so a PASS decision does not stop). The
    "fails before opener or targeting" half of that property did not move with it: it belongs
    here, at the WORKER level, and it is specifically a LIKE-decision property, since a PASS
    decision was never at risk of reaching either. `driver.like` is the single call both the
    opener request (via `item_index=`/`model_item_index=`) and the targeting path route through
    -- so asserting it was never invoked pins both at once."""
    from operation_love.status import RunStatus

    svc = _SequencedOpenerService([])
    driver = FakeDriver(1)
    driver.cards[0] = Profile(
        photos=[b"frame-0", b"frame-1"], items=(),
        items_unnumbered="4 selectable card(s) were considered; 4 could not be judged because "
                          "video_mute_v1 matched every one of them.")
    status = RunStatus("run1", ["bumble"], min_labels=1, mode="auto")

    Worker("bumble", driver, FakeDecider("like"), svc, FakeStore(), "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()

    assert svc.calls == [], "the opener path was never reached"
    assert driver.likes == [] and driver.like_item_indexes == [] \
        and driver.like_model_item_indexes == [], "the targeting path was never reached either"
    assert status.app_view("bumble")["app"]["state"] == "stopped"


# --- DOC 5.6'S HARD STOP: a targeting/verification miss stops the run --------------------------
# WHAT THESE REPLACE. Five tests used to live here pinning the `anchored_opener` repair callback:
# the worker handed the driver a closure, and the driver invoked it once its own heart targeting
# missed, to re-ask the model for text about whatever item the sheet had actually opened on. That
# callback was removed on 2026-08-12. It repaired the TEXT while the LIKE still went to an item
# the model never chose, which is exactly the substitution ops/OPENER-REDESIGN.md 5.6 forbids
# ("no falling back to `hearts[0]`, no 'closest reachable item', no rewriting the opener to match
# whatever we hit"). Their coverage is not deleted, it is inverted: the same situations those
# tests described -- targeting missed, or the sheet turned out to show a different item -- are now
# the stop the tests below pin. The removal itself is pinned by FakeDriver.like's signature (no
# such keyword) and by the driver's explicit interface contract tests.
class _TargetingMissDriver(FakeDriver):
    """Hinge-shaped: its like() reports it could not put the like on the chosen item.

    Raises the DRIVER-AGNOSTIC base.ItemTargetingError rather than a Hinge symbol, which is the
    contract the worker is written against -- worker.py must not have to import a concrete driver
    to recognise the one failure class with a specific, non-error stop to render."""
    halt_on_error = True

    def __init__(self, n, *, stage="navigate", intended=4, actual=None,
                 index_space="model_items", message="could not reach it", identity=None):
        super().__init__(n)
        self.snapshotted = []
        self._exc = ItemTargetingError(message, stage=stage, intended=intended, actual=actual,
                                       index_space=index_space)
        # Optional (defaults None, exactly like every pre-existing construction of this class):
        # lets a test prove that a NEVER-SENT discard is attributed exactly like a committed
        # Like would have been, for the same captured profile.
        self._identity = identity

    def current_profile_identity(self):
        return self._identity

    def like(self, opener=None, item_index=None, *, model_item_index=None):
        self.likes.append(opener)          # recorded so a test can prove it was CALLED and failed
        raise self._exc

    def snapshot_failure(self, exc):
        self.snapshotted.append(exc)


def _run_targeting_stop(driver, *, app="bumble"):
    """Run one auto profile against `driver` and hand back the published app status + the store."""
    from operation_love.status import RunStatus
    svc = _SequencedOpenerService([
        OpenerPick("hi 1", index=1, referenced="r", index_space=INDEX_SPACE_PROFILE_PHOTOS),
    ])
    store = FakeStore()
    status = RunStatus("run1", [app], min_labels=1, mode="auto")
    Worker(app, driver, FakeDecider("like"), svc, store, "run1", _Pacing(),
           threading.Event(), mode="auto", status=status).run()
    return status.app_view(app)["app"], store


def test_a_targeting_miss_stops_the_run_instead_of_crashing_it():
    """The headline. The driver could not put the like on the item the opener was written about,
    so it put it nowhere -- and that is a DECISION the bot made correctly under a standing rule,
    not a crash. Left to run()'s generic handler it would render as state="error" with a single
    traceback line, and an operator who cannot tell a correct refusal from a bug soon stops
    reading red banners. It stops, with the reason on the hub.

    stop_kind="targeting" REPLACED "opener" here (the hub workflow, per doc 5.6's own handover
    note). "opener" was the channel the hub already branched on, and its branch is titled
    "opener capacity exhausted": nothing was exhausted, the opener itself is fine, and the
    operator's next move is to go and read the phone rather than to check a quota. The value is
    asserted rather than merely "not None" because the hub renders a different box for each kind
    and an unrecognized one falls back to neutral wording (assets/hub.html)."""
    app, _store = _run_targeting_stop(_TargetingMissDriver(2))

    assert app["state"] == "stopped"
    assert app["stop_kind"] == "targeting"
    assert app["stop_kind"] != "opener"    # not the capacity/exhaustion channel -- see above
    assert app["error"] is None            # NOT an error banner -- this is a clean stop


def test_the_targeting_stop_records_intended_and_actual():
    """Doc 5.6: "stop the run with intended and actual item both recorded". Both come off the
    exception's own fields rather than being parsed back out of its prose, so the stop record
    cannot drift out of agreement with the stop it describes."""
    app, _store = _run_targeting_stop(
        _TargetingMissDriver(2, stage="verify", intended=4, actual=6,
                             message="the like sheet is NOT showing it"))

    reason = app["stop_reason"]
    assert "item 4" in reason and "item 6" in reason
    assert "Intended" in reason and "Actual" in reason
    assert "model_items" in reason                       # which numbering those two are in
    assert "the like sheet is NOT showing it" in reason  # the driver's own words, kept whole


def test_the_targeting_stop_says_so_when_nothing_was_ever_reached():
    """"we could not reach item 4" and "we reached item 6 while aiming at item 4" are different
    diagnoses, and the second is a much stronger signal that the item list itself is wrong. An
    absent `actual` must therefore say it is absent rather than render as a number."""
    app, _store = _run_targeting_stop(_TargetingMissDriver(2, stage="navigate", intended=4,
                                                          actual=None))

    assert "never got far enough" in app["stop_reason"]


def test_the_targeting_stop_does_not_record_the_like_as_a_decision():
    """The like was NOT sent, so counting it would corrupt both the stats and the daily rate
    limit -- the same rule the pre-existing "do not record a phantom if like() raised" comment
    states. Nothing swipes after it either: the run is over, with two profiles left unseen."""
    driver = _TargetingMissDriver(3)
    app, store = _run_targeting_stop(driver)

    assert store.decisions == []
    assert driver.likes == ["hi 1"]        # exactly one attempt, on the first profile
    assert driver.dislikes == 0
    assert app["state"] == "stopped"


def test_the_targeting_stop_still_snapshots_the_screen():
    """The screen IS the diagnosis here -- doc 5.6 leaves it exactly as it was (a scrolled
    profile, or an open sheet with nothing typed) precisely so it can be read back. Catching the
    exception must not cost the failure snapshot run()'s generic handler would have taken, and
    that snapshot has to happen while the transport is still live, i.e. inside the loop."""
    driver = _TargetingMissDriver(2)
    _run_targeting_stop(driver)

    assert len(driver.snapshotted) == 1
    assert isinstance(driver.snapshotted[0], ItemTargetingError)


def test_training_worker_without_hub_fails_before_opening_the_device_session():
    driver = FakeDriver(1)
    store = FakeStore()
    stop = threading.Event()
    Worker("hinge", driver, FakeDecider("like"), None, store, "run1", _Pacing(), stop,
           mode="training").run()

    assert stop.is_set()
    assert driver.opened is False
    assert driver.likes == [] and driver.dislikes == 0


def test_training_closes_the_driver_when_the_device_action_raises_unexpectedly():
    """An unexpected error out of driver.like() in TRAINING must still tear the driver down.

    Regression cover for the 2026-08-28 halt, which raised UnlocatedControlError from
    `driver.like()`. AUTO's equivalent is already pinned
    (test_auto_mode_halts_on_unexpected_even_when_a_driver_opts_out_of_halting); training's was
    not, and it matters more now: since 2026-08-28 the shipped touch_backend is
    `uhid_persistent`, whose virtual touchscreen lives for the WHOLE session and is unregistered
    by `HingeDriver.close()`. A crash that skipped teardown would leave a real `og_touch_*`
    input device registered on the phone, once per crashed run.

    The opener pick below carries index + INDEX_SPACE_MODEL_ITEMS deliberately: without a target
    the loop breaks at worker.py:532 BEFORE like() is ever called, and every assertion here would
    still pass with nothing raised at all. `raised` and the error state are what make this test
    discriminate the crash path from the clean stop.
    """
    from operation_love.status import RunStatus
    from operation_love.training_actions import TrainingActionBridge

    raised = []

    class _BoomDriver(FakeDriver):
        # All three are required to clear _training_loop's preflight; without them the loop
        # returns BEFORE open_session() and this test proves nothing.
        accepts_opener = True
        supports_training_decision = True
        halt_on_error = False           # even an explicit opt-out must still be closed

        def set_training_decision(self, approval):
            self._approval = approval

        def like(self, opener=None, item_index=None, *, model_item_index=None):
            raised.append((opener, model_item_index))
            raise RuntimeError("the training checkpoint does not show Hinge's pass control")

    class _Service:                      # must name a target, or the loop stops before like()
        disabled = False
        stop_requested = False
        last_skip_reason = None
        last_skip_allows_commentless_like = False

        def maybe_opener(self, *_args, **_kwargs):
            return SimpleNamespace(text="A complete opener", index=1,
                                   index_space=INDEX_SPACE_MODEL_ITEMS)

    status = RunStatus("run1", ["hinge"], min_labels=1, mode="training")
    driver, store, stop = _BoomDriver(1), FakeStore(), threading.Event()
    Worker("hinge", driver, FakeDecider("like"), _Service(), store, "run1", _Pacing(), stop,
           mode="training", status=status,
           training_action_bridge=TrainingActionBridge()).run()

    app = status.snapshot()["apps"]["hinge"]
    assert driver.opened                        # the session really opened, so a transport exists
    assert raised == [("A complete opener", 1)]  # like() WAS reached
    # THE discriminating assertion. state == "error" alone is NOT enough: a driver that returns
    # None instead of raising also halts, via worker.py:574's "training driver returned no
    # verified Like/Dislike outcome". Only the propagated message proves the DEVICE action's own
    # exception is what tore the session down.
    assert app["state"] == "error"
    assert "does not show Hinge's pass control" in str(app["error"])
    assert stop.is_set()                        # supervisor flushes buffered data
    assert driver.likes == []                   # nothing was recorded as sent
    assert driver.closed                        # <-- the persistent virtual touchscreen is released
