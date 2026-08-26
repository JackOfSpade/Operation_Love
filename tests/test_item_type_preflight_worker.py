"""Worker seams for doc 5.8: no gesture on AUTO before a type mismatch stops it."""
import threading

from operation_love.config import PacingCfg
from operation_love.drivers.base import DatingAppDriver
from operation_love.drivers.item_type_preflight import (
    MISMATCH,
    PHOTO,
    WRITTEN,
    ItemTypePreflight,
)
from operation_love.opener.opener import INDEX_SPACE_PROFILE_PHOTOS
from operation_love.opener.service import OpenerPick
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import Decision
from operation_love.worker import Worker


class _PreflightDriver(DatingAppDriver):
    accepts_opener = True

    def __init__(self):
        self.profile = Profile(photos=[b"ranker"], items=[b"numbered crop"])
        self.seen = False
        self.likes = 0
        self.preflight_calls = 0
        self.closed = False

    def open_session(self):
        pass

    def next_profile(self):
        if self.seen:
            return None
        self.seen = True
        return self.profile

    def like(self, opener=None, item_index=None, *, model_item_index=None):
        self.likes += 1
        raise AssertionError("doc 5.8 mismatch must stop before like()")

    def dislike(self):
        raise AssertionError("the model selected like")

    def out_of_profiles(self):
        return self.seen

    def item_type_preflight(self, item_description, model_item_index):
        self.preflight_calls += 1
        assert item_description == "a prompt answer"
        assert model_item_index == 1
        return ItemTypePreflight(MISMATCH, WRITTEN, PHOTO, "model selected a written prompt")

    def close(self):
        self.closed = True


class _LikeDecider:
    def decide(self, profile):
        return Decision("like", 0.95, [0.1], "test")


class _Store:
    def count_today(self, app):
        return 0

    def record_decision(self, *args, **kwargs):
        raise AssertionError("an unsent like must not be recorded")


class _OnePickService:
    disabled = False
    stop_requested = False
    last_skip_reason = None
    exhausted_reason = None

    def maybe_opener(self, *args, **kwargs):
        return OpenerPick("hello", index=1, item_description="a prompt answer")


def test_auto_refuses_a_model_pick_that_calls_a_numbered_photo_a_prompt_before_like():
    driver = _PreflightDriver()
    worker = Worker("test", driver, _LikeDecider(), _OnePickService(), _Store(), "run",
                    PacingCfg(swipe_delay_s=0), threading.Event(), mode="auto")
    worker.run()

    assert driver.preflight_calls == 1
    assert driver.likes == 0
    assert worker.stop_event.is_set()
    assert driver.closed


def test_auto_legacy_profile_photo_pick_skips_model_crop_preflight():
    """A capture-order number is not a numbered-crop number and must never be cross-checked.

    The legacy branch still has its own explicitly translated targeting path.  Invoking the
    model-item preflight hook with its 1-based raw-frame number would inspect an unrelated crop
    and can manufacture a false hard stop, so this driver treats such a call as a test failure.
    """
    class LegacyDriver(_PreflightDriver):
        def __init__(self):
            super().__init__()
            self.like_args = []

        def item_type_preflight(self, *args):
            raise AssertionError("legacy capture-order pick reached model-item preflight")

        def like(self, opener=None, item_index=None, *, model_item_index=None):
            self.like_args.append((opener, item_index, model_item_index))

    class LegacyStore:
        def __init__(self):
            self.decisions = []

        def count_today(self, app):
            return 0

        def record_decision(self, *args, **kwargs):
            self.decisions.append((args, kwargs))

    class LegacyService(_OnePickService):
        def maybe_opener(self, *args, **kwargs):
            return OpenerPick("hello", index=1, item_description="a photo",
                              index_space=INDEX_SPACE_PROFILE_PHOTOS)

    driver = LegacyDriver()
    store = LegacyStore()
    worker = Worker("test", driver, _LikeDecider(), LegacyService(), store, "run",
                    PacingCfg(swipe_delay_s=0), threading.Event(), mode="auto")
    worker.run()

    assert driver.like_args == [("hello", 0, None)]
    assert len(store.decisions) == 1
    assert not worker.stop_event.is_set() and driver.closed
