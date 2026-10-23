"""Resume a scrolled Training composer by restoring its viewport, never by weakening proof."""

from types import SimpleNamespace

import pytest

from operation_love.drivers import hinge
from operation_love.drivers.base import ActionCancelled
from tests.test_hinge_auto_presend_snapshot import _driver
from tests.test_like_composer import (
    _keyboard_open_composer,
    _scrolled_up_priority_like_composer,
)


@pytest.fixture
def world(monkeypatch):
    driver, _ = _driver([])
    driver._current_item_index = SimpleNamespace(identity=object())
    monkeypatch.setattr(
        driver,
        "_require_targeting_calibration",
        lambda _: SimpleNamespace(identity_match_max_dist=1.8125),
    )
    monkeypatch.setattr(
        hinge,
        "compare_profile_identity",
        lambda *_a, **_kw: SimpleNamespace(matched=True),
    )
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    return driver


def test_scrolled_resume_restores_then_verifies_fresh_frame(world, monkeypatch):
    scrolled = _scrolled_up_priority_like_composer()
    restored = _keyboard_open_composer()
    frames = iter([scrolled, restored])
    swipes = []
    verified = []
    monkeypatch.setattr(world, "_screencap", lambda: next(frames))
    monkeypatch.setattr(world, "_swipe", lambda *args, **kwargs: swipes.append(args))
    monkeypatch.setattr(world, "_locate_training_pass", lambda _: (115, 2060))
    monkeypatch.setattr(
        world, "_verify_sheet_shows", lambda frame, *_a, **_kw: verified.append(frame)
    )
    frame, composer, point = world._verify_training_checkpoint(
        scrolled,
        object(),
        1,
        b"before",
        keyboard_recovery=True,
    )
    assert frame == restored
    assert len(swipes) == 2
    assert verified == [restored]
    assert composer.confirm_point[1] > 1296
    assert point == (115, 2060)
    # Touch-down and release both stay above the input, far from either send control.
    assert all(240 < y0 < y1 < 778 for _x0, y0, _x1, y1 in swipes)


@pytest.mark.parametrize("matched", [False, True])
def test_resume_never_skips_final_item_proof(world, monkeypatch, matched):
    scrolled = _scrolled_up_priority_like_composer()
    restored = _keyboard_open_composer()
    swipes = []
    monkeypatch.setattr(
        hinge,
        "compare_profile_identity",
        lambda *_a, **_kw: SimpleNamespace(matched=matched),
    )
    monkeypatch.setattr(world, "_screencap", lambda: restored if matched else scrolled)
    monkeypatch.setattr(world, "_swipe", lambda *args, **kw: swipes.append(args))

    def refuse_item(*args, **kwargs):
        raise hinge.HingeTargetingError(
            "wrong item", stage="verify", intended=1, index_space="model_items"
        )

    monkeypatch.setattr(world, "_verify_sheet_shows", refuse_item)
    with pytest.raises(
        hinge.HingeActionError, match="wrong item" if matched else "no strictly located"
    ):
        world._verify_training_checkpoint(
            scrolled, object(), 1, b"before", keyboard_recovery=True
        )
    assert len(swipes) == (1 if matched else 0)


def test_stuck_viewport_has_bounded_navigation(world, monkeypatch):
    scrolled = _scrolled_up_priority_like_composer()
    swipes = []
    monkeypatch.setattr(world, "_screencap", lambda: scrolled)
    monkeypatch.setattr(world, "_swipe", lambda *args, **kw: swipes.append(args))
    with pytest.raises(hinge.UnlocatedControlError, match="no strictly located"):
        world._verify_training_checkpoint(
            scrolled, object(), 1, b"before", keyboard_recovery=True
        )
    assert len(swipes) == hinge._RESUME_SETTLE_ATTEMPTS


def test_stop_prevents_viewport_navigation(world, monkeypatch):
    scrolled = _scrolled_up_priority_like_composer()
    monkeypatch.setattr(world, "_swipe", lambda *a, **kw: pytest.fail("stopped"))
    with pytest.raises(ActionCancelled):
        world._verify_training_checkpoint(
            scrolled,
            object(),
            1,
            b"before",
            keyboard_recovery=True,
            should_stop=lambda: True,
        )


def test_pre_review_checkpoint_never_restores_viewport(world, monkeypatch):
    monkeypatch.setattr(world, "_swipe", lambda *a, **kw: pytest.fail("no review yet"))
    with pytest.raises(hinge.UnlocatedControlError, match="no strictly located"):
        world._verify_training_checkpoint(
            _scrolled_up_priority_like_composer(), object(), 1, b"before"
        )


def test_incomplete_composer_does_not_authorise_navigation(world, monkeypatch):
    frame = _scrolled_up_priority_like_composer(input_present=False)
    monkeypatch.setattr(world, "_swipe", lambda *a, **kw: pytest.fail("not proven"))
    monkeypatch.setattr(world, "_screencap", lambda: frame)
    with pytest.raises(hinge.UnlocatedControlError):
        world._verify_training_checkpoint(
            frame, object(), 1, b"before", keyboard_recovery=True
        )
