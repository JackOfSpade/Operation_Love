"""Real 10.9.0 warning chrome; all profile/message/account pixels removed."""

import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from operation_love.drivers import hinge
from operation_love.drivers.base import ActionCancelled
from operation_love.drivers.debuglog import DebugLog
from tests.test_hinge_observe import FakeAdb, _drv, _action_frame, _modal_frame


WARNING = (Path(__file__).parent / "assets/hinge_message_warning.png").read_bytes()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(hinge.time, "sleep", lambda *_: None)


def _erase(box):
    image = Image.open(io.BytesIO(WARNING))
    ImageDraw.Draw(image).rectangle(box, fill="white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_real_warning_locates_send_anyway():
    assert hinge._message_warning_controls(WARNING) == (True, (540, 2202))


def test_confirmation_follows_shifted_control_instead_of_fixed_coordinates():
    image = Image.open(io.BytesIO(WARNING))
    shifted = Image.new("RGB", image.size, "white")
    shifted.paste(image, (25, -30))
    output = io.BytesIO()
    shifted.save(output, format="PNG")
    assert hinge._message_warning_controls(output.getvalue()) == (True, (565, 2172))


@pytest.mark.parametrize(
    "frame",
    [b"invalid", _action_frame(), _modal_frame()],
    ids=["undecodable", "deck", "rose_modal"],
)
def test_other_screens_do_not_authorise_confirmation(frame):
    assert hinge._message_warning_controls(frame) == (False, None)


@pytest.mark.parametrize("box", [(390, 2015, 690, 2070), (400, 2175, 680, 2230)])
def test_partial_warning_refuses_tap(box):
    frame = _erase(box)
    assert hinge._message_warning_controls(frame) == (True, None)
    adb = FakeAdb([frame])
    with pytest.raises(hinge.HingeActionError, match="could not be verified"):
        _drv(adb)._verify_like_landed(b"before", confirm_message_warning=True)
    assert adb.taps == []


@pytest.mark.parametrize("halt_on_error", [False, True])
@pytest.mark.parametrize("late", [False, True])
def test_confirmation_once_then_verify_delivery(
    monkeypatch, tmp_path, halt_on_error, late
):
    deck = _action_frame()
    frames = ([deck] if late else []) + [WARNING, deck, deck, deck]
    adb = FakeAdb(frames, advance_on_screencap=True)
    driver = _drv(adb, halt_on_error=halt_on_error)
    driver._dbg = DebugLog(str(tmp_path), run_id="warning")
    monkeypatch.setattr(driver, "_observe_like_sheet_visible", lambda _: False)
    monkeypatch.setattr(driver, "_changed", lambda *_: True)
    driver._verify_like_landed(b"before", confirm_message_warning=True)
    assert adb.taps == [(540, 2202)]
    rows = [
        json.loads(row)
        for row in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    warning = next(row for row in rows if row["action"] == "message_warning")
    assert warning["outcome"] == "send_anyway_attempt"
    assert (driver._dbg.dir / warning["kept_before"]).exists()


@pytest.mark.parametrize("halt_on_error", [False, True])
def test_stuck_warning_stops_without_resending(halt_on_error):
    adb = FakeAdb([WARNING])
    with pytest.raises(hinge.HingeActionError, match="message warning is still open"):
        _drv(adb, halt_on_error=halt_on_error)._verify_like_landed(
            b"before", confirm_message_warning=True
        )
    assert adb.taps == [(540, 2202)]


def test_stop_blocks_confirmation():
    adb = FakeAdb([WARNING])
    with pytest.raises(ActionCancelled):
        _drv(adb)._verify_like_landed(
            b"before", should_stop=lambda: True, confirm_message_warning=True
        )
    assert adb.taps == []


def test_paywall_wins_over_warning(monkeypatch):
    adb = FakeAdb([WARNING])
    driver = _drv(adb)
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _: "paywall")
    with pytest.raises(hinge.HingeDeckBlockedError, match="paywall"):
        driver._verify_like_landed(b"before")
    assert adb.taps == []


def test_paywall_after_confirmation_is_not_delivery(monkeypatch):
    adb = FakeAdb([WARNING, b"paywall"], advance_on_screencap=True)
    driver = _drv(adb)
    monkeypatch.setattr(
        driver,
        "_deck_blocked_reason",
        lambda frame: "paywall" if frame == b"paywall" else None,
    )
    with pytest.raises(hinge.HingeDeckBlockedError, match="paywall"):
        driver._verify_like_landed(b"before", confirm_message_warning=True)
    assert adb.taps == [(540, 2202)]


def test_closed_warning_does_not_replace_advance_proof(monkeypatch):
    adb = FakeAdb([WARNING, _action_frame()], advance_on_screencap=True)
    driver = _drv(adb, halt_on_error=True)
    monkeypatch.setattr(driver, "_observe_like_sheet_visible", lambda _: False)
    monkeypatch.setattr(driver, "_changed", lambda *_: False)
    with pytest.raises(hinge.HingeActionError, match="did not change the screen"):
        driver._verify_like_landed(b"before", confirm_message_warning=True)
    assert adb.taps == [(540, 2202)]


@pytest.mark.parametrize("halt_on_error", [False, True])
@pytest.mark.parametrize(
    "frame",
    [WARNING, _erase((400, 2175, 680, 2230))],
    ids=["recognised", "unrecognised_controls"],
)
def test_unreviewed_warning_is_refused_without_any_tap(tmp_path, halt_on_error, frame):
    """AUTO: nobody read the opener Hinge flagged, so nothing is tapped, recognised or not."""
    adb = FakeAdb([frame])
    driver = _drv(adb, halt_on_error=halt_on_error)
    driver._dbg = DebugLog(str(tmp_path), run_id="unreviewed")
    with pytest.raises(hinge.HingeActionError, match="Nobody reviewed it"):
        driver._verify_like_landed(b"before")
    assert adb.taps == []
    rows = [
        json.loads(row)
        for row in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    warning = next(row for row in rows if row["action"] == "message_warning")
    assert warning["outcome"] == "unreviewed_refused"
    assert (driver._dbg.dir / warning["kept_before"]).exists()


def test_unreviewed_refusal_still_lets_a_paywall_win(monkeypatch):
    adb = FakeAdb([WARNING])
    driver = _drv(adb)
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _: "paywall")
    with pytest.raises(hinge.HingeDeckBlockedError, match="paywall"):
        driver._verify_like_landed(b"before")
    assert adb.taps == []


def _blurred_warning(sigma=0.5):
    """The real warning, blurred just enough to lift its 'Send anyway' button past 0.6 against
    the Rose modal's 'Send Like anyway' template (0.5953 unblurred, 0.6022 at sigma 0.5)."""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(WARNING, np.uint8), cv2.IMREAD_COLOR)
    ok, buf = cv2.imencode(".png", cv2.GaussianBlur(image, (0, 0), sigma))
    return buf.tobytes()


def _upsell_score(frame):
    for threshold in (0.99, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5):
        if hinge._match_glyph(
            frame,
            hinge._load_template("hinge_send_like_anyway.png"),
            side="any",
            threshold=threshold,
        ):
            return threshold
    return 0.0


def test_upsell_button_floor_separates_the_warning_from_a_rose_modal():
    """Pins the measured margin both ways: the warning (and a blurred one, which crossed the
    old 0.6 floor) stays below the floor, and a Rose modal stays above it."""
    assert _upsell_score(_blurred_warning()) >= 0.6  # the case the old floor would tap
    assert _upsell_score(WARNING) < hinge._UPSELL_BUTTON_MATCH_THRESHOLD
    assert _upsell_score(_blurred_warning()) < hinge._UPSELL_BUTTON_MATCH_THRESHOLD
    assert _upsell_score(_modal_frame()) >= hinge._UPSELL_BUTTON_MATCH_THRESHOLD


@pytest.mark.parametrize(
    "frame", [WARNING, _blurred_warning()], ids=["real", "blurred"]
)
def test_rose_helper_never_taps_a_message_warning(frame):
    adb = FakeAdb([frame, frame])
    assert _drv(adb)._handle_rose_upsell() is False
    assert adb.taps == []


def test_rose_helper_skips_warning_even_if_the_glyph_would_match(monkeypatch):
    """The structural guard, independent of any threshold: on a recognised warning the helper
    does not even ask the template."""
    adb = FakeAdb([WARNING, WARNING])
    monkeypatch.setattr(hinge, "_match_glyph", lambda *_a, **_k: [(464, 2197)])
    assert _drv(adb)._handle_rose_upsell() is False
    assert adb.taps == []


@pytest.mark.parametrize(
    "frame", [WARNING, _blurred_warning()], ids=["real", "blurred"]
)
def test_auto_post_send_order_refuses_a_warning_without_any_tap(frame):
    """AUTO's real order after the Send tap: Rose helper, then the landing check. Neither may
    tap the warning; the landing check refuses it."""
    adb = FakeAdb([frame, frame, frame])
    driver = _drv(adb)
    assert driver._handle_rose_upsell() is False
    with pytest.raises(hinge.HingeActionError, match="Nobody reviewed it"):
        driver._verify_like_landed(b"before")
    assert adb.taps == []
