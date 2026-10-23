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


@pytest.mark.parametrize("frame", [b"invalid", _action_frame(), _modal_frame()])
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
