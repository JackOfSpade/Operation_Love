"""AUTO's retained post-type / pre-Send-Like evidence.

This intentionally exercises the real comment-sheet method with its vision/navigation pieces
stubbed at their boundaries.  The assertion is about the action order: the private typed-composer
frame must be written before the irreversible Send Like touch, and remain available even when
ordinary debug shots rotate.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from operation_love.drivers.debuglog import HingeDebugLog
from operation_love.drivers import hinge
from operation_love.drivers.hinge import HingeActionError, HingeDriver, UnlocatedControlError


class _Adb:
    def __init__(self, events):
        self.events = events
        self.texts: list[str] = []

    def screen_size(self):
        return (1080, 2400)

    def devices(self):
        return ["pixel"]

    def shell(self, *_args, **_kwargs):
        return ""

    def foreground_package(self):
        return "co.hinge.app"

    def screencap(self):
        self.events.append(("screencap", None))
        return b"TYPED_OPENER_ON_SELECTED_ITEM"

    def tap(self, x, y):
        self.events.append(("tap", (x, y)))

    def text(self, text):
        self.texts.append(text)

    def keyevent(self, keycode):
        self.events.append(("keyevent", keycode))


def _driver(events):
    class _Config:
        apps = {"hinge": {"serial": "pixel", "halt_on_error": False}}

    adb = _Adb(events)
    driver = HingeDriver(_Config())
    driver._adb = adb
    driver._touch = adb
    return driver, adb


def test_auto_retains_typed_target_snapshot_before_send_touch(tmp_path, monkeypatch):
    events = []
    driver, adb = _driver(events)
    driver.set_auto_session_policy(object())
    debug = HingeDebugLog(str(tmp_path), run_id="r", keep_shots=50)
    original_action = debug.action

    def recorded_action(name, **kwargs):
        events.append(("debug", name))
        return original_action(name, **kwargs)

    debug.action = recorded_action
    driver._dbg = debug
    initial_composer = SimpleNamespace(
        comment_rect=SimpleNamespace(center=(20, 20)), confirm_point=(30, 30))
    fresh_composer = SimpleNamespace(
        comment_rect=SimpleNamespace(center=(21, 21)), confirm_point=(40, 40))
    payload = object()

    # The actual test is the post-typing/send boundary, not target-navigation vision.  These
    # stubs establish the already-verified target path that reaches that boundary.
    monkeypatch.setattr(driver, "_verifiable_payload", lambda _item: payload)
    monkeypatch.setattr(driver, "_confirm_payload_profile", lambda _item: None)
    monkeypatch.setattr(driver, "_navigate_to_model_item", lambda _item, **_kw: (10, 10))
    monkeypatch.setattr(driver, "_snap", lambda: b"PRE_HEART_CARD")
    monkeypatch.setattr(driver, "_await_sheet_open", lambda **_kw: initial_composer)
    monkeypatch.setattr(driver, "_screencap", adb.screencap)
    monkeypatch.setattr(
        hinge, "locate_inline_composer", lambda *_args, **_kw: initial_composer)
    def locate_fresh_composer(_frame):
        # This marker is immediately after the exact frame from which the final Send point was
        # located.  No diagnostic capture may happen until that point is tapped.
        events.append(("pre_send_frame_located", None))
        return fresh_composer

    monkeypatch.setattr(driver, "_locate_inline_composer", locate_fresh_composer)
    monkeypatch.setattr(driver, "_require_targeting_calibration", lambda _item: SimpleNamespace(
        identity_match_max_dist=3.0, inline_item_max_dist=4.0))
    driver._current_item_index = SimpleNamespace(identity=object())
    monkeypatch.setattr(hinge, "compare_profile_identity", lambda *_args, **_kw: SimpleNamespace(
        state="match", distance=0.1, match_max=3.0, mismatched=False, unknown=False))
    monkeypatch.setattr(hinge, "verify_sheet_item", lambda *_args, **_kw: SimpleNamespace(
        state="match", nearest_index=1, distance=0.2, bound=4.0, matched=True,
        preview=SimpleNamespace(y0=1, y1=2, x0=3, x1=4)))
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_args, **_kw: True)
    monkeypatch.setattr(driver, "_handle_rose_upsell", lambda: False)
    monkeypatch.setattr(driver, "_verify_like_landed", lambda _before: None)

    opener = "A specifically targeted opener"
    driver.like(opener, model_item_index=1)

    records = [json.loads(line) for line in (debug.dir / "actions.jsonl").read_text().splitlines()]
    pre_send = next(record for record in records if record["action"] == "auto_opener_pre_send")
    assert pre_send["opener"] == opener
    assert pre_send["opener_chars"] == len(opener)
    assert pre_send["opener_sha256"] == hashlib.sha256(opener.encode()).hexdigest()
    assert pre_send["frame_sha256"] == hashlib.sha256(
        b"TYPED_OPENER_ON_SELECTED_ITEM").hexdigest()
    expected_evidence_id = hashlib.sha256(
        b"TYPED_OPENER_ON_SELECTED_ITEM\0" + opener.encode()).hexdigest()
    assert pre_send["evidence_id"] == expected_evidence_id
    assert pre_send["model_item_index"] == 1
    assert pre_send["kept_before"] == pre_send["before"]
    assert (debug.dir / pre_send["before"]).read_bytes() == b"TYPED_OPENER_ON_SELECTED_ITEM"
    assert events.index(("debug", "auto_opener_pre_send")) < events.index(("tap", (40, 40)))
    assert ("tap", (30, 30)) not in events
    # The final control is re-located on the exact post-type frame. Once that frame is located,
    # no later diagnostic capture may race the touch.
    final_frame_located = max(
        i for i, event in enumerate(events) if event == ("pre_send_frame_located", None))
    send_tap = events.index(("tap", (40, 40)))
    assert ("screencap", None) not in events[final_frame_located + 1:send_tap]

    # The final target proof must use exactly the post-type frame too.  This catches the former
    # `_dbg_action` recapture race independently from the retained evidence record above.
    final_identity = [r for r in records if r["action"] == "verify_sheet_identity"][-1]
    final_item = [r for r in records if r["action"] == "verify_sheet_item"][-1]
    for record in (final_identity, final_item):
        assert (debug.dir / record["after"]).read_bytes() == b"TYPED_OPENER_ON_SELECTED_ITEM"
    for action in ("like_attempt", "like"):
        result = next(record for record in records if record["action"] == action)
        assert result["pre_send_evidence_id"] == expected_evidence_id
    durable = driver.landed_auto_opener_evidence()
    assert durable is not None
    assert durable["evidence_id"] == expected_evidence_id
    assert durable["frame"] == b"TYPED_OPENER_ON_SELECTED_ITEM"
    assert durable["model_item_index"] == 1

def test_auto_refuses_send_when_fresh_presend_frame_has_no_composer(monkeypatch):
    events = []
    driver, adb = _driver(events)
    driver.set_auto_session_policy(object())
    initial_composer = SimpleNamespace(
        comment_rect=SimpleNamespace(center=(20, 20)), confirm_point=(30, 30))

    monkeypatch.setattr(driver, "_verifiable_payload", lambda _item: None)
    monkeypatch.setattr(driver, "_navigate_to_model_item", lambda _item, **_kw: (10, 10))
    monkeypatch.setattr(driver, "_snap", lambda: b"PRE_HEART_CARD")
    monkeypatch.setattr(driver, "_await_sheet_open", lambda **_kw: initial_composer)
    monkeypatch.setattr(driver, "_screencap", adb.screencap)
    monkeypatch.setattr(
        hinge, "locate_inline_composer", lambda *_args, **_kw: initial_composer)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: None)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_args, **_kw: True)

    with pytest.raises(UnlocatedControlError, match="exact post-type pre-send frame"):
        driver.like("A specifically targeted opener", model_item_index=1)

    # The earlier heart and comment-focus taps are reversible prerequisites.  The Send Like
    # control from either the initial or the hypothetical fresh geometry is never tapped.
    assert ("tap", (30, 30)) not in events


def test_training_relocates_and_rechecks_the_scrolled_composer_after_decision(
        tmp_path, monkeypatch):
    """Training approves the target, not coordinates captured before human review.

    The owner may scroll around the selected item while the Training review
    card is open, then return to a fully visible Send Priority Like control at a
    different vertical position.  The irreversible touch has to use a freshly
    located AND item-verified post-review frame; no later capture may separate
    that proof from the tap.
    """
    events = []
    driver, adb = _driver(events)
    driver.set_auto_session_policy(object())
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="shifted-review", keep_shots=50)
    initial = SimpleNamespace(comment_rect=SimpleNamespace(center=(20, 20)),
                              confirm_point=(30, 30))
    pre_review = SimpleNamespace(comment_rect=SimpleNamespace(center=(21, 21)),
                                 confirm_point=(40, 40))
    # Measured in the report's actual post-review error frame: the whole
    # Priority Like composer was still visible, but its action point had moved
    # from the keyboard-open `(690, 1390)` to `(690, 1882)`.
    post_review = SimpleNamespace(comment_rect=SimpleNamespace(center=(540, 1705)),
                                  confirm_point=(690, 1882))
    pre_frame = b"TYPED_OPENER_BEFORE_HUMAN_REVIEW"
    post_frame = b"TYPED_OPENER_AFTER_HUMAN_SCROLL_REVIEW"
    opener = "A specifically targeted opener"
    reviewed = {"done": False}
    verification_frames = []

    def screencap():
        frame = post_frame if reviewed["done"] else pre_frame
        events.append(("screencap", frame))
        return frame

    def locate(frame):
        events.append(("locate", frame))
        return post_review if frame == post_frame else pre_review

    def verify(sheet, *_args, composer_surface=None, **_kwargs):
        verification_frames.append((sheet, composer_surface))
        events.append(("verify", sheet))

    monkeypatch.setattr(driver, "_verifiable_payload", lambda _item: object())
    monkeypatch.setattr(driver, "_confirm_payload_profile", lambda _item: None)
    monkeypatch.setattr(driver, "_navigate_to_model_item", lambda _item, **_kw: (10, 10))
    monkeypatch.setattr(driver, "_snap", lambda: b"PRE_HEART_CARD")
    monkeypatch.setattr(driver, "_await_sheet_open", lambda **_kw: initial)
    monkeypatch.setattr(driver, "_screencap", screencap)
    monkeypatch.setattr(hinge, "locate_inline_composer", lambda *_args, **_kw: initial)
    monkeypatch.setattr(driver, "_locate_inline_composer", locate)
    monkeypatch.setattr(driver, "_verify_sheet_shows", verify)
    monkeypatch.setattr(driver, "_hide_keyboard_for_training", lambda **_kw: pre_frame)
    def training_checkpoint(frame, *_args, **_kwargs):
        composer = locate(frame)
        verify(frame, composer_surface=composer)
        return composer, (80, 80)
    monkeypatch.setattr(driver, "_verify_training_checkpoint", training_checkpoint)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_args, **_kw: True)
    monkeypatch.setattr(driver, "_handle_rose_upsell", lambda: False)
    monkeypatch.setattr(driver, "_verify_like_landed", lambda _before: None)
    monkeypatch.setattr(driver, "_verify_training_like_landed", lambda _item: "identity")

    def decide(frame, evidence):
        assert frame == pre_frame
        events.append(("decision", frame))
        reviewed["done"] = True
        return "like"

    driver.set_training_decision(decide)
    driver.like(opener, model_item_index=1)

    assert ("tap", post_review.confirm_point) in events
    assert ("tap", pre_review.confirm_point) not in events
    assert verification_frames[-1] == (post_frame, post_review)
    final_locate = events.index(("locate", post_frame))
    final_verify = events.index(("verify", post_frame))
    final_tap = events.index(("tap", post_review.confirm_point))
    assert final_locate < final_verify < final_tap
    assert not any(event[0] == "screencap" for event in events[final_verify + 1:final_tap])

    # Decision provenance is immutable even though the live sheet moved. The first row is the
    # exact image/ID the reviewer approved; the resumed row is a separate exact image/ID for the
    # re-verified frame whose coordinates were actually touched, explicitly linked back to it.
    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    opener_bytes = opener.encode()
    expected_approval_id = hashlib.sha256(pre_frame + b"\0" + opener_bytes).hexdigest()
    expected_resumed_id = hashlib.sha256(post_frame + b"\0" + opener_bytes).hexdigest()
    approval_record = next(
        record for record in records if record["action"] == "auto_opener_pre_send")
    resumed_record = next(
        record for record in records if record["action"] == "auto_opener_resumed_send")

    assert approval_record["evidence_id"] == expected_approval_id
    assert approval_record["session_mode"] == "training"
    assert approval_record["frame_sha256"] == hashlib.sha256(pre_frame).hexdigest()
    assert approval_record["kept_before"] == approval_record["before"]
    assert (driver._dbg.dir / approval_record["before"]).read_bytes() == pre_frame

    assert resumed_record["evidence_id"] == expected_resumed_id
    assert resumed_record["evidence_id"] != approval_record["evidence_id"]
    assert resumed_record["session_mode"] == "training"
    assert resumed_record["approval_evidence_id"] == expected_approval_id
    assert resumed_record["frame_sha256"] == hashlib.sha256(post_frame).hexdigest()
    assert resumed_record["kept_before"] == resumed_record["before"]
    assert (driver._dbg.dir / resumed_record["before"]).read_bytes() == post_frame

    for action in ("like_attempt", "like"):
        action_record = next(record for record in records if record["action"] == action)
        assert action_record["pre_send_evidence_id"] == expected_approval_id
        assert action_record["resumed_send_evidence_id"] == expected_resumed_id

    durable = driver.landed_auto_opener_evidence()
    assert durable is not None
    assert durable["evidence_id"] == expected_resumed_id
    assert durable["approval_evidence_id"] == expected_approval_id
    assert durable["frame"] == post_frame
    assert durable["frame_sha256"] == hashlib.sha256(post_frame).hexdigest()
    assert durable["model_item_index"] == 1


def test_presend_snapshot_is_not_written_outside_an_auto_session(tmp_path):
    events = []
    driver, _adb = _driver(events)
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="r")

    driver._record_auto_opener_pre_send(
        b"TYPED_OPENER_ON_SELECTED_ITEM", opener="text", item_index=None,
        model_item_index=1)

    assert not (driver._dbg.dir / "actions.jsonl").exists()
    assert list(driver._dbg.dir.glob("*.png")) == []


def test_training_hides_keyboard_and_freshly_revalidates_like_after_hub_choice(monkeypatch):
    """The training callback sees the post-hide frame, never its stale send coordinate."""
    events = []
    driver, adb = _driver(events)
    driver.set_auto_session_policy(object())
    initial = SimpleNamespace(comment_rect=SimpleNamespace(center=(20, 20)),
                              confirm_point=(30, 30))
    checkpoint = SimpleNamespace(comment_rect=SimpleNamespace(center=(21, 21)),
                                 confirm_point=(40, 40))
    resumed = SimpleNamespace(comment_rect=SimpleNamespace(center=(22, 22)),
                              confirm_point=(50, 50))
    frames = iter((b"SHEET", b"FOCUSED", b"KEYBOARD_HIDDEN", b"FRESH_AFTER_CHOICE"))
    seen = []

    monkeypatch.setattr(driver, "_verifiable_payload", lambda _item: object())
    monkeypatch.setattr(driver, "_confirm_payload_profile", lambda _item: None)
    monkeypatch.setattr(driver, "_navigate_to_model_item", lambda _item, **_kw: (10, 10))
    monkeypatch.setattr(driver, "_snap", lambda: b"PRE_HEART")
    monkeypatch.setattr(driver, "_await_sheet_open", lambda **_kw: initial)
    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(hinge, "locate_inline_composer", lambda *_args, **_kw: initial)
    monkeypatch.setattr(
        driver, "_locate_inline_composer",
        lambda frame: resumed if frame == b"FRESH_AFTER_CHOICE" else checkpoint)
    monkeypatch.setattr(driver, "_verify_sheet_shows", lambda frame, *_a, **_kw: seen.append(frame))
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(driver, "_handle_rose_upsell", lambda: False)
    monkeypatch.setattr(driver, "_verify_like_landed", lambda _before: None)
    monkeypatch.setattr(driver, "_verify_training_like_landed", lambda _item: "identity")
    monkeypatch.setattr(hinge, "_match_glyph", lambda frame, template, **_kw: [(70, 70)])

    def decide(frame, _evidence):
        assert frame == b"KEYBOARD_HIDDEN"
        assert ("keyevent", 4) in events
        return "like"

    driver.set_training_decision(decide)
    assert driver.like("A targeted opener", model_item_index=1) == "like"

    assert ("tap", resumed.confirm_point) in events
    assert ("tap", checkpoint.confirm_point) not in events
    assert seen[-1] == b"FRESH_AFTER_CHOICE"
    assert events.index(("keyevent", 4)) < events.index(("tap", resumed.confirm_point))


def test_training_like_rejects_reflowed_same_profile_after_send(monkeypatch):
    """A Send tap is not a label when the apparent next deck is still this profile."""
    events = []
    driver, _adb = _driver(events)
    driver.set_auto_session_policy(object())
    initial = SimpleNamespace(comment_rect=SimpleNamespace(center=(20, 20)),
                              confirm_point=(30, 30))
    checkpoint = SimpleNamespace(comment_rect=SimpleNamespace(center=(21, 21)),
                                 confirm_point=(40, 40))
    resumed = SimpleNamespace(comment_rect=SimpleNamespace(center=(22, 22)),
                              confirm_point=(50, 50))
    frames = iter((b"SHEET", b"FOCUSED", b"KEYBOARD_HIDDEN", b"FRESH_AFTER_CHOICE",
                   b"REFLOW_A", b"REFLOW_B", b"REFLOW_C"))

    monkeypatch.setattr(driver, "_verifiable_payload", lambda _item: object())
    monkeypatch.setattr(driver, "_confirm_payload_profile", lambda _item: None)
    monkeypatch.setattr(driver, "_navigate_to_model_item", lambda _item, **_kw: (10, 10))
    monkeypatch.setattr(driver, "_snap", lambda: b"PRE_HEART")
    monkeypatch.setattr(driver, "_await_sheet_open", lambda **_kw: initial)
    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(hinge, "locate_inline_composer", lambda *_args, **_kw: initial)
    monkeypatch.setattr(
        driver, "_locate_inline_composer",
        lambda frame: None if frame.startswith(b"REFLOW") else (
            resumed if frame == b"FRESH_AFTER_CHOICE" else checkpoint))
    monkeypatch.setattr(driver, "_verify_sheet_shows", lambda *_a, **_kw: None)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(driver, "_handle_rose_upsell", lambda: False)
    # Isolate the existing generic post-Send checks. The regression is the stronger Training
    # proof that runs immediately afterwards and must refuse a reflowed current card.
    monkeypatch.setattr(driver, "_verify_like_landed", lambda _before: None)
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(driver, "_is_current_profile_frame", lambda *_a, **_kw: True)
    monkeypatch.setattr(hinge, "_match_glyph", lambda *_a, **_kw: [(70, 70)])
    driver.set_training_decision(lambda _frame, _evidence: "like")

    with pytest.raises(HingeActionError, match="semantically different ready deck"):
        driver.like("A targeted opener", model_item_index=1)

    assert ("tap", resumed.confirm_point) in events
    assert driver.landed_auto_opener_evidence() is None


def test_training_freshly_locates_pass_from_keyboard_hidden_composer(monkeypatch):
    """Dislike is a verified pass of the composer profile, not a stale deck tap."""
    events = []
    driver, _adb = _driver(events)
    driver.set_auto_session_policy(object())
    initial = SimpleNamespace(comment_rect=SimpleNamespace(center=(20, 20)),
                              confirm_point=(30, 30))
    composer = SimpleNamespace(comment_rect=SimpleNamespace(center=(21, 21)),
                               confirm_point=(40, 40))
    frames = iter((b"SHEET", b"FOCUSED", b"KEYBOARD_HIDDEN", b"FRESH_FOR_PASS"))
    verified = []

    monkeypatch.setattr(driver, "_verifiable_payload", lambda _item: object())
    monkeypatch.setattr(driver, "_confirm_payload_profile", lambda _item: None)
    monkeypatch.setattr(driver, "_navigate_to_model_item", lambda _item, **_kw: (10, 10))
    monkeypatch.setattr(driver, "_snap", lambda: b"PRE_HEART")
    monkeypatch.setattr(driver, "_await_sheet_open", lambda **_kw: initial)
    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(hinge, "locate_inline_composer", lambda *_args, **_kw: initial)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: composer)
    monkeypatch.setattr(driver, "_verify_sheet_shows", lambda frame, *_a, **_kw: verified.append(frame))
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        driver, "_verify_training_dislike_landed",
        lambda item, **_kw: events.append(("training_advance", item)) or "identity")
    monkeypatch.setattr(hinge, "_match_glyph", lambda frame, template, **_kw: [(80, 80)])
    driver.set_training_decision(lambda _frame, _evidence: "dislike")

    assert driver.like("A targeted opener", model_item_index=1) == "dislike"

    assert ("tap", (80, 80)) in events
    assert ("tap", composer.confirm_point) not in events
    assert verified[-1] == b"FRESH_FOR_PASS"
    assert ("training_advance", 1) in events


def test_training_dislike_rejects_reflowed_same_profile_without_deck_advance(monkeypatch):
    """A visually different composer close on the same profile is not a training pass."""
    events = []
    driver, _adb = _driver(events)
    driver.set_auto_session_policy(object())
    driver.halt_on_error = True
    initial = SimpleNamespace(comment_rect=SimpleNamespace(center=(20, 20)),
                              confirm_point=(30, 30))
    composer = SimpleNamespace(comment_rect=SimpleNamespace(center=(21, 21)),
                               confirm_point=(40, 40))
    # The post-X frames differ from each other and from the pre-heart frame, as a real inline
    # composer reflow can. A generic pixel delta would therefore accept this false pass. The
    # semantic current-profile helper must reject every frame instead.
    frames = iter((b"SHEET", b"FOCUSED", b"KEYBOARD_HIDDEN", b"FRESH_FOR_PASS",
                   b"REFLOW_A", b"REFLOW_B", b"REFLOW_C"))

    monkeypatch.setattr(driver, "_verifiable_payload", lambda _item: object())
    monkeypatch.setattr(driver, "_confirm_payload_profile", lambda _item: None)
    monkeypatch.setattr(driver, "_navigate_to_model_item", lambda _item, **_kw: (10, 10))
    monkeypatch.setattr(driver, "_snap", lambda: b"PRE_HEART")
    monkeypatch.setattr(driver, "_await_sheet_open", lambda **_kw: initial)
    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(hinge, "locate_inline_composer", lambda *_args, **_kw: initial)
    monkeypatch.setattr(
        driver, "_locate_inline_composer",
        lambda frame: None if frame.startswith(b"REFLOW") else composer)
    monkeypatch.setattr(driver, "_verify_sheet_shows", lambda *_a, **_kw: None)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(hinge, "_match_glyph", lambda *_a, **_kw: [(80, 80)])
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(driver, "_is_current_profile_frame", lambda *_a, **_kw: True)
    monkeypatch.setattr(driver, "_dbg_action", lambda name, *_a, **_kw: events.append(("debug", name)))
    driver.set_training_decision(lambda _frame, _evidence: "dislike")

    with pytest.raises(HingeActionError, match="semantically different ready deck"):
        driver.like("A targeted opener", model_item_index=1)

    assert ("tap", (80, 80)) in events
    assert ("debug", "training_dislike") not in events


@pytest.mark.parametrize("verifier_name", [
    "_verify_training_dislike_landed", "_verify_training_like_landed",
])
def test_training_action_accepts_only_two_stable_semantically_new_deck_frames(
        verifier_name, monkeypatch):
    """Both Training actions require a settled independently-proven next card."""
    events = []
    driver, _adb = _driver(events)
    frames = iter((b"NEXT_DECK_FIRST", b"NEXT_DECK_SECOND"))

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(
        driver, "_training_profile_advance_proof",
        lambda _frame, _item: ("identity", None))
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)

    assert getattr(driver, verifier_name)(1) == "identity"


def test_durable_presend_evidence_does_not_depend_on_local_debug_logging():
    driver, _adb = _driver([])
    driver.set_auto_session_policy(None)

    evidence = driver._record_auto_opener_pre_send(
        b"TYPED_OPENER_ON_SELECTED_ITEM", opener="text", item_index=None,
        model_item_index=1)

    assert evidence is not None
    assert evidence["frame"] == b"TYPED_OPENER_ON_SELECTED_ITEM"
    assert evidence["model_item_index"] == 1
