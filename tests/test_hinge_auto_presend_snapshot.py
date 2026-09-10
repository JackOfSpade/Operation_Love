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
        return frame, composer, (80, 80)
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


@pytest.mark.parametrize(("decision", "set_stop", "reason"), [
    ("stop", True, "stop_requested"),
    (None, False, "invalid_decision"),
])
def test_training_cancellation_records_checkpoint_local_unsent_outcome(
        tmp_path, monkeypatch, decision, set_stop, reason):
    """Stop and an invalid review result both end the typed checkpoint before Send Like."""
    events = []
    driver, _adb = _driver(events)
    driver.set_auto_session_policy(object())
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-stop", keep_shots=50)
    initial = SimpleNamespace(comment_rect=SimpleNamespace(center=(20, 20)),
                              confirm_point=(30, 30))
    checkpoint = SimpleNamespace(comment_rect=SimpleNamespace(center=(21, 21)),
                                 confirm_point=(40, 40))
    stopped = {"value": False}

    monkeypatch.setattr(driver, "_verifiable_payload", lambda _item: object())
    monkeypatch.setattr(driver, "_confirm_payload_profile", lambda _item: None)
    monkeypatch.setattr(driver, "_navigate_to_model_item", lambda _item, **_kw: (10, 10))
    monkeypatch.setattr(driver, "_snap", lambda: b"PRE_HEART")
    monkeypatch.setattr(driver, "_await_sheet_open", lambda **_kw: initial)
    monkeypatch.setattr(driver, "_screencap", lambda: b"FOCUSED")
    monkeypatch.setattr(hinge, "locate_inline_composer", lambda *_a, **_kw: initial)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: checkpoint)
    monkeypatch.setattr(driver, "_verify_sheet_shows", lambda *_a, **_kw: None)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(driver, "_hide_keyboard_for_training", lambda **_kw: b"CHECKPOINT")
    monkeypatch.setattr(
        driver, "_verify_training_checkpoint",
        lambda frame, *_a, **_kw: (frame, checkpoint, (80, 80)))

    def decide(_frame, _evidence):
        stopped["value"] = set_stop
        return decision

    driver.set_training_decision(decide)
    with pytest.raises(hinge.ActionCancelled):
        driver.like("A targeted opener", model_item_index=1,
                    should_stop=lambda: stopped["value"])

    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    pre_send = next(record for record in records if record["action"] == "auto_opener_pre_send")
    cancelled = next(record for record in records if record["action"] == "training_cancelled")
    assert cancelled["pre_send_evidence_id"] == pre_send["evidence_id"]
    assert cancelled["model_item_index"] == 1
    assert cancelled["reason"] == reason
    assert not any(record["action"] in {"auto_opener_resumed_send", "like_attempt", "like"}
                   for record in records)
    assert ("tap", checkpoint.confirm_point) not in events


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


def test_training_freshly_locates_pass_from_keyboard_hidden_composer(tmp_path, monkeypatch):
    """Dislike is a verified pass of the composer profile, not a stale deck tap."""
    events = []
    driver, _adb = _driver(events)
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-dislike-evidence")
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
        lambda item, **kw: events.append(("training_advance", item, kw)) or "identity")
    monkeypatch.setattr(hinge, "_match_glyph", lambda frame, template, **_kw: [(80, 80)])
    driver.set_training_decision(lambda _frame, _evidence: "dislike")

    assert driver.like("A targeted opener", model_item_index=1) == "dislike"

    assert ("tap", (80, 80)) in events
    assert ("tap", composer.confirm_point) not in events
    assert verified[-1] == b"FRESH_FOR_PASS"
    assert ("training_advance", 1, {"allow_same_name_successor": True}) in events
    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    approval = next(record for record in records if record["action"] == "auto_opener_pre_send")
    dislike = next(record for record in records if record["action"] == "training_dislike")
    assert dislike["pre_send_evidence_id"] == approval["evidence_id"]
    assert not any(record["action"] in {"like_attempt", "like"} for record in records)


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


def test_training_dislike_records_linked_unverified_outcome_after_tap(tmp_path, monkeypatch):
    """A failed post-X proof is an auditable physical action, never a training label."""
    events = []
    driver, _adb = _driver(events)
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-dislike-unverified")
    checkpoint = b"VERIFIED_COMPOSER_CHECKPOINT"
    evidence_id = "human-reviewed-draft-evidence"

    monkeypatch.setattr(driver, "_screencap", lambda: checkpoint)
    monkeypatch.setattr(
        driver, "_verify_training_checkpoint",
        lambda *_a, **_kw: (checkpoint, object(), (80, 80)))
    monkeypatch.setattr(driver, "_tap", lambda *point: events.append(("tap", point)))
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        driver, "_verify_training_dislike_landed",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            HingeActionError("semantic deck proof was inconclusive")))

    with pytest.raises(HingeActionError, match="semantic deck proof was inconclusive"):
        driver._training_dislike_from_composer(
            object(), 3, b"PRE_HEART", pre_send_evidence={"evidence_id": evidence_id})

    assert events == [("tap", (80, 80))]
    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    unverified = next(
        record for record in records if record["action"] == "training_dislike_unverified")
    assert unverified["model_item_index"] == 3
    assert unverified["pre_send_evidence_id"] == evidence_id
    assert unverified["x"] == [80, 80]
    assert unverified["error"] == "HingeActionError: semantic deck proof was inconclusive"
    assert unverified["kept_before"] == unverified["before"]
    assert (driver._dbg.dir / unverified["before"]).read_bytes() == checkpoint
    assert not any(record["action"] == "training_dislike" for record in records)


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
        lambda _frame, _item, **_kw: ("identity", None))
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)

    assert getattr(driver, verifier_name)(1) == "identity"


def test_training_like_accepts_two_stable_confirmed_top_new_name_frames(tmp_path, monkeypatch):
    """Training records a real Like once two ready frames repeat the new card's name."""
    import numpy as np

    events = []
    driver, _adb = _driver(events)
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-advance", keep_shots=10)
    driver._identity_name = "Mackinley MJ"
    driver._identity_sig = np.zeros((16, 64), dtype="int16")
    driver._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    frames = iter((b"KATE_TOP_FIRST", b"KATE_TOP_SECOND"))

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        hinge, "_band", lambda frame, rect: np.full((16, 64), 99, dtype="int16"))
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda frame, **_kw: SimpleNamespace(
            state="confirmed_top", confirmed=True, distance=0.0,
            reason="canonical scroll top confirmed"))
    monkeypatch.setattr(
        driver, "_ocr_band",
        lambda frame, rect, psm="7": "Kate\nshe her" if psm == "6" else None)

    assert driver._verify_training_like_landed(2) == "name"
    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    probe = next(record for record in records if record["action"] == "training_advance_probe")
    assert probe["outcome"] == "accepted"
    assert probe["first"]["name_candidate"] == "Kate"
    assert probe["second"]["proof"] == "name"
    assert probe["stable"] is True
    assert probe["names_agree"] is True
    serialized = json.dumps(probe)
    assert "name_read" not in serialized
    assert "Kate\\nshe her" not in serialized


def test_training_like_accepts_repeated_structured_single_letter_name(tmp_path, monkeypatch):
    """Regression: a completed Aisha -> S Send Like must not halt after the deck advanced.

    ``S`` alone remains OCR noise; Hinge's exact ``S shows thoughtful signals`` banner binds it
    to the profile-name slot, after which the ordinary canonical-top, repeat, source-agreement,
    and stable-frame gates still license the label.
    """
    import numpy as np

    driver, _adb = _driver([])
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-single-letter-advance")
    driver._identity_name = "Aisha"
    driver._identity_sig = np.zeros((16, 64), dtype="int16")
    driver._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    frames = iter((b"S_TOP_FIRST", b"S_TOP_SECOND"))

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        hinge, "_band", lambda frame, rect: np.full((16, 64), 99, dtype="int16"))
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda frame, **_kw: SimpleNamespace(
            state="confirmed_top", confirmed=True, distance=0.0,
            reason="canonical scroll top confirmed"))
    monkeypatch.setattr(
        driver, "_ocr_band",
        lambda frame, rect, psm="7", **_kw: (
            "S shows thoughtful signals" if psm == "6" else None))

    assert driver._verify_training_like_landed(2) == "name"
    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    probe = next(record for record in records if record["action"] == "training_advance_probe")
    assert probe["outcome"] == "accepted"
    assert probe["first"]["name_candidate"] == "S"
    assert probe["second"]["name_candidate"] == "S"
    assert probe["names_agree"] is True


def test_training_dislike_accepts_repeated_structured_two_letter_name(tmp_path, monkeypatch):
    """Regression for run 24179f2e77e0: ``Ri`` is a real Hinge name, not OCR noise.

    The short candidate remains usable only when Hinge's exact Signals banner binds it to the
    name slot, and only after the ordinary canonical-top, repeated-source, ready-deck, and
    stable-frame gates all pass.
    """
    import numpy as np

    driver, _adb = _driver([])
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-two-letter-advance")
    driver._identity_name = "Francesca"
    driver._identity_sig = np.zeros((16, 64), dtype="int16")
    driver._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    frames = iter((b"RI_TOP_FIRST", b"RI_TOP_SECOND"))

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        hinge, "_band", lambda _frame, _rect: np.full((16, 64), 99, dtype="int16"))
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda *_a, **_kw: SimpleNamespace(
            state="confirmed_top", confirmed=True, distance=0.0,
            reason="canonical scroll top confirmed"))
    monkeypatch.setattr(
        driver, "_ocr_band",
        lambda _frame, _rect, psm="7", **_kw: (
            "Ri shows thoughtful signals" if psm == "6" else None))

    assert driver._verify_training_dislike_landed(3) == "name"

    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    probe = next(record for record in records if record["action"] == "training_advance_probe")
    assert probe["outcome"] == "accepted"
    assert probe["first"]["name_candidate"] == "Ri"
    assert probe["second"]["name_candidate"] == "Ri"
    assert probe["stable"] is True
    assert probe["names_agree"] is True


def test_training_dislike_accepts_repeated_name_below_take_another_look_panel(
        tmp_path, monkeypatch):
    """Holly -> Lauren: the panel is structural evidence for a shifted OCR crop only.

    The semantic Training gate remains unchanged: Hinge must be at canonical top with a closed,
    ready deck, and Lauren must be read twice from the same lower calibrated crop on stable
    frames.  Content mismatch alone still has no authority to create this label.
    """
    import numpy as np

    driver, _adb = _driver([])
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-take-another-look")
    driver._identity_name = "Holly"
    driver._identity_sig = np.zeros((16, 64), dtype="int16")
    driver._identity_top_sig = np.full((16, 64), 99, dtype="int16")
    frames = iter((b"LAUREN_PANEL_FIRST", b"LAUREN_PANEL_SECOND"))

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        hinge, "_band", lambda _frame, _rect: driver._identity_top_sig)
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda *_a, **_kw: SimpleNamespace(
            state="confirmed_top", confirmed=True, distance=0.0,
            reason="canonical scroll top confirmed"))

    def ocr(_frame, rect, psm="7", **_kwargs):
        if psm != "6":
            return None
        if tuple(rect) == driver.identity_top_name_band:
            return "Take another look"
        if tuple(rect) == driver.identity_top_name_take_another_look_band:
            return "Lauren"
        return None

    monkeypatch.setattr(driver, "_ocr_band", ocr)

    assert driver._verify_training_dislike_landed(1) == "name"
    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    probe = next(record for record in records if record["action"] == "training_advance_probe")
    assert probe["outcome"] == "accepted"
    assert probe["first"]["name_candidate"] == "Lauren"
    assert probe["second"]["name_candidate"] == "Lauren"
    assert (probe["first"]["name_source"]
            == "top_card_header_take_another_look")
    assert probe["names_agree"] is True
    assert probe["stable"] is True


def test_training_panel_fuzzy_stored_name_recipe_disagreement_cannot_prove_advance(
        monkeypatch):
    """Two distinct fuzzy reads must not leak the panel source into Training's exact-name path.

    `current_profile=False` plus a content mismatch is deliberately the strongest adversarial
    caller shape: before the paired-candidate rule, ``Holli`` / ``Hollyx`` both fuzzy-matched
    stored Holly, published the lower OCR source, and could be promoted by the exact-name
    conflict helper.  The disagreement now remains a top/inconclusive frame and returns no
    semantic proof.
    """
    import numpy as np

    driver, _adb = _driver([])
    driver._identity_name = "Holly"
    driver._identity_sig = np.zeros((16, 64), dtype="int16")
    driver._identity_top_sig = np.full((16, 64), 99, dtype="int16")
    monkeypatch.setattr(
        hinge, "_band", lambda _frame, _rect: driver._identity_top_sig)

    def ocr(_frame, rect, psm="7", *, upscale=3, **_kwargs):
        if psm != "6":
            return None
        if tuple(rect) == driver.identity_top_name_band:
            return "Take another look"
        if tuple(rect) == driver.identity_top_name_take_another_look_band:
            return "Holli" if upscale == 3 else "Hollyx"
        return None

    monkeypatch.setattr(driver, "_ocr_band", ocr)

    def content_mismatched_noncurrent(frame, *, require_content=False, diagnostics=None):
        assert require_content is True
        assert driver._identity_of(frame)[0] == "top"
        if diagnostics is not None:
            diagnostics.update(
                current_profile=False, current_profile_branch="content_only",
                current_content_exact_matched=False, current_content_shift_matched=False)
        return False

    monkeypatch.setattr(driver, "_is_current_profile_frame", content_mismatched_noncurrent)

    diagnostics = {}
    assert driver._training_profile_advance_proof(
        b"FUZZY_PANEL_DISAGREEMENT", None, diagnostics=diagnostics) is None
    assert diagnostics["current_profile"] is False
    assert diagnostics["current_content_exact_matched"] is False
    assert diagnostics["name_candidate"] is None
    assert driver._identity_name_candidate_source is None
    assert driver._identity_top_name_read_source == "top_card_header"


def test_training_advance_accepts_repeated_new_name_despite_photo_collision(
        tmp_path, monkeypatch):
    """The Ery -> Roisin incident: a new first photo matched Ery's coarse content signature.

    `_is_current_profile_frame` is intentionally conservative and therefore still answered
    current, but its `_identity_of` call had independently read the clean new scroll-top name.
    That name must reach the existing two-frame/name-repeat/stability proof instead of being
    discarded by the coarse content answer.
    """
    driver, _adb = _driver([])
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-photo-collision")
    frames = iter((b"ROISIN_TOP_FIRST", b"ROISIN_TOP_SECOND"))

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)

    def colliding_current_profile(_frame, *, require_content=False, diagnostics=None):
        assert require_content is True
        driver._identity_top_name_verdict = "new"
        driver._identity_name_candidate = "Roisin"
        driver._identity_top_name_read = "Roisin\nshe her"
        driver._identity_name_candidate_source = "top_card_header"
        return True

    monkeypatch.setattr(driver, "_is_current_profile_frame", colliding_current_profile)
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda *_a, **_kw: SimpleNamespace(
            state="confirmed_top", confirmed=True, distance=0.0,
            reason="canonical scroll top confirmed"))

    assert driver._verify_training_dislike_landed(2) == "name"

    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    probe = next(record for record in records if record["action"] == "training_advance_probe")
    assert probe["outcome"] == "accepted"
    assert probe["first"]["current_profile"] is True
    assert probe["first"]["name_candidate"] == "Roisin"
    assert probe["second"]["name_candidate"] == "Roisin"
    assert probe["names_agree"] is True


def test_training_like_accepts_sara_after_marina_despite_four_row_collision(
        tmp_path, monkeypatch):
    """End-to-end regression for the 2026-09-01 completed-but-unlabelled Like.

    Hinge had advanced from Marina to two stable ready frames of Sara. The clean Sara header
    nevertheless fuzzy-matched Marina at the old 0.60 boundary, while four coincidental rows
    at shift +14 matched one Marina capture signature. Together those false ``same`` signals
    made the verifier reject a Like that the retained loading/next-deck frames proved landed.
    """
    import numpy as np

    driver, _adb = _driver([])
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-marina-sara-advance")
    driver._identity_name = "Marina"
    driver._identity_sig = np.zeros((16, 64), dtype="int16")
    driver._identity_top_sig = np.full((16, 64), 200, dtype="int16")

    rng = np.random.default_rng(17)
    size = 24
    r0, r1 = hinge._content_rows(driver.content_band, size)
    marina_sig = rng.integers(0, 255, size=(size, size)).astype("int16")
    sara_frame_sig = rng.integers(0, 255, size=(size, size)).astype("int16")
    sara_frame_sig[r0 + 14:r1] = marina_sig[r0:r1 - 14]
    driver._current_sigs = [marina_sig]
    frames = iter((b"SARA_TOP_FIRST", b"SARA_TOP_SECOND"))

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(hinge, "_downsample", lambda _frame: sara_frame_sig)
    monkeypatch.setattr(
        hinge, "_band", lambda _frame, _rect: driver._identity_top_sig)
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda *_a, **_kw: SimpleNamespace(
            state="confirmed_top", confirmed=True, distance=0.0,
            reason="canonical scroll top confirmed"))
    monkeypatch.setattr(
        driver, "_ocr_band",
        lambda _frame, _rect, psm="7", **_kw: "Sara" if psm == "6" else None)

    assert driver._verify_training_like_landed(2) == "name"

    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    probe = next(record for record in records if record["action"] == "training_advance_probe")
    assert probe["outcome"] == "accepted"
    assert probe["first"]["name_candidate"] == "Sara"
    assert probe["second"]["name_candidate"] == "Sara"
    assert probe["names_agree"] is True


def test_training_like_accepts_sofia_after_sophia(tmp_path, monkeypatch):
    """End-to-end regression for the reported completed Sophia -> Sofia Like.

    The post-send loading screen settled on a visibly different ready card, but the clean Sofia
    header scored 0.727 against stored Sophia and the fuzzy-name guard called it the same person.
    Two stable, canonically-top Sofia reads must license the landed training label while the
    measured 0.80 Zorva OCR near-misses remain protected by the lower-level calibration test.
    """
    import numpy as np

    driver, _adb = _driver([])
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-sophia-sofia-advance")
    driver._identity_name = "Sophia"
    driver._identity_sig = np.zeros((16, 64), dtype="int16")
    driver._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    driver._current_sigs = [np.zeros((24, 24), dtype="int16")]
    sofia_sig = np.full((24, 24), 150, dtype="int16")
    frames = iter((b"SOFIA_TOP_FIRST", b"SOFIA_TOP_SECOND"))

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(hinge, "_downsample", lambda _frame: sofia_sig)
    monkeypatch.setattr(
        hinge, "_band", lambda _frame, _rect: driver._identity_top_sig)
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda *_a, **_kw: SimpleNamespace(
            state="confirmed_top", confirmed=True, distance=0.0,
            reason="canonical scroll top confirmed"))
    monkeypatch.setattr(
        driver, "_ocr_band",
        lambda _frame, _rect, psm="7", **_kw: "Sofia" if psm == "6" else None)

    assert driver._verify_training_like_landed(3) == "name"

    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    probe = next(record for record in records if record["action"] == "training_advance_probe")
    assert probe["outcome"] == "accepted"
    assert probe["first"]["composer_open"] is False
    assert probe["first"]["deck_ready"] is True
    assert probe["first"]["name_candidate"] == "Sofia"
    assert probe["second"]["name_candidate"] == "Sofia"
    assert probe["stable"] is True
    assert probe["names_agree"] is True


def test_training_exact_name_conflict_still_rejects_matching_profile_content(monkeypatch):
    """A likely OCR spelling error cannot become a label when the captured card still matches."""
    import numpy as np

    driver, _adb = _driver([])
    driver._identity_name = "Sophia"
    driver._identity_sig = np.zeros((16, 64), dtype="int16")
    driver._identity_top_sig = np.full((16, 64), 200, dtype="int16")
    captured_sig = np.zeros((24, 24), dtype="int16")
    driver._current_sigs = [captured_sig]

    monkeypatch.setattr(hinge, "_downsample", lambda _frame: captured_sig.copy())
    monkeypatch.setattr(
        hinge, "_band", lambda _frame, _rect: driver._identity_top_sig)
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda *_a, **_kw: SimpleNamespace(
            state="confirmed_top", confirmed=True, distance=0.0,
            reason="canonical scroll top confirmed"))
    monkeypatch.setattr(
        driver, "_ocr_band",
        lambda _frame, _rect, psm="7", **_kw: "Sofia" if psm == "6" else None)

    diagnostics = {}
    assert driver._training_profile_advance_proof(
        b"SAME_PROFILE_WITH_OCR_VARIANT", 3, diagnostics=diagnostics) is None
    assert diagnostics["current_profile"] is True
    assert diagnostics["current_content_exact_matched"] is True
    assert diagnostics["name_verdict"] == "same"


def test_training_dislike_accepts_two_stable_content_disjoint_same_name_successors(
        tmp_path, monkeypatch):
    """Lauren -> Lauren is labelable only on the opt-in post-X Dislike verifier.

    The profile's first-name strip collides, but both captured candidates are content-disjoint,
    use the normal exact card-header name read, and are canonically at top.  The outer verifier
    must still observe the same proof twice on stable, ready deck frames.
    """
    driver, _adb = _driver([])
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-same-name-successor")
    driver._identity_name = "Lauren"
    frames = iter((b"LAUREN_SUCCESSOR_FIRST", b"LAUREN_SUCCESSOR_SECOND"))

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda *_a, **_kw: SimpleNamespace(
            state="confirmed_top", confirmed=True, distance=0.0,
            reason="canonical scroll top confirmed"))

    def different_same_name(_frame, *, require_content=False, diagnostics=None):
        assert require_content is True
        driver._identity_top_name_verdict = "same"
        driver._identity_top_name_read = "Lauren"
        driver._identity_top_name_read_source = "top_card_header"
        if diagnostics is not None:
            diagnostics.update(
                current_profile=False,
                current_identity_state="same",
                current_profile_branch="identity_same_with_content",
                current_content_exact_matched=False,
                current_content_shift_matched=False,
            )
        return False

    monkeypatch.setattr(driver, "_is_current_profile_frame", different_same_name)

    assert driver._verify_training_dislike_landed(
        3, allow_same_name_successor=True) == "same_name_successor"
    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    probe = next(record for record in records if record["action"] == "training_advance_probe")
    assert probe["outcome"] == "accepted"
    assert probe["same_name_successor_enabled"] is True
    assert probe["first"]["proof"] == "same_name_successor"
    assert probe["first"]["same_name_successor_content_disjoint"] is True
    assert probe["first"]["same_name_successor_same_name_verdict"] is True
    assert probe["first"]["same_name_successor_exact_stored_name"] is True
    assert probe["first"]["same_name_successor_source"] == "top_card_header"
    assert probe["first"]["same_name_successor_top_confirmed"] is True
    assert probe["names_agree"] is True
    assert probe["stable"] is True


@pytest.mark.parametrize("exact, shifted, source, text, top_confirmed, verdict, identity_state", [
    (True, False, "top_card_header", "Lauren", True, "same", "same"),
    (False, True, "top_card_header", "Lauren", True, "same", "same"),
    (False, False, "identity_band", "Lauren", True, "same", "same"),
    (False, False, "top_card_header_take_another_look", "Lauren", True, "same", "same"),
    (False, False, "top_card_header", None, True, "same", "same"),
    (False, False, "top_card_header", "Lauren", False, "same", "same"),
    (False, False, "top_card_header", "Lauren", True, "new", "same"),
    (False, False, "top_card_header", "Lauren", True, "same", None),
    (False, False, "top_card_header", "Lauren", True, "same", "new"),
])
def test_training_same_name_successor_rejects_any_missing_conjunction(
        exact, shifted, source, text, top_confirmed, verdict, identity_state, monkeypatch):
    """Content overlap, unsafe OCR, unreadable OCR, or non-top geometry cannot label it."""
    driver, _adb = _driver([])
    driver._identity_name = "Lauren"
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda *_a, **_kw: SimpleNamespace(
            state="confirmed_top" if top_confirmed else "not_top",
            confirmed=top_confirmed, distance=0.0,
            reason="test scroll-top result"))

    diagnostics = {
        "current_profile_branch": "identity_same_with_content",
        "current_content_exact_matched": exact,
        "current_content_shift_matched": shifted,
    }
    if identity_state is not None:
        diagnostics["current_identity_state"] = identity_state
    driver._identity_top_name_verdict = verdict
    driver._identity_top_name_read = text
    driver._identity_top_name_read_source = source

    assert driver._training_same_name_content_disjoint_successor_proof(
        b"SAME_NAME_CANDIDATE", current_profile=False, diagnostics=diagnostics) is None
    assert diagnostics["same_name_successor_considered"] is True
    assert (diagnostics["same_name_successor_content_disjoint"]
            is (not exact and not shifted))
    if not top_confirmed and not exact and not shifted and text is not None and source in {
            "top_card_header", "top_card_header_fallback"}:
        assert diagnostics["same_name_successor_top_confirmed"] is False
    if verdict != "same":
        assert diagnostics["same_name_successor_same_name_verdict"] is False
    if identity_state != "same":
        assert diagnostics["same_name_successor_identity_state_same"] is False


def test_training_like_cannot_use_same_name_content_disjoint_successor(monkeypatch):
    """The same-name exception is unavailable to Training Like and all shared callers."""
    driver, _adb = _driver([])
    driver._identity_name = "Lauren"
    frames = iter((b"LIKE_SAME_NAME_0", b"LIKE_SAME_NAME_1", b"LIKE_SAME_NAME_2"))

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)

    def different_same_name(_frame, *, require_content=False, diagnostics=None):
        assert require_content is True
        driver._identity_top_name_verdict = "same"
        driver._identity_top_name_read = "Lauren"
        driver._identity_top_name_read_source = "top_card_header"
        if diagnostics is not None:
            diagnostics.update(
                current_profile=False,
                current_profile_branch="identity_same_with_content",
                current_content_exact_matched=False,
                current_content_shift_matched=False,
            )
        return False

    monkeypatch.setattr(driver, "_is_current_profile_frame", different_same_name)

    with pytest.raises(HingeActionError, match="semantically different ready deck"):
        driver._verify_training_like_landed(3)


def test_training_same_name_successor_requires_stable_pair(monkeypatch):
    """Two individually valid successor observations still fail while the deck is moving."""
    driver, _adb = _driver([])
    driver._identity_name = "Lauren"
    frames = iter(f"MOVING_LAUREN_{index}".encode() for index in range(6))

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: True)

    def proof(_frame, _item, *, diagnostics, **_kwargs):
        diagnostics["same_name_successor_source"] = "top_card_header"
        return "same_name_successor", "lauren"

    monkeypatch.setattr(driver, "_training_dislike_surface_proof", proof)

    with pytest.raises(HingeActionError, match="semantically different ready deck"):
        driver._verify_training_dislike_landed(3, allow_same_name_successor=True)


def test_training_same_name_successor_rejects_mixed_header_sources(monkeypatch):
    """A repeated same-name proof must come from the same calibrated header crop."""
    driver, _adb = _driver([])
    frames = iter(f"MIXED_LAUREN_{index}".encode() for index in range(6))
    sources = iter(("top_card_header", "top_card_header_fallback") * 3)

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)

    def proof(_frame, _item, *, diagnostics, **_kwargs):
        diagnostics["same_name_successor_source"] = next(sources)
        return "same_name_successor", "lauren"

    monkeypatch.setattr(driver, "_training_dislike_surface_proof", proof)

    with pytest.raises(HingeActionError, match="semantically different ready deck"):
        driver._verify_training_dislike_landed(3, allow_same_name_successor=True)


def test_training_same_name_successor_real_current_matcher_accepts_stable_lauren(
        tmp_path, monkeypatch):
    """Two real same-name/content-disjoint observations prove the next Lauren card."""
    import numpy as np

    driver, _adb = _driver([])
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="same-name-successor", keep_shots=8)
    driver._identity_name = "Lauren"
    driver._identity_sig = np.zeros((16, 64), dtype=np.int16)
    driver._identity_top_sig = np.full((16, 64), 99, dtype=np.int16)
    driver._current_sigs = [np.zeros((24, 24), dtype=np.int16)]
    frames = iter((b"LAUREN_SUCCESSOR_ONE", b"LAUREN_SUCCESSOR_TWO"))

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_deck_blocked_reason", lambda _frame: None)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: None)
    monkeypatch.setattr(driver, "_observe_deck_ready", lambda _frame: True)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)
    monkeypatch.setattr(hinge, "_downsample", lambda _frame: np.full(
        (24, 24), 200, dtype=np.int16))
    monkeypatch.setattr(
        hinge, "_band", lambda _frame, _rect: driver._identity_top_sig.copy())
    monkeypatch.setattr(
        hinge, "confirm_scroll_top", lambda *_a, **_kw: SimpleNamespace(
            confirmed=True, state="top", distance=0.0, reason="synthetic"))

    def ocr(_frame, _rect, psm="7", **_kwargs):
        return "Lauren" if psm == "6" else None

    monkeypatch.setattr(driver, "_ocr_band", ocr)

    assert driver._verify_training_dislike_landed(
        3, allow_same_name_successor=True) == "same_name_successor"
    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    probe = next(record for record in records if record["action"] == "training_advance_probe")
    assert probe["same_name_successor_enabled"] is True
    assert probe["first"]["current_profile_branch"] == "identity_same_with_content"
    assert probe["first"]["current_identity_state"] == "same"
    assert probe["first"]["current_content_exact_matched"] is False
    assert probe["first"]["current_content_shift_matched"] is False
    assert probe["first"]["same_name_successor_source"] == "top_card_header"
    assert probe["first"]["same_name_successor_top_confirmed"] is True
    assert probe["second"]["same_name_successor_source"] == "top_card_header"


def test_training_advance_rejects_non_top_name_candidate_on_current_profile(monkeypatch):
    """A sticky-header OCR hallucination cannot override positive same-profile content proof."""
    driver, _adb = _driver([])

    def current_profile_with_tight_candidate(_frame, *, require_content=False, diagnostics=None):
        assert require_content is True
        driver._identity_top_name_verdict = "new"
        driver._identity_name_candidate = "NotTheProfile"
        driver._identity_top_name_read = "NotTheProfile"
        driver._identity_name_candidate_source = "identity_band"
        return True

    monkeypatch.setattr(
        driver, "_is_current_profile_frame", current_profile_with_tight_candidate)

    diagnostics = {}
    assert driver._training_profile_advance_proof(
        b"SAME_PROFILE_REFLOW", 2, diagnostics=diagnostics) is None
    assert diagnostics["current_profile"] is True
    assert diagnostics["name_source"] == "identity_band"


def test_training_advance_rejects_local_top_name_when_canonical_top_is_unavailable(
        monkeypatch):
    """A pinned chips row can look like local top mid-profile; it cannot license a label."""
    driver, _adb = _driver([])

    def current_profile_with_local_top_candidate(_frame, *, require_content=False, diagnostics=None):
        assert require_content is True
        driver._identity_top_name_verdict = "new"
        driver._identity_name_candidate = "FalseHeaderRead"
        driver._identity_top_name_read = "FalseHeaderRead"
        driver._identity_name_candidate_source = "top_card_header"
        return True

    monkeypatch.setattr(
        driver, "_is_current_profile_frame", current_profile_with_local_top_candidate)
    monkeypatch.setattr(
        hinge, "confirm_scroll_top",
        lambda *_a, **_kw: SimpleNamespace(
            state="check_unavailable", confirmed=False, distance=0.0,
            reason="the chips row is pinned while the page scrolls"))

    diagnostics = {}
    assert driver._training_profile_advance_proof(
        b"SAME_PROFILE_PINNED_CHIPS", 2, diagnostics=diagnostics) is None
    assert diagnostics["current_profile"] is True
    assert diagnostics["name_source"] == "top_card_header"
    assert diagnostics["name_top_state"] == "check_unavailable"
    assert diagnostics["name_top_confirmed"] is False


@pytest.mark.parametrize("proofs", [
    (("name", "roisin"), ("identity", None)),
    (("identity", None), ("name", "roisin")),
])
def test_training_advance_rejects_mixed_single_name_and_identity_proofs(
        proofs, monkeypatch):
    """A single name read is never logged as an agreeing repeated-name observation."""
    driver, _adb = _driver([])
    frames = iter(f"FRAME_{index}".encode() for index in range(6))
    proof_stream = iter(proofs * 3)

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_training_dislike_surface_proof",
                        lambda *_a, **_kw: next(proof_stream))
    monkeypatch.setattr(driver, "_changed", lambda _before, _after: False)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)

    with pytest.raises(HingeActionError, match="semantically different ready deck"):
        driver._verify_training_deck_advanced(2)


def test_training_advance_rejects_primary_fallback_hybrid_and_retains_final_pair(
        tmp_path, monkeypatch):
    """Two equally spelled names from different OCR geometries are not one repeated reading.
    The last rejected pair, but not every retry, is retained for incident replay."""
    driver, _adb = _driver([])
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-final-rejection")
    frames = iter(f"REJECT_{index}".encode() for index in range(6))
    sources = iter(("top_card_header", "top_card_header_fallback") * 3)

    monkeypatch.setattr(driver, "_screencap", lambda: next(frames))
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(driver, "_changed", lambda *_a, **_kw: False)

    def proof(_frame, _item, *, diagnostics):
        diagnostics.update(name_source=next(sources), name_candidate="Lara")
        return "name", "lara"

    monkeypatch.setattr(driver, "_training_dislike_surface_proof", proof)

    with pytest.raises(HingeActionError, match="semantically different ready deck"):
        driver._verify_training_deck_advanced(2)

    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    retries = [row for row in records if row["action"] == "training_advance_probe"]
    assert len(retries) == 3
    assert retries[-1]["names_agree"] is False
    assert "kept_before" in retries[-1] and "kept_after" in retries[-1]
    assert (driver._dbg.dir / retries[-1]["kept_before"]).exists()
    assert (driver._dbg.dir / retries[-1]["kept_after"]).exists()
    assert all("kept_before" not in row for row in retries[:-1])


def test_durable_presend_evidence_does_not_depend_on_local_debug_logging():
    driver, _adb = _driver([])
    driver.set_auto_session_policy(None)

    evidence = driver._record_auto_opener_pre_send(
        b"TYPED_OPENER_ON_SELECTED_ITEM", opener="text", item_index=None,
        model_item_index=1)

    assert evidence is not None
    assert evidence["frame"] == b"TYPED_OPENER_ON_SELECTED_ITEM"
    assert evidence["model_item_index"] == 1


def _checkpoint_driver(events, monkeypatch, *, composer_for, glyph_for, hides):
    """Driver with the training checkpoint's collaborators stubbed at their boundaries."""
    driver, _adb = _driver(events)
    monkeypatch.setattr(driver, "_locate_inline_composer", composer_for)
    monkeypatch.setattr(
        driver, "_hide_keyboard_for_training",
        lambda **_kw: (hides.append(True), b"KEYBOARD_DISMISSED_AGAIN")[1])
    monkeypatch.setattr(hinge, "_match_glyph", glyph_for)
    return driver


def test_training_checkpoint_recovers_a_reviewer_raised_keyboard(tmp_path, monkeypatch):
    """A reviewer re-focusing Hinge's comment field must not cost the run a read profile.

    The floating pass X is covered on the frame handed to the checkpoint and becomes visible only
    after the keyboard is dismissed again.  The checkpoint must re-dismiss exactly once and then
    hand back the RECOVERED frame, so the retained evidence and the coordinates that get tapped
    describe the same screen.
    """
    events: list = []
    hides: list = []
    verified: list = []
    composer = SimpleNamespace(comment_rect=SimpleNamespace(center=(21, 21)),
                               confirm_point=(40, 40))
    driver = _checkpoint_driver(
        events, monkeypatch,
        composer_for=lambda _frame: composer,
        # Only the re-dismissed frame shows Hinge's floating pass control.
        glyph_for=lambda frame, *_a, **_kw: (
            [(80, 80)] if frame == b"KEYBOARD_DISMISSED_AGAIN" else []),
        hides=hides)
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-keyboard-recovery")
    monkeypatch.setattr(driver, "_verify_sheet_shows",
                        lambda frame, *_a, **_kw: verified.append(frame))

    frame, located, pass_point = driver._verify_training_checkpoint(
        b"KEYBOARD_COVERS_PASS", object(), 1, b"BEFORE", keyboard_recovery=True)

    assert frame == b"KEYBOARD_DISMISSED_AGAIN"
    assert located is composer
    assert pass_point == (80, 80)
    assert len(hides) == 1
    # The selected item is re-proved on the frame that will actually be acted on, never carried
    # over from the covered one.
    assert verified == [b"KEYBOARD_COVERS_PASS", b"KEYBOARD_DISMISSED_AGAIN"]
    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    recovery = next(record for record in records
                    if record["action"] == "training_checkpoint_keyboard_recovery")
    assert recovery["model_item_index"] == 1


def test_training_checkpoint_refuses_after_one_keyboard_dismissal(monkeypatch):
    """Recovery is a single attempt: a still-covered pass control is an unknown screen."""
    events: list = []
    hides: list = []
    composer = SimpleNamespace(comment_rect=SimpleNamespace(center=(21, 21)),
                               confirm_point=(40, 40))
    driver = _checkpoint_driver(
        events, monkeypatch,
        composer_for=lambda _frame: composer,
        glyph_for=lambda *_a, **_kw: [],
        hides=hides)
    monkeypatch.setattr(driver, "_verify_sheet_shows", lambda *_a, **_kw: None)

    with pytest.raises(UnlocatedControlError, match="even after re-dismissing"):
        driver._verify_training_checkpoint(
            b"COVERED", object(), 1, b"BEFORE", keyboard_recovery=True)

    assert len(hides) == 1                    # exactly one attempt, never a retry loop
    assert not any(kind == "tap" for kind, _payload in events)


def test_training_checkpoint_does_not_send_back_into_an_unknown_screen(monkeypatch):
    """Back is only safe once THIS frame proved an open composer; otherwise just refuse.

    Without the composer proof a stray Back could dismiss a modal or navigate Hinge, so the
    missing-composer refusal must happen before any recovery is considered.
    """
    events: list = []
    hides: list = []
    driver = _checkpoint_driver(
        events, monkeypatch,
        composer_for=lambda _frame: None,
        glyph_for=lambda *_a, **_kw: [],
        hides=hides)

    with pytest.raises(UnlocatedControlError, match="no strictly located inline Send"):
        driver._verify_training_checkpoint(b"UNKNOWN_SCREEN", object(), 1, b"BEFORE")

    assert hides == []
    assert not any(kind == "tap" for kind, _payload in events)


def test_training_checkpoint_refuses_when_recovery_closes_the_composer(monkeypatch):
    """If the recovery Back dismissed the sheet rather than the IME, nothing may be sent."""
    events: list = []
    hides: list = []
    composer = SimpleNamespace(comment_rect=SimpleNamespace(center=(21, 21)),
                               confirm_point=(40, 40))
    driver = _checkpoint_driver(
        events, monkeypatch,
        # Back closed the composer: the recovered frame no longer has one.
        composer_for=lambda frame: None if frame == b"KEYBOARD_DISMISSED_AGAIN" else composer,
        glyph_for=lambda *_a, **_kw: [],
        hides=hides)
    monkeypatch.setattr(driver, "_verify_sheet_shows", lambda *_a, **_kw: None)

    with pytest.raises(UnlocatedControlError, match="no strictly located inline Send"):
        driver._verify_training_checkpoint(
            b"COVERED", object(), 1, b"BEFORE", keyboard_recovery=True)

    assert len(hides) == 1
    assert not any(kind == "tap" for kind, _payload in events)


def _draft_frame(draft_text: str, rect=(95, 1124, 985, 1302)) -> bytes:
    """A frame-sized PNG whose only varying content is the comment field's rendered text."""
    import cv2
    import numpy as np

    image = np.full((2400, 1080), 255, dtype=np.uint8)
    image[300:1090, 95:985] = 128                      # item preview, identical either way
    cv2.putText(image, draft_text, (rect[0] + 8, rect[1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    return cv2.imencode(".png", image)[1].tobytes()


def _draft_composer(rect=(95, 1124, 985, 1302)):
    x0, y0, x1, y1 = rect
    return SimpleNamespace(
        comment_rect=SimpleNamespace(x0=x0, y0=y0, x1=x1, y1=y1,
                                     center=((x0 + x1) // 2, (y0 + y1) // 2)),
        confirm_point=(690, 1390))


def _training_like_run(tmp_path, monkeypatch, *, recovered_draft: str, run_id: str):
    """Drive a Training Like whose resume finds the IME up, then recovers to `recovered_draft`."""
    events: list = []
    driver, _adb = _driver(events)
    driver._dbg = HingeDebugLog(str(tmp_path), run_id=run_id)
    driver.set_auto_session_policy(object())
    approved = _draft_frame("comfort reread or does Neal Stephenson")
    covered = _draft_frame("KEYBOARD UP")
    recovered = _draft_frame(recovered_draft)
    initial = SimpleNamespace(comment_rect=SimpleNamespace(center=(20, 20)),
                              confirm_point=(30, 30))
    composer = _draft_composer()
    frames = iter((b"SHEET", b"FOCUSED", approved, covered, recovered))

    monkeypatch.setattr(driver, "_verifiable_payload", lambda _item: object())
    monkeypatch.setattr(driver, "_confirm_payload_profile", lambda _item: None)
    monkeypatch.setattr(driver, "_navigate_to_model_item", lambda _item, **_kw: (10, 10))
    monkeypatch.setattr(driver, "_snap", lambda: b"PRE_HEART")
    monkeypatch.setattr(driver, "_await_sheet_open", lambda **_kw: initial)
    monkeypatch.setattr(driver, "_screencap", lambda **_kw: next(frames))
    monkeypatch.setattr(hinge, "locate_inline_composer", lambda *_a, **_kw: initial)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: composer)
    monkeypatch.setattr(driver, "_verify_sheet_shows", lambda *_a, **_kw: None)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(driver, "_handle_rose_upsell", lambda: False)
    monkeypatch.setattr(driver, "_verify_like_landed", lambda _before: None)
    monkeypatch.setattr(driver, "_verify_training_like_landed", lambda _item: "identity")
    # The IME hides the pass X on the resumed frame only.
    monkeypatch.setattr(
        hinge, "_match_glyph",
        lambda frame, *_a, **_kw: [] if frame == covered else [(80, 80)])
    driver.set_training_decision(lambda _frame, _evidence: "like")
    return driver, events, composer


def test_training_like_refuses_when_the_draft_changed_during_review(tmp_path, monkeypatch):
    """The 2026-08-28 incident: the reviewer raised the IME AND edited the opener.

    Recovery dismisses the keyboard, but the draft no longer matches what was approved, so the
    send is refused rather than transmitting text nothing can vouch for.
    """
    driver, events, composer = _training_like_run(
        tmp_path, monkeypatch, recovered_draft="comfort reread?", run_id="like-draft-edited")

    with pytest.raises(UnlocatedControlError, match="no longer matches the opener"):
        driver.like("A targeted opener", model_item_index=1)

    assert ("tap", composer.confirm_point) not in events
    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    assert not any(r["action"] in {"like_attempt", "like", "auto_opener_resumed_send"}
                   for r in records)
    # The recovery DID run — the refusal is the draft proof, not a missing pass control.
    assert any(r["action"] == "training_checkpoint_keyboard_recovery" for r in records)


def test_training_like_recovers_when_the_draft_is_provably_unchanged(tmp_path, monkeypatch):
    """The reviewer only TAPPED the field. The draft is byte-identical, so the Like still lands.

    This is the case the blanket refusal used to throw away: a fully-read profile lost because
    someone touched the screen without changing anything.
    """
    driver, events, composer = _training_like_run(
        tmp_path, monkeypatch, recovered_draft="comfort reread or does Neal Stephenson",
        run_id="like-draft-intact")

    assert driver.like("A targeted opener", model_item_index=1) == "like"

    assert ("tap", composer.confirm_point) in events
    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    assert any(r["action"] == "training_checkpoint_keyboard_recovery" for r in records)
    assert any(r["action"] == "auto_opener_resumed_send" for r in records)


def test_training_dislike_recovers_a_reviewer_raised_keyboard(tmp_path, monkeypatch):
    """The Dislike resume DOES recover: no text is sent, so a raised IME costs nothing."""
    events: list = []
    driver, _adb = _driver(events)
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-dislike-recovery")
    driver.set_auto_session_policy(object())
    initial = SimpleNamespace(comment_rect=SimpleNamespace(center=(20, 20)),
                              confirm_point=(30, 30))
    composer = SimpleNamespace(comment_rect=SimpleNamespace(center=(21, 21)),
                               confirm_point=(40, 40))
    frames = iter((b"SHEET", b"FOCUSED", b"KEYBOARD_HIDDEN",
                   b"RESUMED_KEYBOARD_UP", b"KEYBOARD_HIDDEN_AGAIN"))

    monkeypatch.setattr(driver, "_verifiable_payload", lambda _item: object())
    monkeypatch.setattr(driver, "_confirm_payload_profile", lambda _item: None)
    monkeypatch.setattr(driver, "_navigate_to_model_item", lambda _item, **_kw: (10, 10))
    monkeypatch.setattr(driver, "_snap", lambda: b"PRE_HEART")
    monkeypatch.setattr(driver, "_await_sheet_open", lambda **_kw: initial)
    monkeypatch.setattr(driver, "_screencap", lambda **_kw: next(frames))
    monkeypatch.setattr(hinge, "locate_inline_composer", lambda *_a, **_kw: initial)
    monkeypatch.setattr(driver, "_locate_inline_composer", lambda _frame: composer)
    monkeypatch.setattr(driver, "_verify_sheet_shows", lambda *_a, **_kw: None)
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(driver, "_verify_training_dislike_landed", lambda _item, **_kw: "identity")
    monkeypatch.setattr(
        hinge, "_match_glyph",
        lambda frame, *_a, **_kw: [] if frame == b"RESUMED_KEYBOARD_UP" else [(80, 80)])
    driver.set_training_decision(lambda _frame, _evidence: "dislike")

    assert driver.like("A targeted opener", model_item_index=1) == "dislike"

    # Two Backs: the post-type dismissal, then the recovery at the resume.
    assert [payload for kind, payload in events if kind == "keyevent"] == [4, 4]
    assert ("tap", (80, 80)) in events           # the X located on the RECOVERED frame
    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    assert any(record["action"] == "training_checkpoint_keyboard_recovery"
               for record in records)


# --- the pass control is proved to the Like path's standard --------------------------------

def test_training_pass_locator_rejects_a_weak_chrome_match(monkeypatch):
    """0.6 was only 0.016 clear of the strongest false peak measured over a whole real run.

    The genuine X correlates at exactly 1.0; anything in the 0.6-0.9 band is UI chrome and must
    not become an irreversible Dislike tap.
    """
    driver, _adb = _driver([])
    seen = {}

    def fake_match(frame, _template, *, side, threshold=0.6, **_kw):
        seen["threshold"] = threshold
        # A 0.62 chrome peak: accepted by the old default, rejected by the measured one.
        return [(70, 300)] if threshold <= 0.62 else []

    monkeypatch.setattr(hinge, "_match_glyph", fake_match)
    monkeypatch.setattr(driver, "_template", lambda _name: object())

    assert driver._locate_training_pass(b"FRAME") is None
    assert seen["threshold"] == hinge._PASS_MATCH_THRESHOLD >= 0.9


def test_training_pass_locator_accepts_the_genuine_glyph(monkeypatch):
    driver, _adb = _driver([])
    monkeypatch.setattr(hinge, "_match_glyph", lambda *_a, **_kw: [(125, 2035)])
    monkeypatch.setattr(driver, "_template", lambda _name: object())
    assert driver._locate_training_pass(b"FRAME") == (125, 2035)


def test_training_pass_locator_refuses_two_plausible_controls(monkeypatch):
    """Ambiguity is an unrecognized screen, never a position tie-break."""
    driver, _adb = _driver([])
    monkeypatch.setattr(hinge, "_match_glyph", lambda *_a, **_kw: [(125, 900), (125, 2035)])
    monkeypatch.setattr(driver, "_template", lambda _name: object())
    with pytest.raises(UnlocatedControlError, match="equally plausible Hinge pass controls"):
        driver._locate_training_pass(b"FRAME")


def test_training_reads_are_audited_as_training_not_auto(tmp_path):
    """The 2026-08-28 ledger stamped 250 of 307 inputs "auto" in a supervised run.

    The per-card decision callback is installed only around the Like/Dislike checkpoint, so the
    profile read that precedes it carried no callback and was classified as autonomous — the
    exact opposite of what this ledger exists to prove.
    """
    driver, _adb = _driver([])
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="training-session-mode")

    driver.begin_training_session()
    driver._audit_device_input("swipe", source="_scroll", transport="UhidTouch")
    # A card checkpoint installs and then clears its callback; the run is still supervised.
    driver.set_training_decision(lambda _frame, _evidence: "like")
    driver._audit_device_input("tap", source="_like_comment_sheet", transport="UhidTouch")
    driver.set_training_decision(None)
    driver._audit_device_input("swipe", source="_scroll_down_one", transport="UhidTouch")

    modes = [json.loads(line)["session_mode"] for line
             in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
             if json.loads(line)["action"] == "device_input"]
    assert modes == ["training", "training", "training"]


def test_comment_draft_digest_reads_only_the_comment_rect(monkeypatch):
    """The digest is retained evidence, so it must describe the FIELD, not the whole frame.

    A digest computed over the entire screenshot would change for any reason at all and be
    useless as a record of the draft. Two frames that differ ONLY outside the comment rect must
    therefore produce the SAME digest, and two that differ only inside it must not.
    """
    import cv2
    import numpy as np

    rect = (95, 1124, 985, 1302)

    def frame(draft_text: str, *, elsewhere: int) -> bytes:
        image = np.full((2400, 1080), 255, dtype=np.uint8)
        image[300:1090, 95:985] = elsewhere        # the item preview, OUTSIDE the comment rect
        cv2.putText(image, draft_text, (rect[0] + 8, rect[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
        return cv2.imencode(".png", image)[1].tobytes()

    driver, _adb = _driver([])
    x0, y0, x1, y1 = rect
    composer = SimpleNamespace(comment_rect=SimpleNamespace(x0=x0, y0=y0, x1=x1, y1=y1))

    same_draft_a = driver._comment_draft_digest(frame("an opener", elsewhere=128), composer)
    same_draft_b = driver._comment_draft_digest(frame("an opener", elsewhere=64), composer)
    other_draft = driver._comment_draft_digest(frame("a different opener", elsewhere=128), composer)

    assert same_draft_a["rect"] == [x0, y0, x1, y1]
    # Blind to everything outside the rect...
    assert same_draft_a["sha256"] == same_draft_b["sha256"]
    # ...and sensitive to what is inside it.
    assert same_draft_a["sha256"] != other_draft["sha256"]


def test_pre_send_checkpoint_never_presses_back_a_second_time(monkeypatch):
    """The pre-send checkpoint runs moments after this code hid the keyboard itself.

    No human review window has opened yet, so a covered pass control there is an unknown screen,
    not a reviewer's tap — and pressing Back into it would be blind. Recovery is opt-in and this
    call site does not opt in.
    """
    events: list = []
    hides: list = []
    composer = SimpleNamespace(comment_rect=SimpleNamespace(center=(21, 21)),
                               confirm_point=(40, 40))
    driver = _checkpoint_driver(
        events, monkeypatch,
        composer_for=lambda _frame: composer,
        glyph_for=lambda *_a, **_kw: [],          # pass control never located
        hides=hides)
    monkeypatch.setattr(driver, "_verify_sheet_shows", lambda *_a, **_kw: None)

    with pytest.raises(UnlocatedControlError, match="no longer be proved unedited"):
        driver._verify_training_checkpoint(b"COVERED", object(), 1, b"BEFORE")

    assert hides == []                            # no Back was pressed
    assert not any(kind == "tap" for kind, _payload in events)


def test_ordinary_dislike_button_is_proved_to_the_same_standard(monkeypatch):
    """The deck Dislike guards the same irreversible touch as the training checkpoint.

    Before 2026-08-28 `_locate_button("pass")` kept _match_glyph's generic 0.6 default while the
    training locator used a measured 0.90, so the two consumers of the SAME template were proved
    to different standards. Over the 346 frames of run 75e832ec6ad7 the genuine X scored exactly
    1.0000 and the strongest non-target left peak was 0.5841 — 0.6 left only 0.016 of margin.
    """
    driver, _adb = _driver([])
    seen = {}

    def fake_match(_frame, _template, *, side, threshold=0.6, **_kw):
        seen["threshold"] = threshold
        return [(70, 300)] if threshold <= 0.62 else []      # a 0.62 chrome peak

    monkeypatch.setattr(hinge, "_match_glyph", fake_match)
    monkeypatch.setattr(driver, "_template", lambda _name: object())
    monkeypatch.setattr(driver, "_screencap", lambda **_kw: b"DECK")

    assert driver._locate_button("pass") is None
    assert seen["threshold"] == hinge._PASS_MATCH_THRESHOLD >= 0.9


def test_ordinary_dislike_button_refuses_two_plausible_controls(monkeypatch):
    """Ambiguity on the deck is an unrecognized screen, not a position tie-break."""
    driver, _adb = _driver([])
    monkeypatch.setattr(hinge, "_match_glyph", lambda *_a, **_kw: [(125, 900), (125, 2035)])
    monkeypatch.setattr(driver, "_template", lambda _name: object())
    monkeypatch.setattr(driver, "_screencap", lambda **_kw: b"DECK")

    with pytest.raises(UnlocatedControlError, match="equally plausible Hinge pass controls"):
        driver._locate_button("pass")


def test_opener_evidence_rows_use_the_run_scoped_session_mode(tmp_path):
    """Evidence rows once had their OWN callback-only copy of the mode rule.

    The worker installs the decision callback only around each checkpoint, so any row written
    outside one was stamped "auto" in a supervised run — two conventions in a single ledger.
    """
    driver, _adb = _driver([])
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="evidence-session-mode")
    driver.begin_training_session()
    assert driver._training_decision is None      # no checkpoint callback installed right now

    driver._record_auto_opener_pre_send(
        b"FRAME", opener="An opener", item_index=None, model_item_index=1)

    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    row = next(r for r in records if r["action"] == "auto_opener_pre_send")
    assert row["session_mode"] == "training"


def test_opener_evidence_rows_carry_the_bound_prompt_era(tmp_path):
    """Worker._bind_opener_prompt_stamp binds this once per session (see
    set_opener_prompt_sha256's own docstring); every one of the three local debug rows this
    module writes for an opener event must carry it, so the local corpus becomes
    era-attributable exactly like the durable `openers` table already is."""
    driver, _adb = _driver([])
    driver.set_auto_session_policy(None)
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="evidence-prompt-era")
    driver.set_opener_prompt_sha256("era-abc123")

    pre_send = driver._record_auto_opener_pre_send(
        b"FRAME", opener="An opener", item_index=None, model_item_index=1)
    resumed = driver._record_auto_opener_resumed_send(
        b"FRAME2", opener="An opener", item_index=None, model_item_index=1,
        approval_evidence=pre_send)
    driver._record_training_cancelled(pre_send, model_item_index=1, reason="stop_requested")

    assert pre_send["prompt_sha256"] == "era-abc123"
    assert resumed["prompt_sha256"] == "era-abc123"

    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    by_action = {r["action"]: r for r in records
                if r["action"] in {"auto_opener_pre_send", "auto_opener_resumed_send",
                                   "training_cancelled"}}
    assert by_action["auto_opener_pre_send"]["prompt_sha256"] == "era-abc123"
    assert by_action["auto_opener_resumed_send"]["prompt_sha256"] == "era-abc123"
    assert by_action["training_cancelled"]["prompt_sha256"] == "era-abc123"


def test_opener_evidence_rows_carry_no_prompt_era_when_never_bound(tmp_path):
    """A driver Worker never called set_opener_prompt_sha256 on (a legacy caller, or an opener
    service unavailable this run) must degrade to no era digest, not raise."""
    driver, _adb = _driver([])
    driver.set_auto_session_policy(None)
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="evidence-no-prompt-era")

    evidence = driver._record_auto_opener_pre_send(
        b"FRAME", opener="An opener", item_index=None, model_item_index=1)

    assert evidence["prompt_sha256"] is None
    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    row = next(r for r in records if r["action"] == "auto_opener_pre_send")
    assert row["prompt_sha256"] is None


def test_training_presend_evidence_carries_the_bound_generation_context(tmp_path):
    driver, _adb = _driver([])
    driver.begin_training_session()
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="generation-context")
    context = {
        "model": "gemini-test-model",
        "index_space": "model_items",
        "referenced": "two brown dachshunds on a yoga mat",
        "angle": "playfully asking whether the dogs are spreadsheet tabs",
        "item_description": "photo of two dachshunds",
    }
    driver.set_staged_opener_generation_context(context)

    evidence = driver._record_auto_opener_pre_send(
        b"FRAME", opener="Do those two have their own tabs?", item_index=None,
        model_item_index=2)

    assert evidence["generation_context"] == context
    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    row = next(record for record in records if record["action"] == "auto_opener_pre_send")
    for key, value in context.items():
        assert row[key] == value
    driver.set_staged_opener_generation_context(None)
    assert driver._staged_opener_generation_context is None


def test_current_profile_identity_reads_the_held_index(tmp_path):
    """ranker/profile_key.py's one sanctioned consumer (worker.py's _current_profile_key) reads
    this off whatever `_current_item_index` the driver is currently holding -- a pure state
    read, never a capture. None before anything is captured, the held index's own `.identity`
    once one exists, and None again once the index is invalidated (like()/dislike()'s own
    `finally`, doc 5.3) -- exactly the ONE-PROFILE lifetime this method's docstring promises."""
    driver, _adb = _driver([])
    assert driver.current_profile_identity() is None   # nothing captured yet this session

    fake_identity = object()
    driver._current_item_index = SimpleNamespace(identity=fake_identity)
    assert driver.current_profile_identity() is fake_identity

    driver._invalidate_item_index("simulating like()/dislike()'s own post-action invalidation")
    assert driver.current_profile_identity() is None


def test_training_advance_probe_retains_the_frames_that_license_the_label(tmp_path, monkeypatch):
    """The accepted probe is the SOLE proof that a training label describes a real advance.

    Until 2026-08-28 that row carried no image at all, so a disputed label could never be
    re-checked against what was actually on screen. `keep_before` puts the first settled frame in
    the bounded retained-evidence pool, where ordinary screenshot rotation cannot erase it.
    """
    driver, _adb = _driver([])
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="advance-probe-evidence")
    frames = iter((b"FIRST_SETTLED", b"SECOND_SETTLED"))

    monkeypatch.setattr(driver, "_screencap", lambda **_kw: next(frames))
    monkeypatch.setattr(driver, "_interruptible_sleep", lambda *_a, **_kw: True)
    monkeypatch.setattr(driver, "_training_dislike_surface_proof",
                        lambda _frame, _item, diagnostics=None: ("identity", None))
    monkeypatch.setattr(driver, "_changed", lambda _a, _b: False)   # both frames settled

    assert driver._verify_training_deck_advanced(1) == "identity"

    records = [json.loads(line) for line
               in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()]
    probe = next(r for r in records if r["action"] == "training_advance_probe")
    assert probe["outcome"] == "accepted"
    assert probe["before"] and probe["after"]          # both settled frames are on disk
    assert probe["kept_before"] == probe["before"]     # and the first is rotation-proof
    for name in (probe["before"], probe["after"]):
        assert (driver._dbg.dir / name).exists()
