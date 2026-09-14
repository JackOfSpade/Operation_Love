"""Hermetic safety tests for the targeted-opener calibration harness.

The real inputs are private dating-profile screenshots.  These tests deliberately use only
small synthetic records and replace the two vision pipelines at the boundary of ``measure``;
the safety protocol (splits, ceilings, frozen held-out bounds, and emitted YAML) is what this
file verifies.
"""
from __future__ import annotations

import argparse
import ast
import io
import hashlib
import inspect
import json
import os
import re
import stat
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from operation_love import targeting_policy as tp
from tools import hinge_calibrate as cal
from operation_love.drivers.item_identity import ProfileIdentity
from operation_love.drivers.like_composer import ComposerDetectionError, ComposerSurface, Rect
from operation_love.drivers.scroll_top import (
    SCROLL_TOP_CONFIRMED, SCROLL_TOP_UNKNOWN, ScrollTopVerdict)


_IDENTITY_BAND = (0.10, 0.048, 0.80, 0.094)
_CONTENT_BAND = (0.125, 0.875)


class _Cfg:
    apps = {"hinge": {"serial": "PIXEL-TEST"}}


class _InertDriver:
    """The only driver shape measure may need: no ADB or touch methods exist here."""

    def __init__(self, _cfg):
        self.identity_band = _IDENTITY_BAND
        self.content_band = _CONTENT_BAND

    def _template(self, name):
        assert name in {"like", "confirm"}
        return object()


def _session(tmp_path, name, split, profile_ids, *, identity_band=_IDENTITY_BAND,
             content_band=_CONTENT_BAND, serial="PIXEL-TEST"):
    directory = tmp_path / name
    directory.mkdir()
    profiles = [cal._ProfileData(
        i, pid, [b"card"],
        [(cal._automated_composer_items_for_ordinal(i)[0], b"pre", b"composer")],
        b"advance", b"advance-identity")
                for i, pid in enumerate(profile_ids, 1)]
    return cal._SessionData(
        dir=directory, split=split, device_serial=serial, identity_band=identity_band,
        content_band=content_band, profiles=profiles,
        manifest={"device": _device(serial), "operational_checks": _checks(directory)},
    )


def _device(serial="PIXEL-TEST"):
    return {"serial": serial, "model": "Synthetic Pixel", "display_w": 1080,
            "display_h": 2400, "density": 420, "hinge_package": "co.hinge.app",
            "hinge_version_name": "1.0"}


def _checks(directory: Path | None = None, *, confirmed=True):
    evidence = "synthetic-run-42"
    if confirmed and directory is not None:
        record = {
            "kind": "hinge_supervised_observe_only_check",
            "evidence_scope": cal._PRELIMINARY_OBSERVE_SCOPE,
            "completed": True,
        }
        record["evidence_sha256"] = cal._canonical_json_digest(record)
        path = directory / "observe_check.json"
        path.write_text(json.dumps(record))
        evidence = str(path)
    return {
        key: {"confirmed": confirmed, "evidence": evidence if confirmed else None,
              "recorded_utc": "2026-08-13T12:00:00+00:00" if confirmed else None}
        for key in cal._OPERATIONAL_CHECKS
    }


def _write_operational_evidence(directory: Path, *, device: dict,
                                identity_band=_IDENTITY_BAND) -> Path:
    """Small, real PNG recorder-v2 artifact for measure's evidence-gate seams."""
    from PIL import Image

    directory.mkdir()
    raws = []
    for value in range(1, len(cal._OPERATIONAL_EVIDENCE_ROLES) + 1):
        frame = io.BytesIO()
        Image.new("RGB", (device["display_w"], device["display_h"]), (value, value, value)).save(frame, "PNG")
        raws.append(frame.getvalue())
    frames = []
    for ordinal, role in enumerate(cal._OPERATIONAL_EVIDENCE_ROLES, 1):
        name = f"{ordinal:05d}.png"
        raw = raws[ordinal - 1]
        (directory / name).write_bytes(raw)
        frames.append({"file": name, "sha256": hashlib.sha256(raw).hexdigest(), "role": role,
                       "captured_utc": "2026-08-13T12:00:00+00:00"})
    composer = {"layout_id": cal._COMPOSER_LAYOUT_ID,
                "comment_rect": {"x0": 1, "y0": 2, "x1": 3, "y1": 4},
                "send_rect": {"x0": 5, "y0": 6, "x1": 7, "y1": 8}, "confirm_point": [6, 7]}
    top = {"state": "confirmed_top", "distance": 0.0, "reason": "top", "grid": [16, 4]}
    sticky = {"state": "confirmed_not_top", "distance": 20.0, "reason": "sticky", "grid": [16, 4]}
    analyses = {
        "confirmed_top_item1_pre": {"scroll_top": top},
        "composer_initial_autofocused": {
            "composer": composer, "scroll_top": sticky},
        "composer_stable_autofocused": {
            "composer": composer, "scroll_top": sticky},
        "other_item_composer_moved": {"composer": composer},
        "item1_profile_identity": {"fingerprint": [10, 10], "grid": [64, 16],
                                    "frame_index": 1, "reason": "corroborated"},
        "new_profile_top_clear": {
            "scroll_top": top, "composer_absent": True},
        "new_sticky_identity": {
            "scroll_top": sticky, "fingerprint": [30, 30], "grid": [64, 16],
            "distance_from_item1_profile": 20.0},
    }
    manifest = {
        "tool_version": cal._OPERATIONAL_EVIDENCE_TOOL_VERSION,
        "completed": True, "interrupted": False, "device": device,
        "frame_size_px": [device["display_w"], device["display_h"]],
        "identity_band": list(identity_band), "composer_layout_id": cal._COMPOSER_LAYOUT_ID,
        "config_binding": {"serial": device["serial"], "identity_band": list(identity_band),
                           "composer_layout_id": cal._COMPOSER_LAYOUT_ID,
                           "hinge_package": device["hinge_package"]},
        "new_profile_min_identity_distance": cal._IDENTITY_FALSE_MATCH_DISTANCE,
        "start_utc": "2026-08-13T12:00:00+00:00", "end_utc": "2026-08-13T12:01:00+00:00",
        "frame_count": len(frames), "frames": frames, "analyses": analyses, "failure": None,
    }
    manifest["evidence_sha256"] = cal._canonical_json_digest(manifest)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _wire_operational_replay(monkeypatch, path: Path):
    manifest = json.loads(path.read_text())
    by_raw = {(path.parent / rec["file"]).read_bytes(): ordinal
              for ordinal, rec in enumerate(manifest["frames"])}
    surface = ComposerSurface(cal._COMPOSER_LAYOUT_ID, Rect(1, 2, 3, 4), Rect(5, 6, 7, 8), (6, 7))
    top = ScrollTopVerdict("confirmed_top", 0.0, "top", _IDENTITY_BAND, (16, 4), 3.0, 9.0)
    sticky = ScrollTopVerdict("confirmed_not_top", 20.0, "sticky", _IDENTITY_BAND, (16, 4), 3.0, 9.0)
    monkeypatch.setattr(cal.hinge_mod, "_load_template", lambda _name: object())
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda raw, **_kw: top if by_raw[raw] in (0, 4) else sticky)
    def locate(raw, *_args, **_kw):
        if by_raw[raw] == 4:
            raise ComposerDetectionError("absent")
        return surface
    monkeypatch.setattr(cal, "locate_inline_composer", locate)
    identity = ProfileIdentity(fingerprint=(10, 10), band=_IDENTITY_BAND, grid=(64, 16),
                               frame_index=1, scroll_top_distance=20.0, reason="corroborated")
    monkeypatch.setattr(cal, "capture_profile_identity", lambda *_a, **_kw: identity)
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_a, **_kw: (30, 30))


def _samples(prefix, values):
    """A representative and inline-composer fingerprint for each synthetic profile."""
    out = []
    for number, (card, composer) in enumerate(values, 1):
        out.extend((cal._IdentitySample(f"{prefix}{number}", "representative", (card,)),
                    cal._IdentitySample(f"{prefix}{number}", "composer:1", (composer,))))
    return out


def _record(kind, distance, *, profile="p", item=1, against="q", against_item=1):
    return cal._InlineDistanceRecord(kind, profile, item, against, against_item, distance)


def _top_verdict(state: str, reason: str = "test"):
    return SimpleNamespace(
        state=state,
        confirmed=state == "confirmed_top",
        refuted=state == "confirmed_not_top",
        reason=reason,
    )


class _HybridRewindDriver:
    """Small transport double: only the visual-rewind surface is available."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.position = 0
        self.adb = self
        self.reverse_steps = []

    def screencap(self):
        return self.frames[min(self.position, len(self.frames) - 1)]

    def _scroll_up_one(self, frac, x_frac):
        self.reverse_steps.append((frac, x_frac))
        if self.position < len(self.frames) - 1:
            self.position += 1


def _rewind_driver_plan(monkeypatch):
    planned = SimpleNamespace(frac=0.16, x_frac=0.47, spacing_px=700)
    calls = []
    monkeypatch.setattr(
        cal, "_plan_card_scroll",
        lambda frame, **kwargs: (calls.append((frame, kwargs)) or (planned, 700)))
    return calls, planned


def test_hybrid_rewind_recovers_a_mid_profile_without_scroll_ledger(monkeypatch):
    """The entry top is visually driven, not replayed from process-local history."""
    driver = _HybridRewindDriver([b"mid-profile", b"top"])
    calls, planned = _rewind_driver_plan(monkeypatch)
    monkeypatch.setattr(
        cal, "confirm_scroll_top",
        lambda frame, **_kw: _top_verdict(
            "confirmed_top" if frame == b"top" else "confirmed_not_top"))

    top = cal._rewind_automated_profile_to_confirmed_top(
        driver, ordinal=4, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=0.8)

    assert top == b"top"
    assert driver.reverse_steps == [(planned.frac, planned.x_frac)]
    assert len(calls) == 1
    planned_frame, planned_kwargs = calls[0]
    assert planned_frame == b"mid-profile"
    assert planned_kwargs["content_band"] == _CONTENT_BAND
    assert planned_kwargs["like_threshold"] == 0.8
    assert planned_kwargs["profile_min_spacing_px"] is None


def test_hybrid_rewind_at_confirmed_top_issues_no_gesture(monkeypatch):
    driver = _HybridRewindDriver([b"top"])
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_args, **_kw: _top_verdict("confirmed_top"))
    monkeypatch.setattr(cal, "_plan_card_scroll",
                        lambda *_args, **_kw: pytest.fail("top needs no scroll plan"))

    assert cal._rewind_automated_profile_to_confirmed_top(
        driver, ordinal=1, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=0.8) == b"top"
    assert driver.reverse_steps == []


def test_hybrid_rewind_refuses_unknown_before_a_gesture(monkeypatch):
    driver = _HybridRewindDriver([b"unknown"])
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_args, **_kw: _top_verdict("unknown", "ambiguous strip"))

    with pytest.raises(cal._CaptureAbort, match="unconfirmed scroll-top state"):
        cal._rewind_automated_profile_to_confirmed_top(
            driver, ordinal=1, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)
    assert driver.reverse_steps == []


def test_hybrid_rewind_allows_one_unknown_recovery_only_for_a_proven_live_deck(monkeypatch):
    """An ambiguous chip band needs current Like+Pass deck proof before one recovery swipe."""
    driver = _HybridRewindDriver([b"unknown-live-deck", b"top"])
    driver._observe_deck_ready = lambda frame: frame == b"unknown-live-deck"
    calls, planned = _rewind_driver_plan(monkeypatch)
    monkeypatch.setattr(
        cal, "confirm_scroll_top",
        lambda frame, **_kw: _top_verdict(
            "confirmed_top" if frame == b"top" else "cannot_tell", "ambiguous chips"))

    assert cal._rewind_automated_profile_to_confirmed_top(
        driver, ordinal=1, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=0.8) == b"top"
    assert driver.reverse_steps == [(planned.frac, planned.x_frac)]
    assert len(calls) == 1


def test_hybrid_rewind_unknown_dialog_or_paywall_proof_failure_never_swipes(monkeypatch):
    """Foreground Hinge alone is insufficient: without both deck controls UNKNOWN stays no-input."""
    driver = _HybridRewindDriver([b"unknown-modal"])
    driver._observe_deck_ready = lambda _frame: False
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_args, **_kw: _top_verdict("cannot_tell", "ambiguous strip"))
    monkeypatch.setattr(cal, "_plan_card_scroll",
                        lambda *_args, **_kw: pytest.fail("a modal/paywall must not plan a swipe"))

    with pytest.raises(cal._CaptureAbort, match="refused an unconfirmed scroll-top state"):
        cal._rewind_automated_profile_to_confirmed_top(
            driver, ordinal=1, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)
    assert driver.reverse_steps == []


def test_hybrid_rewind_never_spends_a_second_unknown_deck_recovery_gesture(monkeypatch):
    """A persistent UNKNOWN is not promoted into a scrolling loop, even on a proven deck."""
    driver = _HybridRewindDriver([b"unknown-1", b"unknown-2", b"top"])
    driver._observe_deck_ready = lambda frame: frame.startswith(b"unknown")
    calls, _planned = _rewind_driver_plan(monkeypatch)
    monkeypatch.setattr(cal, "_MAX_UNSETTLED_TOP_REPROBES", 0)
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_args, **_kw: _top_verdict("cannot_tell", "still ambiguous"))

    with pytest.raises(cal._UnsupportedEntryDeck, match="unsupported entry deck") as caught:
        cal._rewind_automated_profile_to_confirmed_top(
            driver, ordinal=1, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)
    assert caught.value.frame == b"unknown-2"
    assert len(calls) == len(driver.reverse_steps) == 1


def test_hybrid_rewind_spends_its_unknown_allowance_on_a_post_gesture_dead_zone(monkeypatch):
    """A first post-gesture `cannot_tell` is the ORDINARY 10.2 entry, not an exotic deck.

    Every automated profile's entry rewind starts from the refuted position the terminal Pass
    proved its sticky header at, and one stroke up from there reads inside scroll_top's
    deliberate 3..9 dead zone (measured live 2026-09-04: 7.859; the confirmable top is one more
    stroke away at 12.281).  Escalating that reading immediately spent a real public Pass -- or a
    permanent Send Priority Like under --send-like -- on a capturable profile, so the single
    guarded UNKNOWN allowance must apply to a post-gesture frame exactly as it does to the entry
    frame.  A SECOND consecutive dead-zone reading still escalates (test above).
    """
    driver = _HybridRewindDriver([b"mid-card", b"dead-zone", b"top"])
    driver._observe_deck_ready = lambda frame: frame == b"dead-zone"
    driver.dislike = lambda: pytest.fail("the bounded rewind must never advance a real profile")
    driver.like = lambda: pytest.fail("the bounded rewind must never advance a real profile")
    calls, planned = _rewind_driver_plan(monkeypatch)
    monkeypatch.setattr(cal, "_MAX_UNSETTLED_TOP_REPROBES", 0)
    monkeypatch.setattr(cal, "confirm_scroll_top", lambda frame, **_kw: _top_verdict(
        {b"mid-card": "confirmed_not_top", b"dead-zone": "cannot_tell"}.get(frame,
                                                                           "confirmed_top"),
        "10.2 tall first-card header"))

    assert cal._rewind_automated_profile_to_confirmed_top(
        driver, ordinal=3, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=0.8) == b"top"

    assert driver.reverse_steps == [(planned.frac, planned.x_frac)] * 2
    assert len(calls) == 2


def test_unsupported_entry_skip_uses_public_pass_and_never_claims_identity(monkeypatch):
    class _Adb:
        frame = b"unsupported-entry"

        def screencap(self):
            return self.frame

    class _Driver:
        dwell_s = 0.0

        def __init__(self):
            self.adb = _Adb()
            self.dislikes = 0

        def _observe_deck_ready(self, _frame):
            return True

        def _template(self, _name):
            return object()

        def dislike(self):
            self.dislikes += 1
            self.adb.frame = b"next-deck"

    driver = _Driver()
    monkeypatch.setattr(
        cal, "locate_inline_composer",
        lambda *_a, **_kw: (_ for _ in ()).throw(ComposerDetectionError("absent")))
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_a, **_kw: _top_verdict("cannot_tell", "next special deck"))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal.time, "sleep", lambda _seconds: None)

    record = cal._skip_automated_unsupported_entry_deck(
        driver, ordinal=2,
        entry=cal._UnsupportedEntryDeck(
            b"unsupported-entry", state="cannot_tell", reason="moved chip row"),
        identity_band=_IDENTITY_BAND)

    assert driver.dislikes == 1
    assert record["reason_code"] == "unsupported_entry_layout"
    assert record["transport"] == "HingeDriver.dislike"
    assert record["pre_scroll_top_state"] == "cannot_tell"
    assert record["post_scroll_top_state"] == "cannot_tell"
    assert record["predicates"]["public_action_progress_verified"] is True
    assert record["predicates"]["new_deck_ready"] is True
    assert record["predicates"]["new_profile_identity_distinct"] is False
    assert record["predicates"]["identity_comparison_not_claimed"] is True


def test_hybrid_rewind_escalates_a_proven_deck_at_the_loop_top_once_the_allowance_is_spent(
        monkeypatch):
    """The two UNKNOWN branches must agree about what a proven deck means.

    A real screen can settle differently between the post-gesture read and the next top-of-loop
    read, so the loop top can meet an UNKNOWN with the allowance already spent. On a frame that
    independently proves both deck controls that is the same advanceable-but-unusable layout the
    post-gesture branch escalates, not a run-ending mystery; an UNPROVEN deck still aborts
    (tests above).
    """
    driver = _HybridRewindDriver([b"unknown-entry", b"moved"])
    driver._observe_deck_ready = lambda _frame: True
    reads: list[bytes] = []

    def flipping_confirm(frame, **_kw):
        reads.append(frame)
        # `moved` reads refuted right after the gesture and ambiguous on the next look.
        return _top_verdict(
            "confirmed_not_top" if frame == b"moved" and reads.count(b"moved") == 1
            else "cannot_tell", "chip row drifted")

    calls, _planned = _rewind_driver_plan(monkeypatch)
    monkeypatch.setattr(cal, "_MAX_UNSETTLED_TOP_REPROBES", 0)
    monkeypatch.setattr(cal, "confirm_scroll_top", flipping_confirm)

    with pytest.raises(cal._UnsupportedEntryDeck, match="unsupported entry deck") as caught:
        cal._rewind_automated_profile_to_confirmed_top(
            driver, ordinal=1, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)

    assert caught.value.frame == b"moved"
    assert len(calls) == len(driver.reverse_steps) == 1


def test_unsupported_entry_escalation_keeps_the_diagnosis_that_asked_for_the_skip(monkeypatch):
    """A pre-action skip escalated by the unsupported layout must not lose WHY it was skipping.

    The layout symptom alone tells the operator nothing about the target/navigation failure that
    sent the profile down the skip path. The original code+detail go into `reason_detail`, which
    `reason_sha256` binds, so the record stays self-verifying and the reviewer's checkpoint
    approves the real diagnosis rather than the symptom.
    """
    class _Adb:
        frame = b"unsupported-entry"

        def screencap(self):
            return self.frame

    class _Driver:
        dwell_s = 0.0

        def __init__(self):
            self.adb = _Adb()
            self.dislikes = 0

        def _observe_deck_ready(self, _frame):
            return True

        def _template(self, _name):
            return object()

        def dislike(self):
            self.dislikes += 1
            self.adb.frame = b"next-deck"

    driver = _Driver()
    gate = _RecordingGate()
    monkeypatch.setattr(
        cal, "locate_inline_composer",
        lambda *_a, **_kw: (_ for _ in ()).throw(ComposerDetectionError("absent")))
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_a, **_kw: _top_verdict("cannot_tell", "next special deck"))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal.time, "sleep", lambda _seconds: None)

    record = cal._skip_automated_unsupported_entry_deck(
        driver, ordinal=2,
        entry=cal._UnsupportedEntryDeck(
            b"unsupported-entry", state="cannot_tell", reason="moved chip row"),
        identity_band=_IDENTITY_BAND, review_gate=gate,
        skip_reason=cal._PreActionProfileRetry(_LIVE_SKIP_CODE, _LIVE_SKIP_DETAIL))

    assert _LIVE_SKIP_CODE in record["reason_detail"]
    assert _LIVE_SKIP_DETAIL in record["reason_detail"]
    assert "moved chip row" in record["reason_detail"]
    assert record["reason_sha256"] == hashlib.sha256(
        f"{record['reason_code']}\n{record['reason_detail']}".encode("utf-8")).hexdigest()
    (_claimed_state, plan), = gate.checkpoints
    assert plan["predicates"]["skip_reason_detail"] == record["reason_detail"]


def test_hybrid_rewind_refuses_an_unplannable_or_stalled_refuted_frame(monkeypatch):
    driver = _HybridRewindDriver([b"mid"])
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_args, **_kw: _top_verdict("confirmed_not_top", "sticky"))
    monkeypatch.setattr(cal, "_plan_card_scroll",
                        lambda *_args, **_kw: (_ for _ in ()).throw(
                            cal.ScrollStepError("no safe step")))

    with pytest.raises(cal._CaptureAbort, match="could not plan a guarded upward scroll"):
        cal._rewind_automated_profile_to_confirmed_top(
            driver, ordinal=1, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)
    assert driver.reverse_steps == []

    calls, _planned = _rewind_driver_plan(monkeypatch)
    with pytest.raises(cal._CaptureAbort, match="rewind stalled"):
        cal._rewind_automated_profile_to_confirmed_top(
            driver, ordinal=1, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)
    assert len(calls) == 1 and len(driver.reverse_steps) == 1


def test_hybrid_rewind_has_a_hard_visual_gesture_budget(monkeypatch):
    driver = _HybridRewindDriver([b"mid-0", b"mid-1", b"mid-2"])
    calls, _planned = _rewind_driver_plan(monkeypatch)
    monkeypatch.setattr(cal, "_MAX_AUTOMATED_TOP_REWIND_STEPS", 2)
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_args, **_kw: _top_verdict("confirmed_not_top", "sticky"))

    with pytest.raises(cal._CaptureAbort, match="exceeded its bounded 2-gesture budget"):
        cal._rewind_automated_profile_to_confirmed_top(
            driver, ordinal=1, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)

    assert len(driver.reverse_steps) == len(calls) == 2


def test_hybrid_rewind_settled_verdict_reprobes_a_transient_cannot_tell_without_a_gesture(monkeypatch):
    """Hinge 10.0.1 can settle its filter-chips strip a few px late (see
    `_MAX_UNSETTLED_TOP_REPROBES`): a screencap taken mid-settle can land in the deliberate
    3..9 `cannot_tell` dead zone even though the same resting screen reads an exact confirmed top
    a moment later. The rewind must re-read the screen -- no gesture -- rather than either
    aborting on the transient reading or treating it as a positive refutation that licenses a
    scroll."""
    driver = _HybridRewindDriver([b"settling"])
    verdicts = iter([_top_verdict("cannot_tell", "mid-settle"), _top_verdict("confirmed_top", "settled")])
    calls = []

    def fake_confirm(frame, **_kw):
        calls.append(frame)
        return next(verdicts)

    monkeypatch.setattr(cal, "confirm_scroll_top", fake_confirm)
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "_plan_card_scroll",
                        lambda *_a, **_kw: pytest.fail("a settling reprobe must not plan/issue a gesture"))

    top = cal._rewind_automated_profile_to_confirmed_top(
        driver, ordinal=1, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=0.8)

    assert top == b"settling"
    assert calls == [b"settling", b"settling"]
    assert driver.reverse_steps == []


def test_post_advance_identity_probe_reads_a_settled_frame_before_spending_a_second_scroll(
        monkeypatch):
    """A mid-snap frame is not an indeterminate profile (found 2026-09-04).

    `adb shell input swipe` returns while Hinge is still snapping the new profile, so the first
    post-gesture frame can still show the filter-chip row -- which `confirm_scroll_top` cannot
    refute. Reading that frame raw turned a perfectly ordinary advance into a second real device
    gesture, and handed the caller a mid-animation frame that is then consumed directly as
    identity evidence. This fixture is that phone: one in-flight frame, then the parked one.
    """
    class Driver:
        identity_band = _IDENTITY_BAND
        content_band = _CONTENT_BAND
        dwell_s = 0.0

        def __init__(self):
            self.adb = self
            self.frames = iter((b"partial-header", b"sticky-header"))
            self.last = None
            self.scrolls = []

        def screencap(self):
            self.last = next(self.frames, self.last)
            return self.last

        def _scroll_down_one(self, frac, x_frac):
            self.scrolls.append((frac, x_frac))

        def _observe_deck_ready(self, frame):
            return frame == b"partial-header"

    driver = Driver()
    monkeypatch.setattr(
        cal, "_plan_card_scroll",
        lambda *_a, **_kw: (SimpleNamespace(frac=0.13, x_frac=0.5), 900))
    monkeypatch.setattr(
        cal, "confirm_scroll_top",
        lambda frame, **_kw: _top_verdict(
            "cannot_tell" if frame == b"partial-header" else "confirmed_not_top"))

    result = cal._scroll_next_profile_to_sticky_header(
        driver, top_frame=b"top", identity_band=_IDENTITY_BAND,
        content_band=_CONTENT_BAND, like_template=object(), like_threshold=0.8,
        context="automated profile 1")

    assert result == b"sticky-header", "the settled frame is what the caller must be handed"
    assert driver.scrolls == [(0.13, 0.5)], "an in-flight frame must not buy a second gesture"


def test_post_advance_identity_probe_allows_one_guarded_second_scroll_for_hinge_10_2(
        monkeypatch):
    """Hinge 10.2 can leave the first post-Pass read in the 3..9 top dead zone.

    A second locally planned scroll is safe only while the first result still proves the
    ordinary Like+Pass deck. It must expose a positively refuted sticky header rather than
    promoting the indeterminate first result into an identity verdict.

    Unlike the settle test above, this phone is genuinely PARKED on the indeterminate frame --
    every read after the first gesture returns the same bytes -- so the second scroll is the
    only thing that can resolve it.
    """
    class Driver:
        identity_band = _IDENTITY_BAND
        content_band = _CONTENT_BAND
        dwell_s = 0.0

        def __init__(self):
            self.adb = self
            self.scrolls = []

        def screencap(self):
            return b"partial-header" if len(self.scrolls) < 2 else b"sticky-header"

        def _scroll_down_one(self, frac, x_frac):
            self.scrolls.append((frac, x_frac))

        def _observe_deck_ready(self, frame):
            return frame == b"partial-header"

    driver = Driver()
    planned_spacing = []

    def plan(_frame, **kwargs):
        planned_spacing.append(kwargs["profile_min_spacing_px"])
        return SimpleNamespace(frac=0.13, x_frac=0.5), 900

    monkeypatch.setattr(cal, "_plan_card_scroll", plan)
    monkeypatch.setattr(
        cal, "confirm_scroll_top",
        lambda frame, **_kw: _top_verdict(
            "cannot_tell" if frame == b"partial-header" else "confirmed_not_top"))

    result = cal._scroll_next_profile_to_sticky_header(
        driver, top_frame=b"top", identity_band=_IDENTITY_BAND,
        content_band=_CONTENT_BAND, like_template=object(), like_threshold=0.8,
        context="automated profile 1")

    assert result == b"sticky-header"
    assert driver.scrolls == [(0.13, 0.5), (0.13, 0.5)]
    assert planned_spacing == [None, 900]


def test_post_advance_identity_probe_refuses_second_scroll_without_fresh_deck_proof(
        monkeypatch):
    class Driver:
        identity_band = _IDENTITY_BAND
        content_band = _CONTENT_BAND
        dwell_s = 0.0

        def __init__(self):
            self.adb = self
            self.scrolls = []

        def screencap(self):
            return b"partial-header"

        def _scroll_down_one(self, frac, x_frac):
            self.scrolls.append((frac, x_frac))

        def _observe_deck_ready(self, _frame):
            return False

    driver = Driver()
    monkeypatch.setattr(
        cal, "_plan_card_scroll",
        lambda *_a, **_kw: (SimpleNamespace(frac=0.13, x_frac=0.5), 900))
    monkeypatch.setattr(
        cal, "confirm_scroll_top",
        lambda *_a, **_kw: _top_verdict("cannot_tell", "10.2 partial header"))

    with pytest.raises(cal._CaptureAbort, match="ordinary deck could not be re-proved"):
        cal._scroll_next_profile_to_sticky_header(
            driver, top_frame=b"top", identity_band=_IDENTITY_BAND,
            content_band=_CONTENT_BAND, like_template=object(), like_threshold=0.8,
            context="automated profile 1")

    assert driver.scrolls == [(0.13, 0.5)]


def _post_advance_reprobe_driver(frames):
    class Driver:
        dwell_s = 0.0

        def __init__(self):
            self.adb = self
            self.frames = iter(frames)
            self.reads = []
            self.gestures = []

        def screencap(self):
            frame = next(self.frames)
            self.reads.append(frame)
            return frame

        # The read-only identity helper must never reach any of these transports.  Keeping them
        # as observable fakes makes an accidental future recovery gesture a test failure.
        def _scroll_down_one(self, *_args):
            self.gestures.append("scroll")

        def _scroll_up_one(self, *_args):
            self.gestures.append("reverse-scroll")

        def _swipe(self, *_args):
            self.gestures.append("edge-back")

        def _tap(self, *_args):
            self.gestures.append("tap")

    return Driver()


def _wire_post_advance_reprobe_measurement(monkeypatch, *, fingerprints, top_states=None,
                                           shifts=None):
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "band_fingerprint", lambda frame, **_kw: fingerprints[frame])
    monkeypatch.setattr(cal, "fingerprint_distance", lambda left, right: abs(left[0] - right[0]))
    top_states = top_states or {}
    monkeypatch.setattr(
        cal, "confirm_scroll_top",
        lambda frame, **_kw: _top_verdict(top_states.get(frame, "confirmed_not_top")))
    shifts = shifts or {}
    monkeypatch.setattr(
        cal, "estimate_shift",
        lambda left, right, **_kw: SimpleNamespace(
            ok=True, delta_px=shifts.get((left, right), 0), reason="synthetic"))


def test_post_advance_identity_reprobe_accepts_two_late_sticky_frames_without_a_gesture(
        monkeypatch):
    """One late repaint may be transitional; the final adjacent pair must prove the repair."""
    driver = _post_advance_reprobe_driver((b"transitional", b"late-1", b"late-2"))
    _wire_post_advance_reprobe_measurement(
        monkeypatch,
        fingerprints={b"initial": (0,), b"transitional": (0,), b"late-1": (10,),
                      b"late-2": (10,)})

    frame, distance, trace = cal._prove_post_advance_identity_distinct(
        driver, prior_fingerprint=(0,), initial_frame=b"initial", identity_band=_IDENTITY_BAND,
        content_band=_CONTENT_BAND, context="automated profile 2")

    assert (frame, distance) == (b"late-2", 10.0)
    assert driver.reads == [b"transitional", b"late-1", b"late-2"]
    assert driver.gestures == []
    assert trace["read_only_reprobe_count"] == 3
    assert trace["accepted_read_only_reprobe_pair"] == [2, 3]
    assert [sample["distance_from_prior"] for sample in trace["identity_distance_samples"]] == [
        0.0, 0.0, 10.0, 10.0]


def test_post_advance_identity_reprobe_refuses_persistent_collision_without_a_gesture(monkeypatch):
    driver = _post_advance_reprobe_driver((b"again-1", b"again-2", b"again-3"))
    _wire_post_advance_reprobe_measurement(
        monkeypatch,
        fingerprints={b"initial": (0,), b"again-1": (0,), b"again-2": (1,),
                      b"again-3": (2,)})

    with pytest.raises(cal._CaptureAbort,
                       match=r"no gesture was issued and identity thresholds unchanged"):
        cal._prove_post_advance_identity_distinct(
            driver, prior_fingerprint=(0,), initial_frame=b"initial", identity_band=_IDENTITY_BAND,
            content_band=_CONTENT_BAND, context="automated profile 2")

    assert len(driver.reads) == 3
    assert driver.gestures == []


@pytest.mark.parametrize(
    ("top_states", "shifts", "fingerprints"),
    [
        ({}, {}, {b"initial": (0,), b"one": (10,), b"two": (14,), b"three": (20,)}),
        ({}, {(b"one", b"two"): 4, (b"two", b"three"): 4},
         {b"initial": (0,), b"one": (10,), b"two": (10,), b"three": (10,)}),
        ({b"one": "confirmed_top", b"two": "confirmed_top", b"three": "confirmed_top"}, {},
         {b"initial": (0,), b"one": (10,), b"two": (10,), b"three": (10,)}),
    ], ids=["disagreement", "motion", "not-refuted"])
def test_post_advance_identity_reprobe_refuses_disagreement_motion_or_nonsticky_frame(
        monkeypatch, top_states, shifts, fingerprints):
    driver = _post_advance_reprobe_driver((b"one", b"two", b"three"))
    _wire_post_advance_reprobe_measurement(
        monkeypatch, fingerprints=fingerprints, top_states=top_states, shifts=shifts)

    with pytest.raises(cal._CaptureAbort, match="read-only screencaps; no gesture was issued"):
        cal._prove_post_advance_identity_distinct(
            driver, prior_fingerprint=(0,), initial_frame=b"initial", identity_band=_IDENTITY_BAND,
            content_band=_CONTENT_BAND, context="automated profile 2")

    assert len(driver.reads) == 3
    assert driver.gestures == []


def test_every_automated_profile_enters_through_the_visual_rewind(monkeypatch, tmp_path):
    class Driver:
        identity_band = _IDENTITY_BAND
        content_band = _CONTENT_BAND

        def _template(self, _name):
            return object()

    calls = []

    def stop_at_entry(_driver, **kwargs):
        calls.append(kwargs)
        raise cal._CaptureAbort("entry gate reached")

    monkeypatch.setattr(cal, "_rewind_automated_profile_to_confirmed_top", stop_at_entry)
    with pytest.raises(cal._CaptureAbort, match="entry gate reached"):
        cal._capture_one_profile_unattended(
            Driver(), tmp_path, ordinal=7, frame_counter=0, frames_meta=[],
            used_profile_ids=set())

    assert len(calls) == 1
    assert calls[0]["ordinal"] == 7
    assert calls[0]["identity_band"] == _IDENTITY_BAND
    assert calls[0]["content_band"] == _CONTENT_BAND
    assert calls[0]["like_threshold"] == cal.hinge_mod._LIKE_MATCH_THRESHOLD


def test_read_scroll_capture_requires_four_consecutive_quiet_post_gesture_frames(monkeypatch):
    """The navigation anchor is not trusted until four adjacent probes are quiet."""
    events = []
    frames = iter((b"moving", b"quiet-once", b"quiet-twice", b"quiet-thrice", b"settled"))
    driver = SimpleNamespace(
        dwell_s=1.25,
        content_band=_CONTENT_BAND,
        adb=SimpleNamespace(screencap=lambda: events.append("capture") or next(frames)),
    )
    monkeypatch.setattr(cal, "human_delay",
                        lambda dwell: events.append(("delay", dwell)) or 0.75)
    monkeypatch.setattr(cal, "time", SimpleNamespace(
        sleep=lambda delay: events.append(("sleep", delay))))
    quiet_deltas = iter((0, 0, 0, 0))
    monkeypatch.setattr(cal, "estimate_shift", lambda *_a, **_kw: SimpleNamespace(
        ok=True, delta_px=next(quiet_deltas), reason="parked"))

    assert cal._settled_read_scroll_frame(driver) == b"settled"
    assert events == [
        ("delay", 1.25), ("sleep", 0.75), "capture",
        ("delay", 1.25), ("sleep", 0.75), "capture",
        ("delay", 1.25), ("sleep", 0.75), "capture",
        ("delay", 1.25), ("sleep", 0.75), "capture",
        ("delay", 1.25), ("sleep", 0.75), "capture",
    ]

    # Both enumeration loops must use the same seam; otherwise one mode can keep recording
    # in-flight card positions after the other is fixed.
    assert "_settled_read_scroll_frame(driver)" in inspect.getsource(cal._capture_one_profile)
    assert "_settled_read_scroll_frame(driver)" in inspect.getsource(
        cal._capture_one_profile_unattended)


def test_read_scroll_settle_ignores_two_quiet_probes_before_late_motion(monkeypatch):
    """A transient quiet run must not become the anchor when Hinge resumes moving.

    Held-out 10.1.0 actually paused for two quiet comparisons, then made a 556px card advance.
    Only four quiet comparisons *after* that motion form a safe enumeration/navigation
    reference.
    """
    frames = iter((b"initial", b"quiet-before-drift-1", b"quiet-before-drift-2",
                   b"late-motion", b"quiet-after-drift-1", b"quiet-after-drift-2",
                   b"quiet-after-drift-3", b"final"))
    comparisons = []
    deltas = iter((0, 0, -556, 0, 0, 0, 0))
    driver = SimpleNamespace(
        dwell_s=0.0, content_band=_CONTENT_BAND,
        adb=SimpleNamespace(screencap=lambda: next(frames)),
    )
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda _delay: None))

    def measured_shift(prior, current, **_kwargs):
        comparisons.append((prior, current))
        delta = next(deltas)
        return SimpleNamespace(ok=True, delta_px=delta, reason=f"shift {delta}px")

    monkeypatch.setattr(cal, "estimate_shift", measured_shift)

    assert cal._settled_read_scroll_frame(driver) == b"final"
    assert comparisons == [
        (b"initial", b"quiet-before-drift-1"),
        (b"quiet-before-drift-1", b"quiet-before-drift-2"),
        (b"quiet-before-drift-2", b"late-motion"),
        (b"late-motion", b"quiet-after-drift-1"),
        (b"quiet-after-drift-1", b"quiet-after-drift-2"),
        (b"quiet-after-drift-2", b"quiet-after-drift-3"),
        (b"quiet-after-drift-3", b"final"),
    ]


def test_read_scroll_settle_refuses_motion_past_the_bounded_probe_budget(monkeypatch):
    driver = SimpleNamespace(
        dwell_s=0.0, content_band=_CONTENT_BAND,
        adb=SimpleNamespace(screencap=lambda: object()),
    )
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda _delay: None))
    monkeypatch.setattr(cal, "estimate_shift", lambda *_a, **_kw: SimpleNamespace(
        ok=True, delta_px=571, reason="still moving 571px"))

    with pytest.raises(cal._CaptureAbort, match="did not park.*8-comparison"):
        cal._settled_read_scroll_frame(driver)


def _unattended_single_item_fixtures(monkeypatch, *, extra_screencaps: int = 0):
    """Wire every seam a one-photo, ordinal=1 automated profile touches through to its terminal
    advance, without a real card scan/navigation stack.

    `extra_screencaps` prepends that many filler framebuffer reads, for callers that make the
    pre-heart loop take a corrective gesture before it reaches the composer -- each visual
    adjustment re-reads the screen, exactly as the reviewer-directed path does.

    `_automated_pass_from_verified_composer`/`_automated_send_from_verified_composer` are spied
    rather than exercised for real here: their own exact trace shapes are covered directly by
    `test_calibration_only_pass_allows_composer_offscreen_after_edge_back_then_uses_guarded_transport`
    and `test_automated_send_from_verified_composer_uses_the_production_send_transport`. This
    fixture is only about which ONE of them `_capture_one_profile_unattended`'s terminal dispatch
    calls, and what it tells a hybrid reviewer beforehand.
    """
    identity = ProfileIdentity((1,), _IDENTITY_BAND, (64, 16), 0, 20.0, "known", agreeing_frames=2)
    index = SimpleNamespace(usable=True, translation=(1,), complete=False)
    payload = SimpleNamespace(item=lambda _n: SimpleNamespace(heart_ordinal=1))
    target = SimpleNamespace(frame=b"target-pre", point=(500, 800),
                             block_frame_rows=(700, 900))

    class _Adb:
        def __init__(self):
            self.frames = iter([*(b"adjustment-%d" % i for i in range(extra_screencaps)),
                                b"composer-1", b"pass-frame", b"advance-identity"])
            self.last = None

        def screencap(self):
            # A settled phone returns the SAME bytes when asked again; it does not run out of
            # screen. `_settled_read_scroll_frame` reads until four consecutive comparisons are
            # quiet, so a fake that raised StopIteration on the second read was modelling
            # something no device does (2026-09-04).
            self.last = next(self.frames, self.last)
            return self.last

    class _Driver:
        identity_band = _IDENTITY_BAND
        content_band = _CONTENT_BAND
        dwell_s = 0.0

        def __init__(self):
            self.adb = _Adb()
            self.taps = []
            self.scrolls = []

        def _template(self, _name):
            return object()

        def _tap(self, *point):
            self.taps.append(point)

        def _scroll_down_one(self, frac, x_frac):
            self.scrolls.append(("down", frac, x_frac))

        def _scroll_up_one(self, frac, x_frac):
            self.scrolls.append(("up", frac, x_frac))

    driver = _Driver()
    pass_calls: list = []
    send_calls: list = []

    def fake_pass(*_a, **_kw):
        pass_calls.append(1)
        return b"advance-frame", {"action": "automated_pass", "send_like_tapped": False,
                                  "composer_clear_visible": True}

    def fake_send(*_a, **_kw):
        send_calls.append(1)
        return b"advance-frame", {
            "action": "automated_send_priority_like",
            "transport": ["HingeDriver._tap(confirm_point)", "HingeDriver._handle_rose_upsell",
                          "HingeDriver._verify_like_landed"],
            "send_like_tapped": True, "confirm_point": [1, 2],
            "predicates": {"inline_composer_and_selected_photo_verified_before_action": True,
                          "send_like_tapped": True, "like_landed_verified": True},
        }

    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_a: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "_rewind_automated_profile_to_confirmed_top", lambda *_a, **_kw: b"top")
    monkeypatch.setattr(cal, "capture_profile_identity", lambda *_a, **_kw: identity)
    monkeypatch.setattr(cal, "build_item_index", lambda *_a, **_kw: index)
    monkeypatch.setattr(cal, "build_item_payload", lambda *_a, **_kw: payload)
    monkeypatch.setattr(cal, "_target_scoped_prefix_reason", lambda *_a, **_kw: None)
    monkeypatch.setattr(cal, "verification_blocker", lambda *_a, **_kw: None)
    monkeypatch.setattr(cal, "navigate_to_item", lambda *_a, **_kw: target)
    # The pre-heart checkpoint reads its mute-control predicate out of this proof, so the stub
    # has to supply real evidence bound to the exact target frame -- an opaque object would be
    # accepted only by an implementation that restates the verdict as a literal.
    monkeypatch.setattr(
        cal, "_verified_target_frame_proof",
        lambda *_a, **_kw: cal._TargetFrameProof(
            block=SimpleNamespace(x0=53, y0=700, x1=1027, y1=900, hearts=(target.point,)),
            frame_sha256=cal._sha256(target.frame), mute_control_absent=True,
            heart_visible=True))
    # The still-photo proof takes a real no-input dwell burst off a real device; its own
    # producer/reader contract is covered directly below.
    monkeypatch.setattr(
        cal, "_verified_still_photo_proof",
        lambda *_a, **_kw: cal._StillPhotoProof(
            action_frame_sha256=cal._sha256(target.frame),
            pre_probe_frame_sha256=cal._sha256(target.frame),
            dwell_frame_sha256s=(cal._sha256(target.frame), cal._sha256(b"dwell")),
            dwell_span_s=6.0, still_photo_verified=True,
            reattach_frame_sha256s=(cal._sha256(target.frame), cal._sha256(b"reattach")),
            reattach_dwell_span_s=6.0,
            # The byte-identical fast path: `action_frame` must actually BE `target.frame`, not
            # merely name it by digest, now that the pre-heart loop gates its rebind on the bytes
            # (`action_frame != target_pre`) rather than on the residual alone.
            page_residual_px=0, action_frame=target.frame))
    monkeypatch.setattr(
        cal, "_fresh_reviewed_target_point",
        lambda _driver, target, **_kw: target.point)
    monkeypatch.setattr(cal, "locate_inline_composer",
                        lambda *_a, **_kw: SimpleNamespace(
                            layout_id=cal._COMPOSER_LAYOUT_ID, confirm_point=(1, 2)))
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_a, **_kw: SimpleNamespace(matched=True, reason="matched"))
    monkeypatch.setattr(cal, "_automated_pass_from_verified_composer", fake_pass)
    monkeypatch.setattr(cal, "_automated_send_from_verified_composer", fake_send)
    monkeypatch.setattr(cal, "_plan_card_scroll",
                        lambda *_a, **_kw: (SimpleNamespace(frac=0.1, x_frac=0.5), None))
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_a, **_kw: SimpleNamespace(confirmed=False, refuted=True, reason="sticky"))
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_a, **_kw: (999,))
    monkeypatch.setattr(cal, "fingerprint_distance",
                        lambda *_a, **_kw: cal._IDENTITY_FALSE_MATCH_DISTANCE + 1)
    return driver, pass_calls, send_calls


def test_capture_time_numbering_never_asks_the_payload_for_a_dwell_it_cannot_have(
        monkeypatch, tmp_path):
    """THE EXACT REGRESSION THAT SHIPPED (e5054f7f), pinned at the call shape.

    The automated enumeration loop wired the strict still-photo gate into its payload build and
    handed it `still_photo_dwell=None`.  The dwell rung can never pass on a None dwell, and a
    read scroll separates every candidate in this loop from its next frame, so there is no
    un-interacted dwell to hand it either: every selectable block was refused, every payload came
    back unusable, and every profile skipped with "photo-only payload is unusable" (2026-08-22,
    two of two real profiles).  Numbering HERE only selects which item to navigate to; the heart
    is licensed later by `_verified_still_photo_proof` on real parked evidence, so passing those
    two kwargs is never right at this call site regardless of what is passed to them.
    """
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    stub = cal.build_item_payload
    recorded: list[dict] = []

    def recorder(*args, **kwargs):
        recorded.append(kwargs)
        return stub(*args, **kwargs)

    monkeypatch.setattr(cal, "build_item_payload", recorder)

    cal._capture_one_profile_unattended(
        driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
        review_gate=None, send_like=False)

    assert recorded, "the automated enumeration loop must have built a payload at all"
    for kwargs in recorded:
        assert "unnumber_without_evidence" not in kwargs
        assert "still_photo_dwell" not in kwargs
        assert kwargs["unnumber"] is cal.unnumber_unless_confident_photo


def test_an_all_exception_enumeration_names_the_exception_in_its_skip_detail(
        monkeypatch, tmp_path):
    """A run where EVERY iteration raised used to report "index not yet built" -- the placeholder
    describing the state before the loop started, not the failure the operator has to fix.  The
    handler now reassigns the reason, so the skip record names the class that actually raised."""
    class _Adb:
        def screencap(self):
            return b"card-frame"

    class _Driver:
        identity_band = _IDENTITY_BAND
        content_band = _CONTENT_BAND
        dwell_s = 0.0

        def __init__(self):
            self.adb = _Adb()
            self.scrolls = []

        def _template(self, _name):
            return object()

        def _scroll_down_one(self, frac, x_frac):
            self.scrolls.append((frac, x_frac))

    identity = ProfileIdentity((1,), _IDENTITY_BAND, (64, 16), 0, 20.0, "known", agreeing_frames=2)

    def always_raises(*_a, **_kw):
        raise cal.ItemIndexError("segmentation disagreed with itself on every frame set")

    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_a: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "_rewind_automated_profile_to_confirmed_top", lambda *_a, **_kw: b"top")
    monkeypatch.setattr(cal, "capture_profile_identity", lambda *_a, **_kw: identity)
    monkeypatch.setattr(cal, "build_item_index", always_raises)
    monkeypatch.setattr(cal, "_plan_card_scroll",
                        lambda *_a, **_kw: (SimpleNamespace(frac=0.1, x_frac=0.5), None))
    monkeypatch.setattr(cal, "_skip_automated_profile_before_heart",
                        lambda _driver, **kwargs: {"reason_code": kwargs["reason"].code,
                                                   "reason_detail": kwargs["reason"].detail})

    with pytest.raises(cal._ProfileSkipped) as excinfo:
        cal._capture_one_profile_unattended(
            _Driver(), tmp_path, ordinal=1, frame_counter=0, frames_meta=[],
            used_profile_ids=set(), review_gate=None, send_like=False)

    detail = excinfo.value.record["reason_detail"]
    assert "ItemIndexError" in detail
    assert "index not yet built" not in detail


def test_pre_action_skip_meeting_the_unsupported_layout_escalates_instead_of_killing_the_session(
        monkeypatch, tmp_path):
    """The pre-action skip's own rewind can reach the same unsupported entry layout the entry
    rewind does -- from deeper in the card, so it crosses more intermediate offsets.  Because
    `_UnsupportedEntryDeck` subclasses `_CaptureAbort`, letting it propagate reached
    `_cmd_capture`'s abort handler and marked the whole session `interrupted`, which both
    `_load_session` and the offline reviewer refuse: every profile already banked in that run was
    discarded over one advanceable deck.  It must escalate exactly like the entry rewind does,
    carrying the diagnosis that asked for the skip."""
    class _Adb:
        def screencap(self):
            return b"card-frame"

    class _Driver:
        identity_band = _IDENTITY_BAND
        content_band = _CONTENT_BAND
        dwell_s = 0.0

        def __init__(self):
            self.adb = _Adb()

        def _template(self, _name):
            return object()

        def _scroll_down_one(self, _frac, _x_frac):
            pass

    identity = ProfileIdentity((1,), _IDENTITY_BAND, (64, 16), 0, 20.0, "known", agreeing_frames=2)
    escalations: list[dict] = []

    def unsupported_rewind(*_a, **_kw):
        raise cal._UnsupportedEntryDeck(
            b"skip-path-dead-zone", state="cannot_tell", reason="chip row out of the top band")

    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_a: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "_rewind_automated_profile_to_confirmed_top", lambda *_a, **_kw: b"top")
    monkeypatch.setattr(cal, "capture_profile_identity", lambda *_a, **_kw: identity)
    monkeypatch.setattr(cal, "build_item_index",
                        lambda *_a, **_kw: (_ for _ in ()).throw(cal.ItemIndexError("no index")))
    monkeypatch.setattr(cal, "_plan_card_scroll",
                        lambda *_a, **_kw: (SimpleNamespace(frac=0.1, x_frac=0.5), None))
    monkeypatch.setattr(cal, "_skip_automated_profile_before_heart", unsupported_rewind)
    monkeypatch.setattr(
        cal, "_skip_automated_unsupported_entry_deck",
        lambda _driver, **kwargs: escalations.append(kwargs) or {
            "reason_code": "unsupported_entry_layout", "ordinal": kwargs["ordinal"]})

    with pytest.raises(cal._ProfileSkipped) as excinfo:
        cal._capture_one_profile_unattended(
            _Driver(), tmp_path, ordinal=4, frame_counter=0, frames_meta=[],
            used_profile_ids=set(), review_gate=None, send_like=True)

    assert excinfo.value.record["reason_code"] == "unsupported_entry_layout"
    (escalated,), = (escalations,)
    assert escalated["entry"].frame == b"skip-path-dead-zone"
    assert escalated["ordinal"] == 4
    assert escalated["send_like"] is True
    assert escalated["skip_reason"].code == "target_unavailable_or_incomplete_index"


def test_default_unattended_capture_still_terminates_with_pass(monkeypatch, tmp_path):
    """Regression guard on the never-send default: with `send_like` left False (no CLI flag),
    the terminal advance must go through the Pass helper, never the Send helper."""
    driver, pass_calls, send_calls = _unattended_single_item_fixtures(monkeypatch)

    meta, _counter = cal._capture_one_profile_unattended(
        driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
        review_gate=None, send_like=False)

    assert pass_calls == [1]
    assert send_calls == []
    terminal = next(a for a in meta["automated_actions"]
                    if a["action"] in {"automated_pass", "automated_send_priority_like"})
    assert terminal["action"] == "automated_pass"
    assert terminal["send_like_tapped"] is False


def test_automated_capture_commits_the_final_identity_reprobe_frame(monkeypatch, tmp_path):
    """The accepted late frame, rather than the colliding initial sticky frame, is evidence."""
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    driver.adb.frames = iter((b"composer-1", b"pass-frame", b"transitional", b"late-1", b"late-2"))
    monkeypatch.setattr(cal, "_scroll_next_profile_to_sticky_header",
                        lambda *_a, **_kw: b"initial-collision")
    _wire_post_advance_reprobe_measurement(
        monkeypatch,
        fingerprints={b"initial-collision": (0,), b"transitional": (0,), b"late-1": (10,),
                      b"late-2": (10,)})

    frames_meta = []
    meta, _counter = cal._capture_one_profile_unattended(
        driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=frames_meta,
        used_profile_ids=set(), review_gate=None, send_like=False)

    identity_record = next(record for record in frames_meta
                           if record["role"] == "profile_advance_identity")
    assert (tmp_path / identity_record["file"]).read_bytes() == b"late-2"
    proof = next(action for action in meta["automated_actions"]
                 if action["action"] == "automated_sticky_header_proof")
    # The fixture's pre-heart identity is ``(1,)``; the final sampled fingerprint is ``(10,)``.
    assert proof["new_profile_identity_distance"] == 9.0
    assert proof["accepted_read_only_reprobe_pair"] == [2, 3]


@pytest.mark.parametrize(
    ("send_like", "expected_action", "expected_source"),
    [
        (False, "automated_pass", "calibration-only verified-composer Pass transport"),
        (True, "automated_send_priority_like", "calibration-only verified-composer Send transport"),
    ],
    ids=["default-pass", "accepted-send"],
)
def test_hybrid_pre_action_checkpoint_action_matches_the_accepted_terminal_transport(
        monkeypatch, tmp_path, send_like, expected_action, expected_source):
    """The checkpoint a hybrid reviewer approves before the terminal action must plainly say
    which transport is about to run -- never disguise an accepted real Send as an ordinary Pass,
    or vice versa -- and `_capture_one_profile_unattended` must actually call that one helper."""
    driver, pass_calls, send_calls = _unattended_single_item_fixtures(monkeypatch)
    checkpoints = []

    class _Gate:
        def checkpoint(self, _frame, *, claimed_state, action_plan):
            checkpoints.append((claimed_state, action_plan))
            return {"decision": "approved", "action_plan": action_plan}

    meta, _counter = cal._capture_one_profile_unattended(
        driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
        review_gate=_Gate(), send_like=send_like)

    terminal = next(a for a in meta["automated_actions"]
                    if a["action"] in {"automated_pass", "automated_send_priority_like"})
    assert terminal["action"] == expected_action
    assert (pass_calls, send_calls) == (([1], []) if not send_like else ([], [1]))

    heart_plan = next(plan for state, plan in checkpoints if state == "target_heart_visible")
    assert heart_plan["predicates"]["positive_still_photo_evidence_verified"] is True
    assert heart_plan["predicates"]["target_frame_mute_control_screened_absent"] is True
    assert heart_plan["predicates"]["verification_blocker_absent"] is True
    assert heart_plan["predicates"]["target_heart_visible"] is True
    assert "photo_only_item_verified" not in heart_plan["predicates"]

    pre_action_plan = next(plan for state, plan in checkpoints if state == "composer_open_before_pass")
    assert pre_action_plan["action"] == expected_action
    assert pre_action_plan["point_source"] == expected_source
    # The checkpoint describes the plan, not an already-completed action: nothing has been
    # tapped yet at the moment the reviewer sees it, regardless of which transport is planned.
    assert pre_action_plan["predicates"]["send_like_tapped"] is False


def test_terminal_send_refusal_cleans_the_current_unsent_composer(monkeypatch, tmp_path):
    """A fresh-frame Send refusal happens after the real heart opened a composer.

    It must invoke the Pass-only abort recovery on a newly captured current frame; otherwise the
    next calibration session starts stranded in that composer.  The refusing Send is not retried
    and the profile remains excluded from evidence.
    """
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    recoveries = []

    def refuse_send(*_args, **_kwargs):
        raise cal._CaptureAbort("fresh composer proof changed before tap")

    def recover(_driver, **kwargs):
        recoveries.append(kwargs)
        return {"outcome": "cleared", "send_like_tapped": False}

    monkeypatch.setattr(cal, "_automated_send_from_verified_composer", refuse_send)
    monkeypatch.setattr(cal, "_recover_automated_abort_from_open_composer", recover)

    class _Gate:
        def checkpoint(self, _frame, *, claimed_state, action_plan):
            return {"decision": "approved", "action_plan": action_plan}

    with pytest.raises(cal._CaptureAbort, match=r"terminal Send refused.*cleanup outcome=cleared"):
        cal._capture_one_profile_unattended(
            driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[],
            used_profile_ids=set(), review_gate=_Gate(), abort_recoveries=[], send_like=True)

    assert len(recoveries) == 1
    assert recoveries[0]["frame"] == b"advance-identity"
    assert recoveries[0]["failure_stage"] == "terminal_send_refused"


def test_approved_hybrid_decision_does_not_require_a_kind_field():
    """Finding-2 pin (2026-09-02): every real on-disk hybrid_review ledger record (checked
    across multiple `manifest.json` files under ops/calibration/) carries checkpoint_file/
    checkpoint_sha256/checkpoint_evidence_sha256/frame_file/frame_sha256/claimed_state/
    action_plan/decision/source/reviewer/human_ground_truth/decided_utc -- and NO "kind" key.
    `_approved_hybrid_decision` must keep validating a record shaped exactly like that, on
    every field it actually checks, without ever gating on a "kind" nothing has ever written.
    A `_HYBRID_REVIEW_TOKEN_KIND` constant sat unread beside this function for exactly that
    reason; see the comment beside `_HYBRID_CHECKPOINT_KIND` for why it was deleted rather than
    wired into an enforced check that would reject every one of those historical sessions."""
    record = {
        "checkpoint_file": "/tmp/x/00001.checkpoint.json", "checkpoint_sha256": "a" * 64,
        "checkpoint_evidence_sha256": "b" * 64,
        "frame_file": "/tmp/x/00001.png", "frame_sha256": "c" * 64,
        "claimed_state": "composer_open_before_pass",
        "action_plan": {"action": "automated_pass", "photo_model_item": None},
        "decision": "approved", "source": "external_ai_review",
        "reviewer": {"name": "terra", "model": "gpt-5.6-terra"},
        "human_ground_truth": False, "decided_utc": "2026-08-14T00:00:00+00:00",
    }
    assert "kind" not in record   # exactly the shape every real on-disk record has
    acceptance = {"reviewer_source": "external_ai_review",
                  "reviewer": {"name": "terra", "model": "gpt-5.6-terra"}}
    review = {"decisions": [record]}

    assert cal._approved_hybrid_decision(
        record, acceptance=acceptance, review=review,
        action="automated_pass", item=None) is True


def test_hybrid_review_token_kind_constant_was_deliberately_removed():
    """See the comment beside `_HYBRID_CHECKPOINT_KIND`: a `_HYBRID_REVIEW_TOKEN_KIND` constant
    was found unread (2026-09-02) and deleted rather than enforced, because real on-disk ledger
    records never carried the field it would have checked for. Re-adding it without a migration
    for the existing decisions is the regression this guards against."""
    assert not hasattr(cal, "_HYBRID_REVIEW_TOKEN_KIND")


class _RecordingGate:
    """A hybrid gate that approves everything and remembers exactly what it was shown."""

    def __init__(self):
        self.checkpoints = []

    def checkpoint(self, _frame, *, claimed_state, action_plan):
        self.checkpoints.append((claimed_state, action_plan))
        return {"decision": "approved", "action_plan": action_plan}


def _mute_proof(*, frame=b"target-pre", mute_control_absent=True, heart_visible=True):
    """One exact-frame target screen result, as `_verified_target_frame_proof` would return it."""
    return cal._TargetFrameProof(
        block=SimpleNamespace(x0=53, y0=700, x1=1027, y1=900, hearts=((500, 800),)),
        frame_sha256=cal._sha256(frame), mute_control_absent=mute_control_absent,
        heart_visible=heart_visible)


def test_pre_heart_checkpoint_publishes_the_mute_verdict_of_the_screen_that_ran(
        monkeypatch, tmp_path):
    """`target_frame_mute_control_screened_absent` may say True only because a mute screen of
    THESE exact bytes said so.  Pair with the refusal cases below: together they fail for any
    implementation that restates the verdict as a literal beside the call that proves it."""
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    monkeypatch.setattr(cal, "_verified_target_frame_proof", lambda *_a, **_kw: _mute_proof())
    gate = _RecordingGate()

    cal._capture_one_profile_unattended(
        driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
        review_gate=gate, send_like=False)

    heart_plan = next(plan for state, plan in gate.checkpoints if state == "target_heart_visible")
    assert heart_plan["predicates"]["target_frame_mute_control_screened_absent"] is True


@pytest.mark.parametrize(
    ("build_proof", "expected"),
    [
        (lambda: _mute_proof(mute_control_absent=False), "found a visible mute control"),
        (lambda: _mute_proof(frame=b"a-different-frame"),
         "no target-frame mute-control screen is bound"),
    ],
    ids=["screen-saw-a-mute-control", "screen-proved-some-other-frame"],
)
def test_pre_heart_checkpoint_refuses_instead_of_claiming_an_unscreened_frame(
        monkeypatch, tmp_path, build_proof, expected):
    """The two ways the claim can come loose from its evidence: the screen returns a negative
    verdict, and the screen ran somewhere else (what a refactor that moves or wraps the call
    actually produces).  Both must stop the run before a reviewer is ever offered the heart --
    an approval on an unscreened frame is how a real video got within one keystroke of a like."""
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    monkeypatch.setattr(cal, "_verified_target_frame_proof", lambda *_a, **_kw: build_proof())
    gate = _RecordingGate()

    with pytest.raises(cal._CaptureAbort, match=expected):
        cal._capture_one_profile_unattended(
            driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
            review_gate=gate, send_like=False)

    assert [state for state, _plan in gate.checkpoints if state == "target_heart_visible"] == []
    assert driver.taps == []


def test_real_target_frame_screen_seeing_a_mute_control_never_reaches_the_checkpoint(
        monkeypatch, tmp_path):
    """End-to-end through the production screen rather than a stubbed proof: a driver that
    reports Hinge's mute control on the exact action frame skips the profile before any
    checkpoint or tap, so no predicate about that frame is ever published."""
    real_proof = cal._verified_target_frame_proof
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    monkeypatch.setattr(cal, "_verified_target_frame_proof", real_proof)
    monkeypatch.setattr(
        cal, "segment_frame",
        lambda *_a, **_kw: SimpleNamespace(
            ok=True, failures=(),
            blocks=(SimpleNamespace(x0=53, y0=700, x1=1027, y1=900, hearts=((500, 800),)),)))
    monkeypatch.setattr(cal, "_skip_automated_profile_before_heart",
                        lambda *_a, **_kw: {"reason_code": "target_verification_blocked"})
    driver._target_frame_video_screen_reason = lambda *_a: (
        "target frame contains Hinge's mute control (score 0.999999); "
        "the selected media is a video")
    gate = _RecordingGate()

    with pytest.raises(cal._ProfileSkipped, match="target_verification_blocked"):
        cal._capture_one_profile_unattended(
            driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
            review_gate=gate, send_like=False)

    assert gate.checkpoints == []
    assert driver.taps == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda m: m.setattr(
            cal, "_verified_still_photo_proof",
            lambda *_a, **_kw: cal._StillPhotoProof(
                action_frame_sha256=cal._sha256(b"some-other-frame"),
                pre_probe_frame_sha256=cal._sha256(b"some-other-frame"),
                dwell_frame_sha256s=("a", "b"),
                dwell_span_s=6.0, still_photo_verified=True)),
         "no still-photo .C1-C3. verdict is bound"),
        (lambda m: m.setattr(
            cal, "_verified_still_photo_proof",
            lambda *_a, **_kw: cal._StillPhotoProof(
                action_frame_sha256=cal._sha256(b"target-pre"),
                pre_probe_frame_sha256=cal._sha256(b"target-pre"),
                dwell_frame_sha256s=("a", "b"),
                dwell_span_s=6.0, still_photo_verified=False,
                # Must actually BE `target-pre`'s bytes, not merely name them by digest, so the
                # loop's bytes-gated rebind stays a no-op and this case still isolates the
                # negative verdict rather than tripping the (correct, but different) binding
                # refusal first.
                page_residual_px=0, action_frame=b"target-pre")),
         "still-photo acceptance did not pass"),
        (lambda m: m.setattr(cal, "_verified_target_frame_proof",
                             lambda *_a, **_kw: _mute_proof(heart_visible=False)),
         "target-heart location did not find"),
        (lambda m: m.setattr(
            cal, "_verified_blocker_absence",
            lambda payload, item_number: cal._VerificationBlockerProof(
                item_number=item_number + 1, heart_ordinal=1, blocker="", blocker_absent=True)),
         "no verification-blocker screen is bound"),
        (lambda m: m.setattr(
            cal, "_verified_blocker_absence",
            lambda payload, item_number: cal._VerificationBlockerProof(
                item_number=item_number, heart_ordinal=99, blocker="", blocker_absent=True)),
         "no verification-blocker screen is bound"),
    ],
    ids=["still-photo-proved-another-frame", "still-photo-verdict-negative",
         "heart-not-located", "blocker-screened-another-item", "blocker-screened-another-heart"],
)
def test_every_pre_heart_predicate_refuses_rather_than_restating_an_unearned_true(
        monkeypatch, tmp_path, mutate, expected):
    """Mutation coverage for the whole predicate block.  Each case breaks ONE underlying check or
    its binding and requires the run to stop before a reviewer is offered the heart.  Together
    they fail for any implementation that publishes a literal beside the call that proves it --
    which is precisely how `photo_only_item_verified` shipped as a hardcoded True."""
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    mutate(monkeypatch)
    gate = _RecordingGate()

    with pytest.raises(cal._CaptureAbort, match=expected):
        cal._capture_one_profile_unattended(
            driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
            review_gate=gate, send_like=False)

    assert [state for state, _plan in gate.checkpoints if state == "target_heart_visible"] == []
    assert driver.taps == []


def test_a_blocked_item_is_skipped_by_the_screen_itself_not_by_a_restated_literal(
        monkeypatch, tmp_path):
    """The positive half of the blocker predicate: the screen's own verdict decides both the
    skip and the published claim, so the two can never disagree."""
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    monkeypatch.setattr(cal, "verification_blocker",
                        lambda *_a, **_kw: "item 1 has no separable reference")
    monkeypatch.setattr(cal, "_skip_automated_profile_before_heart",
                        lambda *_a, **_kw: {"reason_code": "target_verification_blocked"})
    gate = _RecordingGate()

    with pytest.raises(cal._ProfileSkipped, match="target_verification_blocked"):
        cal._capture_one_profile_unattended(
            driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
            review_gate=gate, send_like=False)

    assert gate.checkpoints == []
    assert driver.taps == []


def test_navigation_refusal_forensics_do_not_pollute_the_manifest_root(monkeypatch, tmp_path):
    """A diagnostic pair is useful after a refusal but cannot be an unmanifested root PNG.

    `_load_session` intentionally authenticates every root image against `manifest.frames`.
    Keep navigation-refusal diagnostics under `forensics/`, where they remain private and useful
    without making a later completed capture fail its exact root-frame inventory.
    """
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    anchor = SimpleNamespace(delta_px=-556, status="measured", confidence=1.0,
                             reason="late card snap")

    def refuse_navigation(*_args, **_kwargs):
        raise cal.ItemNavigationError("entry_anchor_unmeasured", "late card snap",
                                      frame=b"entry-frame", anchor=anchor)

    monkeypatch.setattr(cal, "navigate_to_item", refuse_navigation)
    monkeypatch.setattr(
        cal, "_skip_automated_profile_before_heart",
        lambda _driver, **kwargs: {"reason_code": kwargs["reason"].code})

    with pytest.raises(cal._ProfileSkipped, match="pre_heart_navigation_refused"):
        cal._capture_one_profile_unattended(
            driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
            review_gate=None, send_like=False, entry_drift_restart_attempts=1)

    forensics = tmp_path / "forensics"
    stem = "refused_navigation_p1_item1_scan2"
    assert (forensics / f"{stem}_read_reference.png").read_bytes() == b"top"
    assert (forensics / f"{stem}_entry.png").read_bytes() == b"entry-frame"
    body = json.loads((forensics / f"{stem}.json").read_text())
    assert body["calibration_evidence"] is False
    assert body["anchor_delta_px"] == -556
    assert list(tmp_path.glob("*.png")) == []


def test_large_measured_entry_drift_restarts_once_then_uses_the_normal_preaction_skip(
        monkeypatch, tmp_path):
    """Never rebase a changed entry frame onto an old enumeration/index.

    The first measured large drift discards the local scan before a heart, reviewer label, or
    public Like.  The outer capture loop will re-enumerate the same profile.  If it recurs on
    that fresh scan, the established bounded pre-action skip remains the only advance path.
    """
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    anchor = SimpleNamespace(delta_px=-547, status="measured", confidence=1.0,
                             reason="late Hinge snap")

    def refuse_navigation(*_args, **_kwargs):
        raise cal.ItemNavigationError(cal.NAV_ANCHOR_UNMEASURED, "large entry drift",
                                      frame=b"entry-frame", anchor=anchor)

    skip_calls = []
    monkeypatch.setattr(cal, "navigate_to_item", refuse_navigation)
    monkeypatch.setattr(
        cal, "_skip_automated_profile_before_heart",
        lambda _driver, **kwargs: skip_calls.append(kwargs) or {
            "reason_code": kwargs["reason"].code})

    with pytest.raises(cal._RestartProfile, match="discarding the scan"):
        cal._capture_one_profile_unattended(
            driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
            review_gate=None, send_like=False, entry_drift_restart_attempts=0)
    assert skip_calls == []
    assert driver.taps == []

    with pytest.raises(cal._ProfileSkipped, match="pre_heart_navigation_refused"):
        cal._capture_one_profile_unattended(
            driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
            review_gate=None, send_like=False, entry_drift_restart_attempts=1)
    assert len(skip_calls) == 1
    assert skip_calls[0]["reason"].code == "pre_heart_navigation_refused"
    assert driver.taps == []
    # Both exact frame pairs survive: the second recurrent refusal must not overwrite the first
    # discarded scan's diagnostic merely because profile ordinal/item number are unchanged.
    for scan in (1, 2):
        stem = f"refused_navigation_p1_item1_scan{scan}"
        assert (tmp_path / "forensics" / f"{stem}_read_reference.png").read_bytes() == b"top"
        assert (tmp_path / "forensics" / f"{stem}_entry.png").read_bytes() == b"entry-frame"


_REFUSED_COMPOSER_ABORT_MESSAGE = (
    "automated profile 1 item 1: post-tap composer/item verification refused: drift; "
    "abort cleanup outcome=cleared")


def test_post_tap_refusal_persists_the_refusing_composer_frame_as_a_forensic_diagnostic(
        monkeypatch, tmp_path):
    """LIVE 2026-08-22: Hinge 10.0.1 refused post-tap composer/item verification (hiding 34.08%
    of a 1109px stored crop against the 27.86% `_INLINE_REFRAME_MAX_HIDDEN_FRACTION` limit
    measured on 9.134), and the refusing composer frame was discarded together with the abort --
    nothing wrote it to disk, so the new composer geometry could never be re-measured from the
    session afterward.  This pins that the frame and a sibling refusal record now survive the
    abort as a clearly-forensic, never-evidence diagnostic, without changing the refusal itself
    (identical message, identical cleanup outcome)."""
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_a, **_kw: SimpleNamespace(matched=False, reason="drift"))

    with pytest.raises(cal._CaptureAbort) as excinfo:
        cal._capture_one_profile_unattended(
            driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
            review_gate=None, send_like=False)

    assert str(excinfo.value) == _REFUSED_COMPOSER_ABORT_MESSAGE

    png_path = tmp_path / "refused_composer_p1_item1.png"
    json_path = tmp_path / "refused_composer_p1_item1.json"
    assert png_path.read_bytes() == b"composer-1"
    body = json.loads(json_path.read_text())
    assert body["ordinal"] == 1
    assert body["item_number"] == 1
    assert body["refusal"] == "drift"
    assert body["frame_sha256"] == cal._sha256(b"composer-1")
    # The fixture's `_Adb` has no `.shell()`/`.screen_size()` and its stub `payload` has no
    # `.height` -- both optional fields must be OMITTED on that failure, never raised through.
    assert "hinge_version_name" not in body
    assert "frame_size_px" not in body
    assert "stored_crop_height_px" not in body


def test_post_tap_refusal_diagnostic_frame_is_excluded_from_manifest_evidence(
        monkeypatch, tmp_path):
    """The diagnostic PNG this abort writes must never be mistaken for calibration evidence.
    Drive the refusal through the real `_cmd_capture` (not a direct unit call) so the manifest is
    actually assembled the production way -- `frames` only ever grows through
    `_save_frame`/`_commit_profile_frames` and `profiles` only ever grows after a profile
    completes -- and confirm the forensic file's name appears in neither, nor anywhere else in
    the written manifest.json, even though it sits on disk right beside it."""
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_a, **_kw: SimpleNamespace(matched=False, reason="drift"))
    # Extend the fixture's driver with just enough real shape for `_cmd_capture`'s own outer
    # shell and the module's read-only `_device_evidence` probe -- both exercised for real here.
    driver.serial = "PIXEL-TEST"
    driver.package = "co.hinge.app"
    driver.open_session = lambda: None
    driver.close = lambda: None
    driver.adb.shell = lambda cmd: (
        "Pixel 7a" if "ro.product.model" in cmd else
        "420" if "wm density" in cmd else "versionName=10.0.1\n")
    driver.adb.screen_size = lambda: (1080, 2400)

    monkeypatch.setattr(cal.cfg_mod, "load", lambda _path: SimpleNamespace(apps={}))
    monkeypatch.setattr(cal.cfg_mod, "validate", lambda _cfg: None)
    monkeypatch.setattr(cal, "HingeDriver", lambda _cfg: driver)
    monkeypatch.setattr(cal, "_preflight_serial", lambda _cfg: ("PIXEL-TEST", "adb"))
    monkeypatch.setattr(cal, "_capture_out_dir", lambda *_a, **_kw: tmp_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("apps:\n  hinge:\n    serial: PIXEL-TEST\n")
    args = argparse.Namespace(
        profiles=1, split="calibration", config=str(config_path), out=str(tmp_path),
        unattended=True, hybrid_review=False, confirmation=cal._UNATTENDED_CONFIRMATION,
        record_operational_checks=False, send_like=False, send_like_confirmation="")

    with pytest.raises(SystemExit):
        cal._cmd_capture(args)

    diagnostic_png = tmp_path / "refused_composer_p1_item1.png"
    assert diagnostic_png.exists()
    manifest_text = (tmp_path / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    assert manifest["interrupted"] is True
    assert manifest["frames"] == []
    assert manifest["profiles"] == []
    assert diagnostic_png.name not in [entry["file"] for entry in manifest["frames"]]
    assert diagnostic_png.name not in json.dumps(manifest["profiles"])
    assert diagnostic_png.name not in manifest_text


def test_device_evidence_shares_the_operational_evidence_probe():
    """Finding-3 pin (2026-09-02): `hinge_calibrate._device_evidence` and
    `hinge_operational_evidence._read_device_evidence` used to be two independently hand-rolled
    copies of the identical ~20-line "getprop/wm density/dumpsys package, then parse
    versionName" probe, and had already drifted (this module's copy was missing the density
    check the other one had). Assert IDENTITY, not just equal behaviour, so the two can never
    silently re-diverge onto separate implementations the way they did before."""
    from tools import hinge_operational_evidence as op_evidence

    assert cal._read_shared_device_evidence is op_evidence._read_device_evidence


def test_device_evidence_refuses_on_missing_density(monkeypatch):
    """Regression for the finding-3 drift: `_device_evidence` used to accept an empty `wm
    density` reading (no check at all), unlike the operational-evidence probe it now shares,
    which always refused one. A phone that cannot report its density is exactly the kind of
    unidentified-build evidence this module's own fail-loud rule exists to catch."""
    class _Adb:
        def shell(self, cmd):
            if "ro.product.model" in cmd:
                return "Pixel 7a\n"
            if "wm density" in cmd:
                return "   \n"   # blank after strip -- the exact case the fix now catches
            return "versionName=10.1.0\n"

        def screen_size(self):
            return (1080, 2400)

    driver = SimpleNamespace(adb=_Adb(), serial="PIXEL-TEST", package="co.hinge.app")

    with pytest.raises(RuntimeError, match="device/build evidence"):
        cal._device_evidence(driver)


def test_post_tap_refusal_diagnostic_write_failure_never_masks_the_real_abort(
        monkeypatch, tmp_path):
    """The diagnostic write is forensic best-effort only.  If the private-write helper itself
    raises (disk full, permission error, ...), the operator must still see the exact same
    refusal that actually stopped the run -- byte-identical to the case where the write
    succeeds -- never a diagnostic-tooling exception standing in its place, and never a silently
    swallowed abort either."""
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_a, **_kw: SimpleNamespace(matched=False, reason="drift"))
    monkeypatch.setattr(
        cal, "atomic_write_private_bytes",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(cal._CaptureAbort) as excinfo:
        cal._capture_one_profile_unattended(
            driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
            review_gate=None, send_like=False)

    assert str(excinfo.value) == _REFUSED_COMPOSER_ABORT_MESSAGE
    assert list(tmp_path.iterdir()) == []


# The pre-heart loop's centring arithmetic, in the same geometry the fixtures use: a 2000px
# frame with content band (0.125, 0.875) puts the content centre at row 1000 and makes the band
# 1500px tall, so a card's signed offset is (centre_row - 1000) / 1500 against a 0.150 limit.
_PARK_FRAME_HEIGHT_PX = 2000
_PARK_OFF_ZONE_ROWS = (557, 757)      # centre 657  -> -0.229, the live 2026-08-22 measurement
_PARK_IN_ZONE_ROWS = (900, 1100)      # centre 1000 ->  0.000, dead centre


def _parked_target(rows):
    """What `navigate_to_item` hands back after parking the card at `rows` on the action frame."""
    return SimpleNamespace(frame=b"target-pre", point=(500, 800), block_frame_rows=rows)


def _park_bound_proof(_driver, target, **_kw):
    """`_verified_target_frame_proof` as the real one behaves: the block it matched is the one
    whose frame rows ARE `target.block_frame_rows` on this exact frame."""
    return cal._TargetFrameProof(
        block=SimpleNamespace(x0=53, y0=target.block_frame_rows[0], x1=1027,
                              y1=target.block_frame_rows[1], hearts=(target.point,)),
        frame_sha256=cal._sha256(target.frame), mute_control_absent=True, heart_visible=True)


def test_a_card_parked_outside_the_autoplay_zone_is_recentred_instead_of_skipped(
        monkeypatch, tmp_path):
    """THE LIVE FAILURE (2026-08-22, campaign attempt 5), pinned.

    Navigation parked a depth-1 target one read-scroll quantum low; the still-photo proof
    honestly refused at its centring rung, and because that refusal happens BEFORE any reviewer
    checkpoint the loop's own SCROLL_UP/SCROLL_DOWN machinery could never be asked to fix it, so
    a profile the same card would have passed from true scroll top was Passed instead.  A
    positional refusal must now cost one bounded corrective scroll and a re-navigation -- and
    must cost it before the proof spends its dwell bursts and probe gestures.
    """
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(
        monkeypatch, extra_screencaps=1)
    monkeypatch.setattr(cal, "_frame_height_px", lambda _frame: _PARK_FRAME_HEIGHT_PX)
    parks = iter([_PARK_OFF_ZONE_ROWS, _PARK_IN_ZONE_ROWS])
    navigations: list[tuple[int, int]] = []

    def next_park(*_a, **_kw):
        rows = next(parks)
        navigations.append(rows)
        return _parked_target(rows)

    monkeypatch.setattr(cal, "navigate_to_item", next_park)
    monkeypatch.setattr(cal, "_verified_target_frame_proof", _park_bound_proof)
    stub_proof = cal._verified_still_photo_proof
    proved_rows: list[tuple[int, int]] = []

    def recording_proof(driver_, *, frame, block):
        proved_rows.append((block.y0, block.y1))
        return stub_proof(driver_, frame=frame, block=block)

    monkeypatch.setattr(cal, "_verified_still_photo_proof", recording_proof)
    gate = _RecordingGate()

    cal._capture_one_profile_unattended(
        driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
        review_gate=gate, send_like=False)

    # Re-navigated rather than translating the stale rect, and only once.
    assert navigations == [_PARK_OFF_ZONE_ROWS, _PARK_IN_ZONE_ROWS]
    # The card sat HIGH of the content centre, so the CONTENT had to come down: exactly one
    # planner-sized `_scroll_up_one`, and it came before anything else this profile scrolled.
    assert driver.scrolls[0] == ("up", 0.1, 0.5)
    assert [direction for direction, _frac, _x in driver.scrolls].count("up") == 1
    # The expensive proof never ran on the off-zone park -- that is the whole point of measuring
    # the position from evidence already in hand.
    assert proved_rows == [_PARK_IN_ZONE_ROWS]
    assert [state for state, _plan in gate.checkpoints if state == "target_heart_visible"]


def test_centring_corrections_share_the_reviewer_adjustment_budget(monkeypatch, tmp_path):
    """A card that keeps parking off-zone must not scroll forever: the corrections draw on the
    SAME bounded per-heart allowance a reviewer's adjustments draw on, and the profile skips
    when it is spent.  The skip detail names every offset that was tried and the zone, because
    "refused for centring" alone cannot tell a hopeless geometry from a correction that was
    converging and ran out of budget -- and those need opposite responses from the operator."""
    budget = cal._HYBRID_MAX_ADJUSTMENTS_PER_ACTION
    driver, _pass_calls, _send_calls = _unattended_single_item_fixtures(
        monkeypatch, extra_screencaps=budget)
    monkeypatch.setattr(cal, "_frame_height_px", lambda _frame: _PARK_FRAME_HEIGHT_PX)
    # Converging, but never far enough: -0.229, -0.220, -0.211, -0.203.
    parks = iter([(557, 757), (570, 770), (583, 783), (596, 796)])
    monkeypatch.setattr(cal, "navigate_to_item", lambda *_a, **_kw: _parked_target(next(parks)))
    monkeypatch.setattr(cal, "_verified_target_frame_proof", _park_bound_proof)
    monkeypatch.setattr(cal, "_skip_automated_profile_before_heart",
                        lambda _driver, **kwargs: {"reason_code": kwargs["reason"].code,
                                                   "reason_detail": kwargs["reason"].detail})
    gate = _RecordingGate()

    with pytest.raises(cal._ProfileSkipped) as excinfo:
        cal._capture_one_profile_unattended(
            driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
            review_gate=gate, send_like=False)

    assert excinfo.value.record["reason_code"] == "target_verification_blocked"
    detail = excinfo.value.record["reason_detail"]
    assert "-0.229 then -0.220 then -0.211 then -0.203" in detail
    assert f"{budget} corrective scroll(s)" in detail
    assert "limit 0.150" in detail
    # Exactly `budget` gestures were spent, and no heart was ever offered or taken.
    assert [direction for direction, _frac, _x in driver.scrolls] == ["up"] * budget
    assert gate.checkpoints == []
    assert driver.taps == []


def test_a_card_parked_inside_the_autoplay_zone_is_never_scrolled_before_its_proof(
        monkeypatch, tmp_path):
    """The common case must be untouched: an in-zone park navigates once, takes no corrective
    gesture, and runs straight through the proof, checkpoint and terminal Pass as it does today.
    A centring guard that re-scrolls a card already inside the zone would be spending real
    gestures -- and real re-navigations -- on every profile."""
    driver, pass_calls, send_calls = _unattended_single_item_fixtures(monkeypatch)
    monkeypatch.setattr(cal, "_frame_height_px", lambda _frame: _PARK_FRAME_HEIGHT_PX)
    navigations: list[tuple[int, int]] = []

    def one_park(*_a, **_kw):
        navigations.append(_PARK_IN_ZONE_ROWS)
        return _parked_target(_PARK_IN_ZONE_ROWS)

    monkeypatch.setattr(cal, "navigate_to_item", one_park)
    monkeypatch.setattr(cal, "_verified_target_frame_proof", _park_bound_proof)
    gate = _RecordingGate()

    cal._capture_one_profile_unattended(
        driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
        review_gate=gate, send_like=False)

    assert navigations == [_PARK_IN_ZONE_ROWS]
    # The only scroll this profile makes is the terminal bounded sticky-header probe.
    assert [direction for direction, _frac, _x in driver.scrolls] == ["down"]
    assert (pass_calls, send_calls) == ([1], [])
    assert [state for state, _plan in gate.checkpoints if state == "target_heart_visible"]


def _rebindable_target_frame_proof(calls: list[dict]):
    """A `_verified_target_frame_proof` stand-in that behaves like the real one for binding
    purposes -- its block sits at exactly `expected_rows`/`expected_point`, defaulting to the
    navigator's own rows/point exactly as the real function's own docstring specifies -- and
    records every call, so a test can assert both HOW MANY times the pre-heart loop re-proved the
    card and WHAT rows/point (and frame) each proof ran against.
    """
    def proof(_driver, target, *, frame, expected_rows=None, expected_point=None, **_kw):
        rows = tuple(target.block_frame_rows if expected_rows is None else expected_rows)
        point = tuple(target.point if expected_point is None else expected_point)
        calls.append({"frame": frame, "expected_rows": expected_rows,
                      "expected_point": expected_point})
        return cal._TargetFrameProof(
            block=SimpleNamespace(x0=53, y0=rows[0], x1=1027, y1=rows[1], hearts=(point,)),
            frame_sha256=cal._sha256(frame), mute_control_absent=True, heart_visible=True)
    return proof


def test_a_clock_tick_during_the_probe_rebinds_the_frame_without_translating_the_card(
        monkeypatch, tmp_path):
    """THE LIVE FAILURE (2026-08-22, campaign attempt 10), pinned.

    The still-photo probe's return leg can come back byte-DIFFERENT from `target_pre` while the
    measured residual is exactly 0: Android's status-bar clock ticks a minute forward inside the
    probe's own dwell window, which changes bytes outside the content band without moving
    anything the residual estimator looks at. The OLD gate (`if page_residual_px:`) read that as
    "nothing to do" and left `target_pre` on the stale bytes, so the still-photo proof's own
    `action_frame_sha256` -- correctly bound to the NEW bytes -- could never match
    `_sha256(target_pre)` and the checkpoint refused a card that never actually failed anything.
    The gate must fire on the bytes changing, not on the residual, and a residual of 0 must leave
    the parked rows/point untouched: nothing moved, only the chrome did.
    """
    driver, pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    clock_ticked_frame = b"target-pre-with-a-different-clock"
    proof = cal._StillPhotoProof(
        action_frame_sha256=cal._sha256(clock_ticked_frame),
        pre_probe_frame_sha256=cal._sha256(b"target-pre"),
        dwell_frame_sha256s=(cal._sha256(b"target-pre"), cal._sha256(b"dwell")),
        dwell_span_s=6.0, still_photo_verified=True,
        reattach_frame_sha256s=(cal._sha256(clock_ticked_frame), cal._sha256(b"reattach")),
        reattach_dwell_span_s=6.0,
        page_residual_px=0, action_frame=clock_ticked_frame)
    monkeypatch.setattr(cal, "_verified_still_photo_proof", lambda *_a, **_kw: proof)
    calls: list[dict] = []
    monkeypatch.setattr(cal, "_verified_target_frame_proof", _rebindable_target_frame_proof(calls))
    gate = _RecordingGate()

    cal._capture_one_profile_unattended(
        driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
        review_gate=gate, send_like=False)

    # Re-proved on the new bytes, not the stale ones the navigator originally parked on.
    assert [call["frame"] for call in calls] == [b"target-pre", clock_ticked_frame]
    # (700, 900) and (500, 800) are `_unattended_single_item_fixtures`'s fixed navigator rows and
    # point: unchanged, because a zero residual means there was nothing to translate.
    assert calls[1]["expected_rows"] == (700, 900)
    assert calls[1]["expected_point"] == (500, 800)
    assert cal._verified_still_photo_evidence(proof, clock_ticked_frame) is True
    heart_plan = next(plan for state, plan in gate.checkpoints if state == "target_heart_visible")
    assert heart_plan["predicates"]["positive_still_photo_evidence_verified"] is True
    assert heart_plan["predicates"]["target_frame_mute_control_screened_absent"] is True
    assert pass_calls == [1]


def test_a_measured_residual_still_translates_the_parked_rows_and_point(monkeypatch, tmp_path):
    """Guards the residual-translate behaviour 9fd32ab9 introduced, across the gate rewrite
    above: when the probe's return leg leaves a genuine non-zero residual, the rebind still
    happens (as it always did) AND the navigator's rows/point are still translated by exactly
    that residual. The bytes-based gate changes WHEN the branch fires, never what it does once it
    has fired.
    """
    driver, pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    settled_frame = b"target-pre-settled-elsewhere"
    residual = 40
    proof = cal._StillPhotoProof(
        action_frame_sha256=cal._sha256(settled_frame),
        pre_probe_frame_sha256=cal._sha256(b"target-pre"),
        dwell_frame_sha256s=(cal._sha256(b"target-pre"), cal._sha256(b"dwell")),
        dwell_span_s=6.0, still_photo_verified=True,
        reattach_frame_sha256s=(cal._sha256(settled_frame), cal._sha256(b"reattach")),
        reattach_dwell_span_s=6.0,
        page_residual_px=residual, action_frame=settled_frame)
    monkeypatch.setattr(cal, "_verified_still_photo_proof", lambda *_a, **_kw: proof)
    calls: list[dict] = []
    monkeypatch.setattr(cal, "_verified_target_frame_proof", _rebindable_target_frame_proof(calls))
    gate = _RecordingGate()

    cal._capture_one_profile_unattended(
        driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
        review_gate=gate, send_like=False)

    assert [call["frame"] for call in calls] == [b"target-pre", settled_frame]
    # (700, 900) and (500, 800) are the navigator's own rows/point; both must come back
    # translated by exactly the measured residual.
    assert calls[1]["expected_rows"] == (700 - residual, 900 - residual)
    assert calls[1]["expected_point"] == (500, 800 - residual)
    assert cal._verified_still_photo_evidence(proof, settled_frame) is True
    assert pass_calls == [1]


def test_a_byte_identical_return_leg_takes_the_fast_path_with_no_extra_reproof(
        monkeypatch, tmp_path):
    """The common case, pinned against a regression that would make every profile pay for a
    re-proof it does not need: when the probe's return leg lands back on the EXACT bytes the
    still-photo proof was handed, `_verified_target_frame_proof` must run exactly once for the
    item -- the original navigation-bound proof -- with no second call rebinding a frame that
    never changed.
    """
    driver, pass_calls, _send_calls = _unattended_single_item_fixtures(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(cal, "_verified_target_frame_proof", _rebindable_target_frame_proof(calls))
    gate = _RecordingGate()

    cal._capture_one_profile_unattended(
        driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=[], used_profile_ids=set(),
        review_gate=gate, send_like=False)

    assert len(calls) == 1
    assert calls[0]["frame"] == b"target-pre"
    assert pass_calls == [1]


# A 1080x2000 action frame puts the content band's centre at row 1000, so the block used
# throughout these tests (rows 700..1300, centre 1000) sits dead centre of Hinge's autoplay
# trigger zone. The zone is a precondition on the dwell (owner fact 2026-08-21: a video only
# plays near the centre of the screen), so a proof taken anywhere else refuses.
_ACTION_FRAME_SIZE = (1080, 2000)
_CENTRED_BLOCK = dict(x0=53, y0=700, x1=1027, y1=1300)
_OFF_CENTRE_BLOCK = dict(x0=53, y0=1500, x1=1027, y1=1900)


def _action_frame(value: int = 9) -> bytes:
    return _png(value, size=_ACTION_FRAME_SIZE)


def _patched_action_frame(value: int = 9, patch: int = 255) -> bytes:
    """An action frame with a handful of pixels changed inside the card rect.

    Byte-different from `_action_frame()` while its 32x32 signature distance stays two orders of
    magnitude under `_STILL_PHOTO_MAX_SIGNATURE_DRIFT`: the frame a stalled video emits when it
    finally restarts does not have to move much to stop being the same bytes.
    """
    image = np.full((_ACTION_FRAME_SIZE[1], _ACTION_FRAME_SIZE[0]), value, np.uint8)
    image[800:808, 100:108] = patch
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def _still_photo_driver(*, burst=None, span_s=6.0, mute_score=0.0,
                        content_band=(0.125, 0.875), probe=...,
                        probe_burst=None, probe_span_s=6.0, measured_page_shift=0):
    """A driver stub exposing only what the still-photo proof consumes: a dwell burst, the mute
    matcher, the content band the autoplay-centring precondition is measured against, and the
    re-attach probe.

    The FIRST burst issues no input; the probe does, and it hands back the settled frame it
    finished on. `probe=None` is the probe that could not complete, which must refuse the card
    rather than fall through to the first burst's verdict.

    `measured_page_shift` stands in for `HingeDriver._measured_page_shift`, consulted only when
    the probe's anchor is byte-different from the frame the first burst was chained to (an
    unmeasurable return leg is `None`, matching the real driver's own "cannot tell" contract).
    """
    if burst is None:
        # Byte-IDENTICAL to the anchor, because that is what a still photograph's dwell actually
        # produces and because the proof now measures its own C1 drift over exactly these frames.
        # A burst of differently-valued frames only ever passed because `dwell_exact_over_rect`
        # was monkeypatched over it.
        burst = (_action_frame(), _action_frame())
    frame = _action_frame()
    if probe is ...:
        probe = SimpleNamespace(
            anchor=frame,
            frames=tuple(probe_burst if probe_burst is not None
                         else (_action_frame(), _action_frame())),
            span_s=probe_span_s, page_shift_px=0)
    return SimpleNamespace(
        content_band=content_band,
        _still_photo_dwell_burst=lambda: (list(burst), span_s),
        _still_photo_reattach_probe=lambda _frame, _rect: probe,
        _match_video_mute=lambda _frame, _rect: (True, mute_score),
        _measured_page_shift=lambda _before, _after: measured_page_shift)


def test_without_a_verified_bound_the_still_photo_proof_can_never_pass(monkeypatch):
    """The unlicensed case is not special-cased anywhere: the SAME acceptance runs and refuses
    with the policy blocker, so a capture with no artifact cannot reach a heart even if every
    other gate above it were satisfied.  (The payload also numbers nothing, so the capture path
    never gets here in the first place -- this pins the second, independent stop.)"""
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: True)

    with pytest.raises(cal._CaptureAbort, match="positive still-photo discriminator unavailable"):
        cal._verified_still_photo_proof(
            _still_photo_driver(), frame=_action_frame(),
            block=SimpleNamespace(**_CENTRED_BLOCK))


@pytest.fixture
def installed_still_photo_bound():
    """The numbering licence, installed through the real config-validation API and torn down."""
    tp.install_verified_still_photo_bound(tp.StillPhotoBoundSummary(
        ground_truth_channel=tp.STILL_PHOTO_BOUND_GROUND_TRUTH_CHANNEL,
        human_ground_truth=True, video_cards=60, video_accepts=0, photo_cards=60,
        photo_false_refusals=0, max_video_exact_run_s=1.0, artifact_sha256="d" * 64,
        device="synthetic-pixel", hinge_version_name="10.0.1"))
    yield
    tp._reset_installed_still_photo_bound_for_tests()


def test_the_still_photo_proof_binds_the_action_frame_and_every_dwell_frame_by_digest(
        monkeypatch, installed_still_photo_bound):
    """With a bound installed and TWO byte-exact, fully screened dwells, the proof passes -- and
    it names the exact frames each was computed from, action frame first in both."""
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: True)

    frame = _action_frame()
    proof = cal._verified_still_photo_proof(
        _still_photo_driver(), frame=frame,
        block=SimpleNamespace(**_CENTRED_BLOCK))

    assert proof.still_photo_verified is True
    # The byte-identical fast path: the probe restored the screen exactly, so the action frame
    # and the pre-probe frame are the same bytes and the residual is zero.
    assert proof.action_frame_sha256 == cal._sha256(frame)
    assert proof.pre_probe_frame_sha256 == cal._sha256(frame)
    assert proof.page_residual_px == 0
    assert proof.action_frame == frame
    assert proof.dwell_frame_sha256s[0] == cal._sha256(frame)
    assert len(proof.dwell_frame_sha256s) == 3
    # The re-attach burst is bound the SAME way: the action frame leads it, because the probe
    # only accepts a round trip that came back byte-for-byte.
    assert proof.reattach_frame_sha256s[0] == cal._sha256(frame)
    assert len(proof.reattach_frame_sha256s) == 3
    assert proof.reattach_dwell_span_s == 6.0
    assert cal._verified_still_photo_evidence(proof, frame) is True
    with pytest.raises(cal._CaptureAbort, match="no still-photo .C1-C3. verdict is bound"):
        cal._verified_still_photo_evidence(proof, b"a-different-frame")


def test_the_still_photo_proof_passes_a_depth_one_target_with_no_read_scroll_drift(
        monkeypatch, installed_still_photo_bound):
    """THE DEPTH-1 CASE, which the shipped proof could never pass.

    The ladder's C1 rung used to be fed the CALLER's read-scroll drift, taken from the payload
    crop.  For the first card of a top-down read that number does not exist -- no other
    enumeration frame's analysed band contains the card's page rows, so `_signature_drift`
    returns `(None, ())` by construction -- so every odd-ordinal target refused here forever.
    The proof now measures C1 over the frames it actually holds, all of them of one parked
    screen, and the caller's read-scroll drift is not an input at all.
    """
    assert "signature_drift" not in inspect.signature(
        cal._verified_still_photo_proof).parameters
    assert "drift_frames" not in inspect.signature(
        cal._verified_still_photo_proof).parameters
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: True)

    proof = cal._verified_still_photo_proof(
        _still_photo_driver(), frame=_action_frame(), block=SimpleNamespace(**_CENTRED_BLOCK))

    assert proof.still_photo_verified is True


def test_the_proof_measures_its_own_drift_rather_than_assuming_a_parked_card_is_still(
        monkeypatch, installed_still_photo_bound):
    """The byte-exactness requirement makes the measured drift ~0.0; measuring it anyway is what
    keeps the drift rung load-bearing if that requirement is ever weakened.

    `dwell_exact_over_rect` is forced True here, so byte-exactness cannot be what refuses: a
    burst that visibly moved must be caught by a drift the proof computed for itself.  A
    hardcoded 0.0 would sail straight past this.
    """
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: True)
    moving = (_action_frame(30), _action_frame(30))

    with pytest.raises(cal._CaptureAbort, match="re-observation drift"):
        cal._verified_still_photo_proof(
            _still_photo_driver(burst=moving), frame=_action_frame(),
            block=SimpleNamespace(**_CENTRED_BLOCK))


def test_the_parked_drift_is_computed_with_the_modules_own_signature_primitives():
    """No second implementation of the measurement: the pre-heart proof and the payload builder
    have to agree by construction, so this reads `signature_of` and `CropSignature.distance`
    exactly as `_signature_drift` does, and reports ignorance the same way (`(None, ())`)."""
    rect = (53, 700, 1027, 1300)
    still = _action_frame()

    assert cal._parked_signature_drift([still], rect) == (None, ())
    assert cal._parked_signature_drift([], rect) == (None, ())
    drift, frames = cal._parked_signature_drift([still, still, still], rect)
    assert drift == 0.0 and frames == (1, 2)
    moved_drift, moved_frames = cal._parked_signature_drift([still, _action_frame(30)], rect)
    assert moved_drift == pytest.approx(21.0) and moved_frames == (1,)


@pytest.mark.parametrize("proof_kwargs, expected", [
    ({"reattach_frame_sha256s": ()},
     "no re-attach dwell burst is chained"),
    ({"reattach_frame_sha256s": ("a" * 64,)},
     "no re-attach dwell burst is chained"),
    ({"reattach_frame_sha256s": ("a" * 64, "b" * 64)},
     "no re-attach dwell burst is chained"),
    ({"dwell_frame_sha256s": ("a" * 64, "b" * 64)},
     "un-interacted dwell is not chained"),
], ids=["second-burst-missing", "second-burst-too-short", "second-burst-wrong-frame",
        "first-burst-wrong-frame"])
def test_the_checkpoint_refuses_a_proof_whose_bursts_do_not_bind_the_action_frame(
        proof_kwargs, expected):
    """A published predicate is a claim.  Both bursts have to name the frame whose heart would be
    approved, or the claim is about some other screen -- which is how a real video came one
    keystroke from a like."""
    frame = _action_frame()
    fields = {"action_frame_sha256": cal._sha256(frame),
              "pre_probe_frame_sha256": cal._sha256(frame),
              "dwell_frame_sha256s": (cal._sha256(frame), "b" * 64),
              "dwell_span_s": 6.0, "still_photo_verified": True,
              "reattach_frame_sha256s": (cal._sha256(frame), "c" * 64),
              "reattach_dwell_span_s": 6.0}
    proof = cal._StillPhotoProof(**{**fields, **proof_kwargs})

    with pytest.raises(cal._CaptureAbort, match=expected):
        cal._verified_still_photo_evidence(proof, frame)


@pytest.mark.parametrize("kwargs, expected", [
    ({"probe": None}, "re-attach probe could not take this card"),
    # A probe that gestured but brought back no burst made no observation either: one frame is
    # zero consecutive pairs, so it lands on the same "no probe" refusal rather than on motion.
    ({"probe_burst": ()}, "no re-attach probe scrolled this card"),
], ids=["probe-refused", "probe-took-no-frames"])
def test_the_still_photo_proof_refuses_a_card_the_re_attach_probe_could_not_clear(
        monkeypatch, installed_still_photo_bound, kwargs, expected):
    """The stalled-video residual, at the load-bearing gate: a card whose media was never asked
    to restart is skipped, exactly like a card the mute screen rejects."""
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: True)

    with pytest.raises(cal._CaptureAbort, match=expected):
        cal._verified_still_photo_proof(
            _still_photo_driver(**kwargs), frame=_action_frame(),
            block=SimpleNamespace(**_CENTRED_BLOCK))


def test_the_still_photo_proof_binds_the_post_return_frame_when_the_probes_residual_is_measured(
        monkeypatch, installed_still_photo_bound):
    """BYTE IDENTITY WAS NEVER THE PROBE'S CONTRACT (found live 2026-08-22, attempt 7): its
    return leg only ever promised a MEASURED net displacement, and refusing whenever it did not
    land back byte-for-byte made deeper targets structurally lucky.  A measured residual is now
    a second position to re-prove the card at -- rows translated by exactly that residual, per
    `frameshift.estimate_shift`'s own sign convention -- and the published proof binds the frame
    the page actually settled on (the one the heart is about to be offered and tapped on), not
    the one the first burst was taken on."""
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: True)
    frame = _action_frame()
    # `_patched_action_frame`'s own docstring: its 32x32 signature distance from `_action_frame()`
    # stays two orders of magnitude under the drift ceiling -- exactly what a still photo
    # re-rasterised at a new position should measure: small, not zero, nowhere near 0.24.
    settled = _patched_action_frame()
    probe = SimpleNamespace(anchor=settled, frames=(settled, settled), span_s=6.0,
                            page_shift_px=0)

    proof = cal._verified_still_photo_proof(
        _still_photo_driver(probe=probe, measured_page_shift=40), frame=frame,
        block=SimpleNamespace(**_CENTRED_BLOCK))

    assert proof.still_photo_verified is True
    assert proof.page_residual_px == 40
    assert proof.action_frame_sha256 == cal._sha256(settled)
    assert proof.pre_probe_frame_sha256 == cal._sha256(frame)
    assert proof.action_frame == settled
    assert proof.reattach_frame_sha256s[0] == cal._sha256(settled)
    assert cal._verified_still_photo_evidence(proof, settled) is True
    # The pre-probe frame no longer binds a checkpoint: the action frame is the one that ran.
    with pytest.raises(cal._CaptureAbort, match="no still-photo .C1-C3. verdict is bound"):
        cal._verified_still_photo_evidence(proof, frame)


def test_the_still_photo_proof_refuses_an_unmeasurable_probe_return_leg_residual(
        monkeypatch, installed_still_photo_bound):
    """A residual that CANNOT be measured is still a refusal: there is nothing to translate the
    card's rect by, so the second burst cannot be bound to the reviewed card at all -- the same
    "I cannot tell" contract `_measured_page_shift` documents for its every other caller."""
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: True)
    moved = SimpleNamespace(anchor=_action_frame(77), frames=(_action_frame(77), _action_frame(77)),
                            span_s=6.0, page_shift_px=12)

    with pytest.raises(cal._CaptureAbort, match="could not be measured"):
        cal._verified_still_photo_proof(
            _still_photo_driver(probe=moved, measured_page_shift=None), frame=_action_frame(),
            block=SimpleNamespace(**_CENTRED_BLOCK))


def test_the_still_photo_proof_refuses_a_residual_that_pushes_the_card_off_screen(
        monkeypatch, installed_still_photo_bound):
    """A residual large enough to carry the card's translated rows past the edge of the frame
    leaves no rect there to re-prove the second burst against, however small the true
    displacement of the pixels underneath it actually was."""
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: True)
    settled = _action_frame(50)
    probe = SimpleNamespace(anchor=settled, frames=(settled, settled), span_s=6.0,
                            page_shift_px=0)

    with pytest.raises(cal._CaptureAbort, match="800px residual"):
        cal._verified_still_photo_proof(
            _still_photo_driver(probe=probe, measured_page_shift=800), frame=_action_frame(),
            block=SimpleNamespace(**_CENTRED_BLOCK))


def test_the_cross_position_drift_is_a_real_measurement_that_can_still_refuse(
        monkeypatch, installed_still_photo_bound):
    """The two-position re-observation the residual path substitutes for byte-identity is a
    MEASUREMENT, not an assumption: build a second position whose own burst is internally
    byte-exact (so nothing above the reattach rung catches it) but whose card crop CONTENT is
    far from the first position's, and the ladder's 0.24 ceiling refuses it exactly as it
    refuses in-place motion."""
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: True)
    settled = _action_frame(230)  # far from frame's constant value 9: a real measurement, not 0
    probe = SimpleNamespace(anchor=settled, frames=(settled, settled), span_s=6.0,
                            page_shift_px=0)

    with pytest.raises(cal._CaptureAbort, match="re-observation drift"):
        cal._verified_still_photo_proof(
            _still_photo_driver(probe=probe, measured_page_shift=40), frame=_action_frame(),
            block=SimpleNamespace(**_CENTRED_BLOCK))


def test_the_second_burst_is_measured_for_real_and_motion_in_it_refuses_the_card(
        monkeypatch, installed_still_photo_bound):
    """Byte-exactness of the SECOND burst is recomputed from its own bytes.  Media that starts
    when Hinge re-attaches it is a video, however still it was while the first burst watched."""
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: True)
    # A few pixels, deliberately: this motion is far below the drift ceiling (the 32x32
    # signature averages it away almost entirely), so the ONLY rung that can see it is the
    # second burst's byte-exactness.  A whole-frame change would be caught by the drift rung
    # first and this test would stop proving anything about rung (g).
    playing = (_patched_action_frame(), _patched_action_frame())

    with pytest.raises(cal._CaptureAbort, match="second dwell burst"):
        cal._verified_still_photo_proof(
            _still_photo_driver(probe_burst=playing), frame=_action_frame(),
            block=SimpleNamespace(**_CENTRED_BLOCK))


def test_a_card_that_fails_a_cheaper_rung_is_never_charged_a_probe(
        monkeypatch, installed_still_photo_bound):
    """The probe costs two real gestures, so it runs only once every rung that answers from
    frames we already hold has passed -- and the refusal names that rung, not the probe."""
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: False)
    probes = []
    driver = _still_photo_driver()
    driver._still_photo_reattach_probe = (
        lambda _frame, _rect: probes.append(1) or SimpleNamespace(
            anchor=_action_frame(), frames=(_action_frame(),), span_s=6.0, page_shift_px=0))

    with pytest.raises(cal._CaptureAbort, match="no un-interacted dwell proved"):
        cal._verified_still_photo_proof(
            driver, frame=_action_frame(), block=SimpleNamespace(**_CENTRED_BLOCK))

    assert probes == [], "a card refused by a cheaper rung must not move the screen"


@pytest.mark.parametrize(
    ("kwargs", "exact", "expected"),
    [
        ({}, False, "no un-interacted dwell proved"),
        ({"mute_score": 0.999}, True, "mute-control screening did not complete"),
        ({"span_s": 0.0}, True, "dwell window is missing or non-positive"),
        ({"burst": ()}, True, "no un-interacted dwell proved"),
    ],
    ids=["not-byte-exact", "mute-control-rendered-during-dwell", "no-window", "no-dwell"],
)
def test_the_still_photo_proof_refuses_the_card_rather_than_returning_a_negative_verdict(
        monkeypatch, installed_still_photo_bound, kwargs, exact, expected):
    """A card the acceptance rejects is skipped exactly like one the mute screen rejects, so the
    reviewer is never offered its heart and no False can reach a checkpoint at all."""
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: exact)

    with pytest.raises(cal._CaptureAbort, match=expected):
        cal._verified_still_photo_proof(
            _still_photo_driver(**kwargs), frame=_action_frame(),
            block=SimpleNamespace(**_CENTRED_BLOCK))


@pytest.mark.parametrize("block, band, reason", [
    (_OFF_CENTRE_BLOCK, (0.125, 0.875), "autoplay trigger zone"),
    (_CENTRED_BLOCK, None, "autoplay trigger zone"),
], ids=["off-centre-card", "no-content-band-to-measure-against"])
def test_the_still_photo_proof_refuses_a_card_outside_the_autoplay_trigger_zone(
        monkeypatch, installed_still_photo_bound, block, band, reason):
    """Hinge plays a video only near the centre, so a dwell taken elsewhere proves nothing.

    The second case is the fail-closed one: a driver that cannot say where the card sat has not
    made the observation, and an unmeasured position must never read as a passing one.
    """
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: True)

    with pytest.raises(cal._CaptureAbort, match=reason):
        cal._verified_still_photo_proof(
            _still_photo_driver(content_band=band), frame=_action_frame(),
            block=SimpleNamespace(**block))


def _png(value: int, size=(1000, 600)) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((size[1], size[0]), value, np.uint8))
    assert ok
    return buf.tobytes()


def _frame_with_patch(value: int, patch: int, *, rows: tuple[int, int], cols: tuple[int, int],
                      size=_ACTION_FRAME_SIZE) -> bytes:
    """An otherwise-uniform frame with `patch` written into `rows`x`cols`.

    Generalizes `_patched_action_frame` so a test can place the differing pixels precisely
    inside or outside the content band -- e.g. a status-bar clock tick above the band, or a
    handful of pixels inside the reviewed card's own rect.
    """
    image = np.full((size[1], size[0]), value, np.uint8)
    image[rows[0]:rows[1], cols[0]:cols[1]] = patch
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def test_content_band_rect_is_derived_from_the_configured_band_not_hardcoded():
    """Guards against a hardcoded 300/2100: those exact numbers are what (0.125, 0.875) happens
    to produce on the Pixel 7a's own 1080x2400 frame, which is exactly why a regression that
    hardcoded them would pass every other test in this file. A DIFFERENT band fraction on the
    SAME realistic frame size must move the rect proportionally, and the rect must span the
    frame's full decoded width, not a hardcoded 1080."""
    frame = _png(9, size=(1080, 2400))

    assert cal._content_band_rect_px(frame, (0.125, 0.875)) == (0, 300, 1080, 2100)
    assert cal._content_band_rect_px(frame, (0.1, 0.9)) == (0, 240, 1080, 2160)
    assert cal._identity_band_rect_px(frame, _IDENTITY_BAND) == (108, 115, 864, 225)
    assert cal._reviewed_target_protected_prefix_rect(
        frame, (0.125, 0.875), (700, 1500)) == (0, 300, 1080, 1500)
    # Complete, frame-bounded target cards may start slightly above the content-band boundary.
    # The protected region grows upward to keep the entire target exact rather than refusing it.
    assert cal._reviewed_target_protected_prefix_rect(
        frame, (0.125, 0.875), (250, 1500)) == (0, 250, 1080, 1500)


def test_target_frame_proof_carries_the_screen_verdict_bound_to_the_screened_frame(monkeypatch):
    """The proof object itself is the contract the checkpoint reads: it must report the screen's
    own verdict for the exact bytes it was given, and vouch for no other frame."""
    target = SimpleNamespace(frame=b"target-pre", point=(500, 800), block_frame_rows=(700, 900))
    driver = SimpleNamespace()
    driver._target_frame_video_screen_reason = lambda *_a: None
    monkeypatch.setattr(
        cal, "segment_frame",
        lambda *_a, **_kw: SimpleNamespace(
            ok=True, failures=(),
            blocks=(SimpleNamespace(x0=53, y0=700, x1=1027, y1=900, hearts=((500, 800),)),)))

    proof = cal._verified_target_frame_proof(
        driver, target, frame=b"target-pre", content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=0.8)

    assert proof.mute_control_absent is True
    assert proof.heart_visible is True
    assert proof.frame_sha256 == cal._sha256(b"target-pre")
    assert cal._screened_mute_control_absent(proof, b"target-pre") is True
    assert cal._located_target_heart_visible(proof, b"target-pre") is True
    with pytest.raises(cal._CaptureAbort, match="no target-frame mute-control screen is bound"):
        cal._screened_mute_control_absent(proof, b"another-frame")
    with pytest.raises(cal._CaptureAbort, match="no target-frame mute-control screen is bound"):
        cal._screened_mute_control_absent(True, b"target-pre")
    with pytest.raises(cal._CaptureAbort, match="no target-heart location is bound"):
        cal._located_target_heart_visible(proof, b"another-frame")
    with pytest.raises(cal._CaptureAbort, match="no target-heart location is bound"):
        cal._located_target_heart_visible(True, b"target-pre")


# Real, decodable PNGs standing in for the reviewed frame and a fresh capture of it. The protected
# prefix/identity comparisons decode both sides (they have to, to know each frame's dimensions),
# so a fake placeholder like the old `b"target-pre"` no longer stands in for a screen -- these are still
# 1080-wide like `_ACTION_FRAME_SIZE` so the (700, 900) card rect used throughout these tests
# sits inside the content band exactly as it did before.
_FRESH_TARGET_FRAME = _action_frame()
# Changed INSIDE the content band (rows 250..1750 for this 2000-tall frame) and inside the
# reviewed card's own rect (700..900): the class of change the byte-identity gate exists to catch.
_FRESH_TARGET_FRAME_BAND_CHANGED = _frame_with_patch(9, 200, rows=(750, 758), cols=(100, 108))


@pytest.mark.parametrize(
    ("fresh_frame", "reviewed_point", "detected_point", "identity_matched"),
    [
        (_FRESH_TARGET_FRAME_BAND_CHANGED, [500, 800], (500, 800), True),
        (_FRESH_TARGET_FRAME, [501, 800], (500, 800), True),
        (_FRESH_TARGET_FRAME, [500, 800], (501, 800), True),
        (_FRESH_TARGET_FRAME, [500, 800], (500, 800), False),
    ],
    ids=["frame-changed", "plan-point-changed", "detected-point-changed", "identity-changed"],
)
def test_fresh_reviewed_target_refuses_changed_frame_point_or_identity_without_tapping(
        monkeypatch, fresh_frame, reviewed_point, detected_point, identity_matched):
    prior_identity = SimpleNamespace(identity=SimpleNamespace(), match_max=1.0)
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=prior_identity)
    block = SimpleNamespace(y0=700, y1=900, hearts=(detected_point,))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: fresh_frame), identity_band=_IDENTITY_BAND,
        taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: None
    monkeypatch.setattr(
        cal, "segment_frame",
        lambda *_a, **_kw: SimpleNamespace(ok=True, failures=(), blocks=(block,)))
    monkeypatch.setattr(
        cal, "compare_profile_identity",
        lambda *_a, **_kw: SimpleNamespace(matched=identity_matched, reason="identity changed"))

    with pytest.raises(cal._CaptureAbort):
        point = cal._fresh_reviewed_target_point(
            driver, target, reviewed_point=reviewed_point, content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)
        driver._tap(*point)

    assert driver.taps == []


def test_a_status_bar_clock_tick_does_not_invalidate_a_reviewed_heart(monkeypatch):
    """Found live 2026-08-22 (campaign attempt 8, then attempt 8's post-mortem): Android
    repaints the status-bar clock every 60 seconds with no bearing whatsoever on the tap target,
    so the old full-frame byte check refused a heart a reviewer had correctly just approved --
    the diffed refusal showed only rows 43-76 (the clock) differing, with the card and heart
    byte-identical. Scoping the comparison to the content band must accept exactly this class of
    change (here stood in for by a patch at rows 40..80, well above the band's row-250 start on
    this frame) while still spending the reviewed coordinates on the one true tap.
    """
    fresh = _frame_with_patch(9, 200, rows=(40, 80), cols=(0, _ACTION_FRAME_SIZE[0]))
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: fresh), identity_band=_IDENTITY_BAND, taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: None
    monkeypatch.setattr(
        cal, "segment_frame",
        lambda *_a, **_kw: SimpleNamespace(
            ok=True, failures=(),
            blocks=(SimpleNamespace(y0=700, y1=900, hearts=((500, 800),)),)))
    monkeypatch.setattr(
        cal, "compare_profile_identity",
        lambda *_a, **_kw: SimpleNamespace(matched=True, reason="matched"))

    point = cal._fresh_reviewed_target_point(
        driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=0.8)
    driver._tap(*point)

    assert driver.taps == [(500, 800)]


def test_lower_card_only_change_preserves_the_reviewed_target_and_reproves_before_tap(monkeypatch):
    """An unrelated later card may animate; the reviewed prefix and identity cannot."""
    # Rows 1200..1208 are below the reviewed card bottom (900), but still inside the configured
    # content band.  This is the live lower-video shape that a full-band equality gate rejected.
    fresh = _frame_with_patch(9, 200, rows=(1200, 1208), cols=(100, 108))
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    video_checks, identity_checks = [], []
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: fresh), identity_band=_IDENTITY_BAND, taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda frame, block: video_checks.append(
        (frame, block)) or None
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_a, **_kw: _stub_scroll_top_verdict("confirmed_not_top", distance=12.0))
    monkeypatch.setattr(
        cal, "segment_frame",
        lambda *_a, **_kw: SimpleNamespace(
            ok=True, failures=(),
            blocks=(SimpleNamespace(x0=53, y0=700, x1=1027, y1=900, hearts=((500, 800),)),)))
    monkeypatch.setattr(
        cal, "compare_profile_identity",
        lambda *args, **_kw: identity_checks.append(args) or SimpleNamespace(matched=True, reason="matched"))

    point = cal._fresh_reviewed_target_point(
        driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=0.8)
    driver._tap(*point)

    assert driver.taps == [(500, 800)]
    assert video_checks and video_checks[0][0] == fresh
    assert identity_checks, "the sticky-header identity re-proof must remain mandatory"


@pytest.mark.parametrize("rows", [(400, 408), (750, 758)], ids=["header-prefix", "target-card"])
def test_any_change_in_the_protected_prefix_refuses_before_tap(monkeypatch, rows):
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    fresh = _frame_with_patch(9, 200, rows=rows, cols=(100, 108))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: fresh), identity_band=_IDENTITY_BAND, taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: pytest.fail("re-proof follows only exact prefix")

    with pytest.raises(cal._CaptureAbort, match="protected content prefix changed"):
        cal._fresh_reviewed_target_point(
            driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)

    assert driver.taps == []


def test_lower_card_change_plus_identity_band_mutation_refuses_before_tap(monkeypatch):
    """Excluding later cards cannot exempt the configured header ROI from exact equality."""
    image = np.full((_ACTION_FRAME_SIZE[1], _ACTION_FRAME_SIZE[0]), 9, np.uint8)
    image[1200:1208, 100:108] = 200  # unrelated later card, below the protected prefix
    image[120:128, 120:128] = 201    # inside configured identity_band (rows 96..188)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: encoded.tobytes()), identity_band=_IDENTITY_BAND,
        taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: pytest.fail("identity mutation must stop first")

    with pytest.raises(cal._CaptureAbort, match="profile identity band changed"):
        cal._fresh_reviewed_target_point(
            driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)

    assert driver.taps == []


def test_rebound_expected_rows_set_the_protected_prefix_bottom(monkeypatch):
    """A post-reattach residual rebind must not protect only the navigator's old rows."""
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    # This differs below the old bottom (900) but inside rebound rows ending at 1050.
    fresh = _frame_with_patch(9, 200, rows=(975, 983), cols=(100, 108))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: fresh), identity_band=_IDENTITY_BAND, taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: pytest.fail("prefix must refuse first")

    with pytest.raises(cal._CaptureAbort, match="protected content prefix changed"):
        cal._fresh_reviewed_target_point(
            driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8,
            expected_frame=_FRESH_TARGET_FRAME, expected_rows=(800, 1050),
            expected_point=(500, 800))

    assert driver.taps == []


def test_a_small_change_inside_the_reviewed_card_still_refuses_a_stale_checkpoint(monkeypatch):
    """The protected-prefix comparison stays byte-EXACT, not merely 'close enough': a few pixels
    flipped inside the reviewed card's own rect (a stalled video waking up, a control fading in)
    is exactly the class of change the old full-frame check existed to catch, and narrowing the
    scope below the target must not let any of it through. Also pins the refusal message, which
    must name the protected content prefix rather than the old whole-frame wording.
    """
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: _FRESH_TARGET_FRAME_BAND_CHANGED),
        identity_band=_IDENTITY_BAND, taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: None

    with pytest.raises(cal._CaptureAbort, match="protected content prefix changed"):
        point = cal._fresh_reviewed_target_point(
            driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)
        driver._tap(*point)

    assert driver.taps == []


def test_fresh_reviewed_target_unchanged_revalidates_then_taps_once(monkeypatch):
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: _FRESH_TARGET_FRAME), identity_band=_IDENTITY_BAND,
        taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: None
    monkeypatch.setattr(
        cal, "segment_frame",
        lambda *_a, **_kw: SimpleNamespace(
            ok=True, failures=(),
            blocks=(SimpleNamespace(y0=700, y1=900, hearts=((500, 800),)),)))
    monkeypatch.setattr(
        cal, "compare_profile_identity",
        lambda *_a, **_kw: SimpleNamespace(matched=True, reason="matched"))

    point = cal._fresh_reviewed_target_point(
        driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=0.8)
    driver._tap(*point)

    assert driver.taps == [(500, 800)]


def test_fresh_reviewed_target_refuses_auto_hidden_video_control_without_tapping(monkeypatch):
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: _FRESH_TARGET_FRAME), identity_band=_IDENTITY_BAND,
        taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: "selected media is a video"
    monkeypatch.setattr(
        cal, "segment_frame",
        lambda *_a, **_kw: SimpleNamespace(
            ok=True, failures=(),
            blocks=(SimpleNamespace(y0=700, y1=900, hearts=((500, 800),)),)))

    with pytest.raises(cal._CaptureAbort, match="selected media is a video"):
        point = cal._fresh_reviewed_target_point(
            driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)
        driver._tap(*point)

    assert driver.taps == []


def test_fresh_reviewed_target_refuses_when_the_reviewed_frame_cannot_be_decoded(monkeypatch):
    """The reviewed frame's own bytes are what the band rect is measured against; if they cannot
    even be decoded, the comparison cannot be scoped at all, and that must fail closed with a
    message naming the real cause rather than a generic or silently-passed refusal."""
    target = SimpleNamespace(
        frame=b"not a real png", point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: _FRESH_TARGET_FRAME), identity_band=_IDENTITY_BAND,
        taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: None

    with pytest.raises(cal._CaptureAbort, match="could not be read to scope the protected"):
        point = cal._fresh_reviewed_target_point(
            driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)
        driver._tap(*point)

    assert driver.taps == []


def test_fresh_reviewed_target_wraps_an_unreadable_band_rect_as_a_refusal(monkeypatch):
    """An inverted `content_band` produces a structurally invalid rect, which is a CALLER bug
    per `dwell_exact_over_rect`'s own contract (it raises `ItemCropError` rather than returning
    False for that case). That must still surface here as an ordinary refusal naming the content
    band, never as an uncaught exception escaping the checkpoint."""
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: _FRESH_TARGET_FRAME), identity_band=_IDENTITY_BAND,
        taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: None

    with pytest.raises(cal._CaptureAbort, match="protected prefix/identity comparison"):
        point = cal._fresh_reviewed_target_point(
            driver, target, reviewed_point=[500, 800], content_band=(0.9, 0.1),
            like_template=object(), like_threshold=0.8)
        driver._tap(*point)

    assert driver.taps == []


def test_the_identity_gate_compares_the_reviewed_frame_not_the_navigators_frame(monkeypatch):
    """Found live 2026-08-22 (campaign attempt 9): the pre-heart loop rebinds this SAME card to a
    POST-PROBE `expected_frame` after the still-photo probe's return leg leaves a measured page
    residual. `target.identity` is bound to the profile's original index-build frame, which can
    legitimately sit at a DIFFERENT scroll position than `expected_frame` -- Hinge's identity band
    shows the profile-independent filter-chips row at the very top and the sticky per-profile
    header once scrolled at all, and those are different content by design. A live run diffed a
    fresh frame byte-identical to its reviewed checkpoint and still had a valid heart refused at
    19.156 grey levels, because the old code compared against that stale, differently-scrolled
    reference instead of against the reviewed frame itself.

    This exercises the REAL `confirm_scroll_top`/`band_fingerprint`/`compare_profile_identity`
    pipeline (no mocking of the identity comparison itself): `target.identity` is deliberately
    bound to a fingerprint that would refuse the fresh frame if it were consulted, so the test
    would fail under the old (buggy) code path and only passes because the gate now fingerprints
    `expected_frame` directly and compares the fresh frame against THAT.
    """
    navigators_frame = _action_frame()  # target.frame's own identity band: untouched, value 9
    rebound_frame = _frame_with_patch(9, 130, rows=(96, 188), cols=(108, 864))  # sticky header
    stale_reference = ProfileIdentity(
        fingerprint=cal.band_fingerprint(
            navigators_frame, identity_band=_IDENTITY_BAND, grid=cal._IDENTITY_GRID),
        band=_IDENTITY_BAND, grid=cal._IDENTITY_GRID, frame_index=0, scroll_top_distance=200.0,
        reason="stale reference bound to the navigator's own frame, not the reviewed one",
        agreeing_frames=3)
    target = SimpleNamespace(
        frame=navigators_frame, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=stale_reference, match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: rebound_frame), identity_band=_IDENTITY_BAND,
        taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: None
    monkeypatch.setattr(
        cal, "segment_frame",
        lambda *_a, **_kw: SimpleNamespace(
            ok=True, failures=(),
            blocks=(SimpleNamespace(y0=700, y1=900, hearts=((500, 800),)),)))

    point = cal._fresh_reviewed_target_point(
        driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=0.8,
        expected_frame=rebound_frame, expected_rows=(700, 900), expected_point=(500, 800))
    driver._tap(*point)

    assert driver.taps == [(500, 800)]


@pytest.mark.parametrize(
    "rebind", [False, True], ids=["default-navigator-frame", "rebound-expected-frame"])
def test_a_genuinely_different_profile_in_the_fresh_frame_still_refuses(monkeypatch, rebind):
    """A changed identity strip now refuses before fuzzy semantic identity comparison.

    The exact ROI is deliberately stricter than the later sticky-header comparator: a distinct
    profile is stopped without giving any score threshold the opportunity to tolerate it.  The
    unchanged-ROI semantic comparator remains covered by the separate identity-mismatch test.
    """
    reviewed_frame = _frame_with_patch(9, 130, rows=(96, 188), cols=(108, 864))
    foreign_profile_fresh = _frame_with_patch(9, 230, rows=(96, 188), cols=(108, 864))
    target = SimpleNamespace(
        frame=reviewed_frame, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: foreign_profile_fresh),
        identity_band=_IDENTITY_BAND, taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: None
    monkeypatch.setattr(
        cal, "segment_frame",
        lambda *_a, **_kw: SimpleNamespace(
            ok=True, failures=(),
            blocks=(SimpleNamespace(y0=700, y1=900, hearts=((500, 800),)),)))
    identity_calls = []

    def _spy_compare_profile_identity(*args, **kwargs):
        identity_calls.append((args, kwargs))
        return SimpleNamespace(matched=False, reason="must not be reached")

    monkeypatch.setattr(cal, "compare_profile_identity", _spy_compare_profile_identity)
    kwargs = ({"expected_frame": reviewed_frame, "expected_rows": (700, 900),
              "expected_point": (500, 800)} if rebind else {})

    with pytest.raises(cal._CaptureAbort, match="profile identity band changed"):
        point = cal._fresh_reviewed_target_point(
            driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8, **kwargs)
        driver._tap(*point)

    assert driver.taps == []
    assert identity_calls == []


def _raise_scroll_top_error(*_args, **_kwargs):
    raise cal.ScrollTopError("identity band could not be decoded")


def _stub_scroll_top_verdict(state: str, *, distance: float | None = 5.0) -> ScrollTopVerdict:
    """A canned verdict for driving `_fresh_reviewed_target_point`'s branch by STATE, rather than
    by constructing a frame that happens to land there. The CONTEXT that motivated this fix
    explicitly prefers this for determinism over threading real fingerprints through a frame."""
    return ScrollTopVerdict(
        state=state, distance=distance, band=_IDENTITY_BAND, grid=(16, 4),
        confirm_max=3.0, refute_min=9.0, reason=f"stubbed verdict for state {state!r}")


def test_the_identity_gate_refuses_when_the_reviewed_frame_cannot_be_looked_at_at_all(
        monkeypatch):
    """`confirm_scroll_top` itself can raise `ScrollTopError` when it cannot decode the reviewed
    frame's identity band at all -- 'could not look', distinct from a verdict reached BY looking
    (confirmed/refuted/cannot_tell). That must always hard-abort, in every regime, because there
    is no verdict yet to make a skip-vs-fingerprint decision from.
    """
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: _FRESH_TARGET_FRAME), identity_band=_IDENTITY_BAND,
        taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: None
    monkeypatch.setattr(cal, "confirm_scroll_top", _raise_scroll_top_error)

    with pytest.raises(
            cal._CaptureAbort,
            match="the reviewed frame's profile identity could not be re-read"):
        point = cal._fresh_reviewed_target_point(
            driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)
        driver._tap(*point)

    assert driver.taps == []


def test_the_identity_gate_refuses_when_the_refuted_frames_band_cannot_be_fingerprinted(
        monkeypatch):
    """Adapted from the pre-fix version of this test, which also covered a `cannot_tell` verdict
    aborting here -- that case no longer aborts (see
    `test_fresh_reviewed_target_skips_identity_fingerprint_when_band_is_not_refuted`), because
    `cannot_tell` carries no usable per-profile signal and the content-band proof above already
    established same-profile. What remains correct is this: `confirm_scroll_top` on
    `_FRESH_TARGET_FRAME` (a uniform frame, nowhere near any filter-chips fingerprint) reads
    REFUTED, which is the one regime where this gate must take a fingerprint -- and if THAT step
    cannot even decode the frame, that is still 'could not look', not 'looked and it is chrome',
    and must still hard-abort with the same message rather than silently skipping.
    """
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: _FRESH_TARGET_FRAME), identity_band=_IDENTITY_BAND,
        taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: None
    monkeypatch.setattr(cal, "band_fingerprint", _raise_scroll_top_error)

    with pytest.raises(
            cal._CaptureAbort,
            match="the reviewed frame's profile identity could not be re-read"):
        point = cal._fresh_reviewed_target_point(
            driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)
        driver._tap(*point)

    assert driver.taps == []


@pytest.mark.parametrize(
    ("state", "distance"),
    [(SCROLL_TOP_CONFIRMED, 1.5), (SCROLL_TOP_UNKNOWN, 8.078)],
    ids=["reviewed-frame-confirmed-top", "reviewed-frame-cannot-tell"],
)
def test_fresh_reviewed_target_skips_identity_fingerprint_when_band_is_not_refuted(
        monkeypatch, state, distance):
    """THE CORRECT DESIGN this fix ships: when the reviewed frame's own scroll-top verdict does
    NOT refute top, the identity band is either Hinge's profile-independent filter-chips row
    (`confirmed`) or genuinely indeterminate (`cannot_tell`) -- carrying no usable per-profile
    signal either way -- so the gate must SKIP the sticky-band fingerprint/comparison rather than
    abort, and still reach a valid tap. `compare_profile_identity` must not be called at all: the
    content-band byte comparison that already ran above is the (stronger) proof of same-profile,
    because Hinge renders the profile's name header inside that same band.

    Live evidence this exact regime fixes (2026-08-22, campaign attempt 9): the reviewed frame
    returned `cannot_tell` at 8.078 grey levels, inside the deliberate 3.0-9.0 dead zone, because
    the card parks just below top on this campaign; the fresh frame was byte-identical to it
    across both the identity band and the content band, and the old code aborted anyway.
    """
    monkeypatch.setattr(
        cal, "confirm_scroll_top",
        lambda *_a, **_kw: _stub_scroll_top_verdict(state, distance=distance))
    identity_calls = []
    monkeypatch.setattr(
        cal, "compare_profile_identity",
        lambda *a, **kw: identity_calls.append((a, kw))
        or SimpleNamespace(matched=True, reason="must not be reached"))
    monkeypatch.setattr(
        cal, "segment_frame",
        lambda *_a, **_kw: SimpleNamespace(
            ok=True, failures=(),
            blocks=(SimpleNamespace(y0=700, y1=900, hearts=((500, 800),)),)))
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: _FRESH_TARGET_FRAME), identity_band=_IDENTITY_BAND,
        taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: None

    point = cal._fresh_reviewed_target_point(
        driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=0.8)
    driver._tap(*point)

    assert driver.taps == [(500, 800)]
    assert identity_calls == []


@pytest.mark.parametrize(
    "state", [SCROLL_TOP_CONFIRMED, SCROLL_TOP_UNKNOWN],
    ids=["reviewed-frame-confirmed-top", "reviewed-frame-cannot-tell"],
)
def test_fresh_reviewed_target_still_refuses_content_band_change_when_identity_check_is_skipped(
        monkeypatch, state):
    """Proves the skip above did not open a hole. Even when the reviewed frame's scroll-top
    verdict is not refuted (so the sticky-band fingerprint is skipped per the test above), a
    fresh frame that differs INSIDE the content band -- the class of change the byte-identity
    gate exists to catch -- must still refuse, at the same content-band comparison that runs
    unconditionally, before the skip decision is even reached.
    """
    monkeypatch.setattr(
        cal, "confirm_scroll_top", lambda *_a, **_kw: _stub_scroll_top_verdict(state))
    identity_calls = []
    monkeypatch.setattr(
        cal, "compare_profile_identity",
        lambda *a, **kw: identity_calls.append((a, kw))
        or SimpleNamespace(matched=True, reason="must not be reached"))
    target = SimpleNamespace(
        frame=_FRESH_TARGET_FRAME, point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: _FRESH_TARGET_FRAME_BAND_CHANGED),
        identity_band=_IDENTITY_BAND, taps=[])
    driver._tap = lambda *point: driver.taps.append(point)
    driver._target_frame_video_screen_reason = lambda *_a: None

    with pytest.raises(cal._CaptureAbort, match="protected content prefix changed"):
        point = cal._fresh_reviewed_target_point(
            driver, target, reviewed_point=[500, 800], content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=0.8)
        driver._tap(*point)

    assert driver.taps == []
    assert identity_calls == []


@pytest.mark.parametrize("item_number", [1, 3], ids=["odd-profile-photo-1", "even-profile-photo-3"])
def test_calibration_only_pass_allows_composer_offscreen_after_edge_back_then_uses_guarded_transport(
        monkeypatch, item_number):
    """The special composer route is narrow and never delegates to public `dislike()`.

    Production Pass correctly requires the ordinary deck pair of glyphs.  On the live Hinge
    surface, edge-back hides the keyboard *and restores the old scroll offset*, so the verified
    composer can be completely offscreen while the floating Pass X returns.  The identity header
    is the post-edge invariant, rather than a made-up absolute composer position.
    """
    focused = ComposerSurface(cal._COMPOSER_LAYOUT_ID, Rect(80, 1150, 1000, 1290),
                              Rect(390, 1300, 985, 1420), (690, 1360))
    identity = ProfileIdentity((7,), _IDENTITY_BAND, (1, 1), 1, 20.0, "known",
                               agreeing_frames=2)

    class _Adb:
        def __init__(self):
            self.state = "focused"

        def screen_size(self):
            return 1080, 2400

        def screencap(self):
            return self.state.encode()

    class _Driver:
        dwell_s = 0.0

        def __init__(self):
            self.adb = _Adb()
            self.calls = []

        def _swipe(self, *args):
            self.calls.append(("swipe", args))
            assert self.adb.state == "focused"
            self.adb.state = "composer-offscreen"

        def _await_button(self, which):
            self.calls.append(("await", which))
            assert which == "pass" and self.adb.state == "composer-offscreen"
            return 130, 2040

        def _locate_button(self, which):
            self.calls.append(("locate", which))
            assert which == "pass" and self.adb.state == "composer-offscreen"
            return 133, 2041

        def _tap(self, *point):
            self.calls.append(("tap", point))
            assert point == (133, 2041)
            self.adb.state = "new-top"

        def _observe_deck_ready(self, frame):
            return frame == b"new-top"

        def dislike(self):
            pytest.fail("calibration composer Pass must not weaken/delegate to HingeDriver.dislike")

    driver = _Driver()
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(
        cal, "locate_inline_composer",
        lambda frame, *_args, **_kw: (
            focused if frame == b"focused"
            else (_ for _ in ()).throw(ComposerDetectionError("composer absent"))))
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_args, **_kw: SimpleNamespace(matched=True, reason="matched"))
    monkeypatch.setattr(
        cal, "confirm_scroll_top",
        lambda frame, **_kw: SimpleNamespace(confirmed=frame == b"new-top",
                                               refuted=frame != b"new-top", reason="top"))
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_args, **_kw: (7,))

    advance, trace = cal._automated_pass_from_verified_composer(
        driver, frame=b"focused", confirm_template=object(), payload=SimpleNamespace(),
        item_number=item_number, identity=identity, identity_band=_IDENTITY_BAND)

    assert advance == b"new-top"
    # A focused composer first gets a read-only Pass-locator probe; its absence then takes the
    # one guarded edge-back route.  The probe is not a gesture and cannot hit Send Like.
    assert [name for name, _args in driver.calls] == ["locate", "swipe", "await", "locate", "tap"]
    assert trace["transport"] == ["HingeDriver._swipe(android_edge_back)",
                                  "HingeDriver._await_button(pass)",
                                  "HingeDriver._locate_button(pass)", "HingeDriver._tap"]
    assert trace["predicates"]["post_edge_same_profile_sticky_header_verified"] is True
    assert trace["predicates"]["post_edge_composer_reverified"] is False
    assert trace["predicates"]["post_edge_composer_visibility"] == "not_required_may_be_offscreen"
    assert trace["predicates"]["new_profile_top_confirmed"] is True
    assert trace["pre_frame_sha256"] == hashlib.sha256(b"focused").hexdigest()
    assert trace["post_frame_sha256"] == hashlib.sha256(b"new-top").hexdigest()


def test_automated_send_from_verified_composer_uses_fresh_review_binding_and_production_transport(
        monkeypatch):
    """The owner-opted-in real-send path must use the exact same production send transport as an
    ordinary AUTO/OBSERVE like -- never a raw tap, and never the paid Rose/upsell control."""
    surface = ComposerSurface(cal._COMPOSER_LAYOUT_ID, Rect(80, 1150, 1000, 1290),
                              Rect(390, 1300, 985, 1420), (690, 1360))

    class _Adb:
        def __init__(self):
            self.frames = iter((b"composer", b"advance-frame"))

        def screencap(self):
            return next(self.frames)

    class _Driver:
        def __init__(self):
            self.halt_on_error = True
            self.adb = _Adb()
            self.calls = []

        def _tap(self, *point):
            self.calls.append(("tap", point))

        def _handle_rose_upsell(self):
            self.calls.append(("rose_upsell",))

        def _verify_like_landed(self, before):
            self.calls.append(("verify_landed", before))

    driver = _Driver()
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_cooldown", lambda _s: 0.0)
    monkeypatch.setattr(cal, "locate_inline_composer", lambda *_a, **_kw: surface)
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_a, **_kw: SimpleNamespace(matched=True, reason="matched"))

    advance_frame, trace = cal._automated_send_from_verified_composer(
        driver, frame=b"composer", confirm_template=object(), payload=SimpleNamespace(),
        item_number=3, reviewed_confirm_point=surface.confirm_point)

    assert advance_frame == b"advance-frame"
    assert [name for name, *_rest in driver.calls] == ["tap", "rose_upsell", "verify_landed"]
    assert driver.calls[0][1] == surface.confirm_point
    assert driver.calls[2][1] == b"composer"
    assert trace["action"] == "automated_send_priority_like"
    assert trace["transport"] == ["HingeDriver._tap(confirm_point)", "HingeDriver._handle_rose_upsell",
                                  "HingeDriver._verify_like_landed"]
    assert trace["send_like_tapped"] is True
    assert trace["confirm_point"] == list(surface.confirm_point)
    assert trace["pre_frame_sha256"] == hashlib.sha256(b"composer").hexdigest()
    assert trace["pre_tap_frame_sha256"] == hashlib.sha256(b"composer").hexdigest()
    assert trace["post_frame_sha256"] == hashlib.sha256(b"advance-frame").hexdigest()
    assert trace["predicates"] == {
        "inline_composer_and_selected_photo_verified_before_action": True,
        "fresh_composer_and_selected_photo_reverified_before_action": True,
        "send_like_tapped": True,
        "like_landed_verified": True,
    }


@pytest.mark.parametrize(
    ("halt_on_error", "fresh_frame", "fresh_point"),
    [
        (False, b"composer", (690, 1360)),
        (True, b"changed-composer", (690, 1360)),
        (True, b"composer", (691, 1360)),
    ],
    ids=["landed-proof-disabled", "frame-changed", "point-changed"],
)
def test_automated_send_refuses_stale_or_unverifiable_review_with_zero_taps(
        monkeypatch, halt_on_error, fresh_frame, fresh_point):
    reviewed_surface = ComposerSurface(
        cal._COMPOSER_LAYOUT_ID, Rect(80, 1150, 1000, 1290),
        Rect(390, 1300, 985, 1420), (690, 1360))
    fresh_surface = ComposerSurface(
        cal._COMPOSER_LAYOUT_ID, Rect(80, 1150, 1000, 1290),
        Rect(390, 1300, 985, 1420), fresh_point)

    class _Driver:
        def __init__(self):
            self.halt_on_error = halt_on_error
            self.adb = SimpleNamespace(screencap=lambda: fresh_frame)
            self.taps = []

        def _tap(self, *point):
            self.taps.append(point)

    driver = _Driver()
    surfaces = iter((reviewed_surface, fresh_surface))
    monkeypatch.setattr(cal, "locate_inline_composer", lambda *_a, **_kw: next(surfaces))
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_a, **_kw: SimpleNamespace(matched=True, reason="matched"))

    with pytest.raises(cal._CaptureAbort):
        cal._automated_send_from_verified_composer(
            driver, frame=b"composer", confirm_template=object(), payload=SimpleNamespace(),
            item_number=3, reviewed_confirm_point=reviewed_surface.confirm_point)

    assert driver.taps == []


def test_automated_send_allows_only_a_reverified_empty_comment_caret_blink(monkeypatch):
    """A focused composer may blink its caret after approval without moving any action surface."""
    size = (1080, 2400)
    reviewed_image = np.full((size[1], size[0], 3), 245, np.uint8)
    fresh_image = reviewed_image.copy()
    # The Android clock/network strip may also tick while the reviewer is deciding. It is not
    # an app surface and cannot move or relabel the Hinge action below it.
    fresh_image[30:70, 100:190] = 25
    # Exactly the thin vertical empty-field caret.  Any change outside this box is refused below.
    fresh_image[990:1042, 139:142] = 25
    ok, reviewed_buf = cv2.imencode(".png", reviewed_image)
    assert ok
    ok, fresh_buf = cv2.imencode(".png", fresh_image)
    assert ok
    reviewed_frame, fresh_frame = reviewed_buf.tobytes(), fresh_buf.tobytes()
    surface = ComposerSurface(cal._COMPOSER_LAYOUT_ID, Rect(95, 935, 985, 1108),
                              Rect(390, 1140, 985, 1260), (690, 1190))

    class _Adb:
        def __init__(self):
            # The first fresh read sees the opposite caret phase; the bounded second read proves
            # that exact reviewed pixels return before Send, then the final read is post-action.
            self.frames = iter((fresh_frame, reviewed_frame, b"advance-frame"))

        def screencap(self):
            return next(self.frames)

    class _Driver:
        halt_on_error = True

        def __init__(self):
            self.adb = _Adb()
            self.taps = []
            self.landed_before = None

        def _tap(self, *point):
            self.taps.append(point)

        def _handle_rose_upsell(self):
            return None

        def _verify_like_landed(self, before):
            self.landed_before = before

    driver = _Driver()
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_cooldown", lambda _s: 0.0)
    monkeypatch.setattr(cal, "locate_inline_composer", lambda *_a, **_kw: surface)
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_a, **_kw: SimpleNamespace(matched=True, reason="matched"))

    _advance, trace = cal._automated_send_from_verified_composer(
        driver, frame=reviewed_frame, confirm_template=object(), payload=SimpleNamespace(),
        item_number=1, reviewed_confirm_point=surface.confirm_point)

    assert driver.taps == [surface.confirm_point]
    assert driver.landed_before == reviewed_frame
    assert trace["pre_tap_frame_sha256"] == hashlib.sha256(reviewed_frame).hexdigest()


def test_automated_send_refuses_a_persistent_caret_shaped_mark(monkeypatch):
    """A narrow glyph can resemble a caret geometrically but cannot alternate like one."""
    reviewed_image = np.full((2400, 1080, 3), 245, np.uint8)
    marked_image = reviewed_image.copy()
    marked_image[990:1042, 139:142] = 25
    ok, reviewed_buf = cv2.imencode(".png", reviewed_image)
    assert ok
    ok, marked_buf = cv2.imencode(".png", marked_image)
    assert ok
    reviewed_frame, marked_frame = reviewed_buf.tobytes(), marked_buf.tobytes()
    surface = ComposerSurface(cal._COMPOSER_LAYOUT_ID, Rect(95, 935, 985, 1108),
                              Rect(390, 1140, 985, 1260), (690, 1190))

    class _Driver:
        halt_on_error = True

        def __init__(self):
            self.adb = SimpleNamespace(screencap=lambda: marked_frame)
            self.taps = []

        def _tap(self, *point):
            self.taps.append(point)

    driver = _Driver()
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "locate_inline_composer", lambda *_a, **_kw: surface)
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_a, **_kw: SimpleNamespace(matched=True, reason="matched"))

    with pytest.raises(cal._CaptureAbort, match="did not return to the exact reviewed blink phase"):
        cal._automated_send_from_verified_composer(
            driver, frame=reviewed_frame, confirm_template=object(), payload=SimpleNamespace(),
            item_number=1, reviewed_confirm_point=surface.confirm_point)

    assert driver.taps == []


def test_caret_screen_rejects_an_adjacent_narrow_mark():
    reviewed = np.full((2400, 1080, 3), 245, np.uint8)
    adjacent = reviewed.copy()
    # The previous +16..+72 x-window accepted this second vertical mark beside the live caret.
    adjacent[990:1039, 144:149] = 25
    ok, reviewed_buf = cv2.imencode(".png", reviewed)
    assert ok
    ok, adjacent_buf = cv2.imencode(".png", adjacent)
    assert ok
    surface = ComposerSurface(cal._COMPOSER_LAYOUT_ID, Rect(95, 935, 985, 1108),
                              Rect(390, 1140, 985, 1260), (690, 1190))

    assert cal._only_transient_empty_comment_caret_change(
        reviewed_buf.tobytes(), adjacent_buf.tobytes(), surface=surface) is False


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        ((1530, 1542), (30, 60)),  # Gboard changed
        ((990, 1042), (180, 183)),  # text beyond the sole caret position
        ((930, 938), (95, 985)),  # comment-outline/layout change
    ],
    ids=["keyboard", "typed-text-position", "comment-border"],
)
def test_automated_send_refuses_non_caret_frame_changes_after_fresh_reverification(
        monkeypatch, rows, cols):
    size = (1080, 2400)
    reviewed_image = np.full((size[1], size[0], 3), 245, np.uint8)
    fresh_image = reviewed_image.copy()
    fresh_image[rows[0]:rows[1], cols[0]:cols[1]] = 25
    ok, reviewed_buf = cv2.imencode(".png", reviewed_image)
    assert ok
    ok, fresh_buf = cv2.imencode(".png", fresh_image)
    assert ok
    reviewed_frame, fresh_frame = reviewed_buf.tobytes(), fresh_buf.tobytes()
    surface = ComposerSurface(cal._COMPOSER_LAYOUT_ID, Rect(95, 935, 985, 1108),
                              Rect(390, 1140, 985, 1260), (690, 1190))

    class _Driver:
        halt_on_error = True

        def __init__(self):
            self.adb = SimpleNamespace(screencap=lambda: fresh_frame)
            self.taps = []

        def _tap(self, *point):
            self.taps.append(point)

    driver = _Driver()
    monkeypatch.setattr(cal, "locate_inline_composer", lambda *_a, **_kw: surface)
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_a, **_kw: SimpleNamespace(matched=True, reason="matched"))

    with pytest.raises(cal._CaptureAbort, match="outside the empty-comment caret blink"):
        cal._automated_send_from_verified_composer(
            driver, frame=reviewed_frame, confirm_template=object(), payload=SimpleNamespace(),
            item_number=1, reviewed_confirm_point=surface.confirm_point)

    assert driver.taps == []


def test_abort_cleanup_discards_structural_composer_when_item_verification_refuses(monkeypatch):
    """A false-negative item comparison must not strand an unsent composer or tap Send Like."""
    surface = ComposerSurface(cal._COMPOSER_LAYOUT_ID, Rect(80, 1150, 1000, 1290),
                              Rect(390, 1300, 985, 1420), (690, 1360))
    identity = ProfileIdentity((7,), _IDENTITY_BAND, (1, 1), 1, 20.0, "known",
                               agreeing_frames=2)

    class _Adb:
        def __init__(self): self.state = "composer"
        def screen_size(self): return 1080, 2400
        def screencap(self): return self.state.encode()

    class _Driver:
        dwell_s = 0.0
        def __init__(self): self.adb, self.calls = _Adb(), []
        def _swipe(self, *_args):
            self.calls.append("edge_back")
            self.adb.state = "composer-offscreen"
        def _await_button(self, which):
            self.calls.append(("await", which))
            return 130, 2040
        def _locate_button(self, which):
            self.calls.append(("locate", which))
            return 133, 2041
        def _tap(self, *point):
            self.calls.append(("tap", point))
            self.adb.state = "new-top"
        def _observe_deck_ready(self, frame): return frame == b"new-top"

    driver = _Driver()
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(
        cal, "locate_inline_composer",
        lambda frame, *_a, **_kw: surface if frame == b"composer" else (
            (_ for _ in ()).throw(ComposerDetectionError("composer offscreen"))))
    # This is precisely the incident path: Hinge rendered an inline composer, but the item's
    # bounded image comparison refused.  The cleanup may discard, never certify, this profile.
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_a, **_kw: SimpleNamespace(matched=False, reason="render drift"))
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda frame, **_kw: SimpleNamespace(confirmed=frame == b"new-top",
                                                              refuted=frame != b"new-top", reason="top"))
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_a, **_kw: (7,))

    recovery = cal._recover_automated_abort_from_open_composer(
        driver, frame=b"composer", confirm_template=object(), payload=SimpleNamespace(),
        item_number=1, identity=identity, identity_band=_IDENTITY_BAND,
        failure_stage="post_tap_composer_or_item_verification_refused")

    assert recovery["outcome"] == "cleared"
    assert recovery["calibration_evidence"] is False
    assert recovery["selected_item_verified_before_cleanup"] is False
    assert recovery["send_like_tapped"] is False
    # CRITICAL: the cleanup route is always Pass, by construction -- never Send Like, even for a
    # profile that would otherwise have ended with an accepted real send.
    assert recovery["cleanup_trace"]["action"] == "automated_pass"
    assert recovery["cleanup_trace"]["predicates"]["send_like_tapped"] is False
    assert recovery["cleanup_trace"]["predicates"]["inline_composer_and_selected_photo_verified_before_action"] is False
    # The composer is already unfocused and exposes a stable floating Pass.  Cleanup must not
    # issue Android back here: on the live surface that can leave Hinge for the launcher.
    assert driver.calls == [("locate", "pass"), ("locate", "pass"), ("tap", (133, 2041))]
    assert recovery["cleanup_trace"]["predicates"]["edge_back_transport_performed"] is False
    assert recovery["cleanup_trace"]["predicates"]["unfocused_composer_pass_visible_before_edge_back"] is True


def test_abort_cleanup_never_threads_or_calls_the_send_transport():
    """CRITICAL invariant: abort cleanup is Pass-only by construction, never by a runtime flag.

    `_recover_automated_abort_from_open_composer` has no `send_like` parameter at all, and the
    capture loop's own `recover_unsent_composer` closure never threads one through either -- so a
    stranded composer is always cleared with the calibration-only Pass route, even during a
    `--send-like` session. This is a structural guarantee, not merely a tested behavior: assert it
    directly so a future refactor cannot quietly grow a send path here.
    """
    sig = inspect.signature(cal._recover_automated_abort_from_open_composer)
    assert "send_like" not in sig.parameters

    recovery_source = inspect.getsource(cal._recover_automated_abort_from_open_composer)
    for forbidden in ("_automated_send_from_verified_composer", "_handle_rose_upsell",
                      "_verify_like_landed"):
        assert forbidden not in recovery_source
    # `send_like_tapped` (a recorded, always-False fact) legitimately contains this substring;
    # only the bare `send_like` identifier (a flag/parameter name) must never appear.
    assert not re.search(r"\bsend_like\b", recovery_source)

    capture_source = inspect.getsource(cal._capture_one_profile_unattended)
    closure_start = capture_source.index("def recover_unsent_composer")
    closure_end = capture_source.index("\n    def ", closure_start + 1)
    assert not re.search(r"\bsend_like\b", capture_source[closure_start:closure_end])


def test_abort_cleanup_refuses_without_an_current_structural_composer(monkeypatch):
    """A remembered earlier composer can never authorize an abort-time gesture."""
    class _Driver:
        def __init__(self): self.touches = 0
        def _swipe(self, *_args): self.touches += 1

    monkeypatch.setattr(cal, "locate_inline_composer",
                        lambda *_a, **_kw: (_ for _ in ()).throw(
                            ComposerDetectionError("not a composer")))
    recovery = cal._recover_automated_abort_from_open_composer(
        _Driver(), frame=b"unknown", confirm_template=object(), payload=SimpleNamespace(),
        item_number=3,
        identity=ProfileIdentity((7,), _IDENTITY_BAND, (1, 1), 1, 20.0, "known",
                                 agreeing_frames=2),
        identity_band=_IDENTITY_BAND, failure_stage="post_tap_composer_or_item_verification_refused")
    assert recovery["outcome"] == "not_cleared"
    assert "no supported inline composer" in recovery["refusal"]


def test_abort_cleanup_refuses_wrong_identity_before_any_edge_or_pass_gesture(monkeypatch):
    surface = ComposerSurface(cal._COMPOSER_LAYOUT_ID, Rect(80, 1150, 1000, 1290),
                              Rect(390, 1300, 985, 1420), (690, 1360))
    identity = ProfileIdentity((7,), _IDENTITY_BAND, (1, 1), 1, 20.0, "known",
                               agreeing_frames=2)

    class _Adb:
        def screen_size(self): return 1080, 2400
        def screencap(self): return b"wrong-profile"

    class _Driver:
        dwell_s = 0.0
        def __init__(self): self.adb, self.touches = _Adb(), []
        def _swipe(self, *_args): self.touches.append("edge_back")
        def _locate_button(self, *_args): self.touches.append("locate")
        def _tap(self, *_args): self.touches.append("tap")

    driver = _Driver()
    monkeypatch.setattr(cal, "locate_inline_composer", lambda *_a, **_kw: surface)
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_a, **_kw: SimpleNamespace(matched=False, reason="drift"))
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_a, **_kw: _top_verdict("confirmed_not_top", "sticky"))
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_a, **_kw: (99,))

    recovery = cal._recover_automated_abort_from_open_composer(
        driver, frame=b"wrong-profile", confirm_template=object(), payload=SimpleNamespace(),
        item_number=1, identity=identity, identity_band=_IDENTITY_BAND,
        failure_stage="post_tap_composer_or_item_verification_refused")

    assert recovery["outcome"] == "not_cleared"
    assert "different or unstable profile" in recovery["refusal"]
    assert driver.touches == []


def test_post_pass_settle_accepts_a_ready_top_without_an_extra_edge_back(monkeypatch):
    class _Adb:
        def screen_size(self): return 1080, 2400
        def screencap(self): return b"top"

    class _Driver:
        dwell_s = 0.0
        adb = _Adb()
        def _observe_deck_ready(self, frame): return frame == b"top"
        def _swipe(self, *_args): pytest.fail("valid ordinary top needs no modal recovery")

    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_a, **_kw: _top_verdict("confirmed_top"))
    monkeypatch.setattr(cal, "locate_inline_composer",
                        lambda *_a, **_kw: (_ for _ in ()).throw(ComposerDetectionError("absent")))
    settled, trace = cal._settle_automated_post_pass_to_top(
        _Driver(), frame=b"top", confirm_template=object(), identity_band=_IDENTITY_BAND)
    assert settled == b"top"
    assert trace["modal_edge_back_used"] is False


def test_post_pass_settle_reprobes_transient_until_an_ordinary_top_without_a_swipe(monkeypatch):
    """A loading post-Pass frame can settle to the next deck without modal recovery."""
    class _Adb:
        def __init__(self): self.frames = iter((b"settled-top",))
        def screen_size(self): return 1080, 2400
        def screencap(self): return next(self.frames)

    class _Driver:
        dwell_s = 0.0
        def __init__(self): self.adb, self.swipes = _Adb(), 0
        def _observe_deck_ready(self, frame): return frame == b"settled-top"
        def _swipe(self, *_args): self.swipes += 1

    driver, composer_candidates = _Driver(), []
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(
        cal, "confirm_scroll_top",
        lambda frame, **_kw: _top_verdict(
            "cannot_tell" if frame == b"loading" else "confirmed_top", "settled"))
    monkeypatch.setattr(
        cal, "locate_inline_composer",
        lambda frame, *_a, **_kw: composer_candidates.append(frame)
        or (_ for _ in ()).throw(ComposerDetectionError("absent")))

    settled, trace = cal._settle_automated_post_pass_to_top(
        driver, frame=b"loading", confirm_template=object(), identity_band=_IDENTITY_BAND)

    assert settled == b"settled-top"
    assert driver.swipes == 0
    assert composer_candidates == [b"loading", b"settled-top"]
    assert trace["modal_edge_back_used"] is False


def test_post_pass_settle_preserves_refuted_sticky_header_without_a_swipe(monkeypatch):
    """A real scrolled/sticky state is never treated as a dismissible modal."""
    class _Adb:
        def screen_size(self): return 1080, 2400
        def screencap(self): return b"sticky-header"

    class _Driver:
        dwell_s = 0.0
        def __init__(self): self.adb, self.swipes = _Adb(), 0
        def _observe_deck_ready(self, _frame): pytest.fail("a refuted top must not query deck readiness")
        def _swipe(self, *_args): self.swipes += 1

    driver = _Driver()
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_a, **_kw: _top_verdict("confirmed_not_top", "sticky header"))
    monkeypatch.setattr(cal, "locate_inline_composer",
                        lambda *_a, **_kw: (_ for _ in ()).throw(ComposerDetectionError("absent")))

    with pytest.raises(cal._CaptureAbort,
                       match="preserved state: scroll-top remained refuted/scrolled"):
        cal._settle_automated_post_pass_to_top(
            driver, frame=b"sticky-header", confirm_template=object(),
            identity_band=_IDENTITY_BAND)
    assert driver.swipes == 0


def test_post_pass_settle_uses_one_edge_back_for_top_looking_promo_then_requires_deck(monkeypatch):
    class _Adb:
        state = b"promo"
        def screen_size(self): return 1080, 2400
        def screencap(self): return self.state

    class _Driver:
        dwell_s = 0.0
        def __init__(self): self.adb, self.swipes = _Adb(), 0
        def _observe_deck_ready(self, frame): return frame == b"new-top"
        def _swipe(self, *_args):
            self.swipes += 1
            self.adb.state = b"new-top"

    driver = _Driver()
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_a, **_kw: _top_verdict("confirmed_top"))
    monkeypatch.setattr(cal, "locate_inline_composer",
                        lambda *_a, **_kw: (_ for _ in ()).throw(ComposerDetectionError("absent")))
    settled, trace = cal._settle_automated_post_pass_to_top(
        driver, frame=b"promo", confirm_template=object(), identity_band=_IDENTITY_BAND)
    assert settled == b"new-top"
    assert driver.swipes == 1
    assert trace["modal_edge_back_used"] is True


def test_post_pass_settle_refuses_after_one_edge_back_when_ordinary_deck_is_still_missing(monkeypatch):
    class _Adb:
        state = b"promo"
        def screen_size(self): return 1080, 2400
        def screencap(self): return self.state

    class _Driver:
        dwell_s = 0.0
        def __init__(self): self.adb, self.swipes = _Adb(), 0
        def _observe_deck_ready(self, _frame): return False
        def _swipe(self, *_args):
            self.swipes += 1
            self.adb.state = b"still-promo"

    driver = _Driver()
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_a, **_kw: _top_verdict("confirmed_top"))
    monkeypatch.setattr(cal, "locate_inline_composer",
                        lambda *_a, **_kw: (_ for _ in ()).throw(ComposerDetectionError("absent")))
    with pytest.raises(cal._CaptureAbort, match="ordinary new profile top"):
        cal._settle_automated_post_pass_to_top(
            driver, frame=b"promo", confirm_template=object(), identity_band=_IDENTITY_BAND)
    assert driver.swipes == 1


# The exact diagnosis a live run produced twice in a row.  Its shape is the point: the code is
# a class of failure, and everything that says WHICH failure lives in the detail.
_LIVE_SKIP_CODE = "target_unavailable_or_incomplete_index"
_LIVE_SKIP_DETAIL = ("did not obtain a safe target-scoped photo prefix for (1,) within 40 bounded "
                     "read scrolls (item 1 has an unresolved predecessor)")


def _preaction_skip_fixtures(monkeypatch):
    """The minimum world one pre-action skip needs: a public `dislike` that advances the deck, a
    confirmed top before and after, and a distinct sticky identity on the next profile.  Shared
    by the transport test and the diagnosis tests so all three exercise one identical skip."""
    identity = ProfileIdentity((7,), _IDENTITY_BAND, (1, 1), 1, 20.0, "known",
                               agreeing_frames=2)

    class _Adb:
        state = b"top"
        def screencap(self): return self.state

    class _Driver:
        dwell_s = 0.0
        def __init__(self): self.adb, self.dislikes, self.likes, self.taps = _Adb(), 0, 0, 0
        def _template(self, _name): return object()
        def dislike(self):
            self.dislikes += 1
            self.adb.state = b"advanced-top"
        def like(self):
            self.likes += 1
            self.adb.state = b"advanced-top"
        def _scroll_down_one(self, *_args): self.adb.state = b"next-sticky"
        def _tap(self, *_args): self.taps += 1

    driver = _Driver()
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "_rewind_automated_profile_to_confirmed_top",
                        lambda *_a, **_kw: b"top")
    monkeypatch.setattr(cal, "locate_inline_composer",
                        lambda *_a, **_kw: (_ for _ in ()).throw(ComposerDetectionError("absent")))
    monkeypatch.setattr(cal, "_settle_automated_post_pass_to_top",
                        lambda *_a, **_kw: (b"advanced-top", {
                            "modal_edge_back_used": False, "ordinary_deck_ready": True}))
    monkeypatch.setattr(cal, "_plan_card_scroll",
                        lambda *_a, **_kw: (SimpleNamespace(frac=.1, x_frac=.5), None))
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda frame, **_kw: _top_verdict(
                            "confirmed_not_top" if frame == b"next-sticky" else "confirmed_top"))
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_a, **_kw: (99,))
    monkeypatch.setattr(cal, "fingerprint_distance", lambda *_a, **_kw: 20.0)
    return driver, identity


def test_preaction_skip_uses_public_dislike_and_records_distinct_retry(monkeypatch):
    driver, identity = _preaction_skip_fixtures(monkeypatch)

    record = cal._skip_automated_profile_before_heart(
        driver, ordinal=2,
        reason=cal._PreActionProfileRetry("pre_heart_navigation_refused", "navigation refused"),
        identity=identity, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=.8)
    assert driver.dislikes == 1
    assert driver.taps == 0
    assert record["transport"] == "HingeDriver.dislike"
    assert record["predicates"]["new_profile_identity_distinct"] is True
    assert record["automated_sticky_header_proof"]["read_only_reprobe_count"] == 0
    assert record["automated_sticky_header_proof"]["identity_distance_samples"][0][
        "distance_from_prior"] == 20.0


def test_send_like_run_advances_unusable_profile_with_public_like_not_dislike(monkeypatch):
    """Owner rule: even a profile excluded from calibration evidence advances by Like."""
    driver, identity = _preaction_skip_fixtures(monkeypatch)
    gate = _RecordingGate()

    record = cal._skip_automated_profile_before_heart(
        driver, ordinal=2,
        reason=cal._PreActionProfileRetry("pre_heart_navigation_refused", "navigation refused"),
        identity=identity, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=.8, review_gate=gate, send_like=True)

    assert driver.likes == 1
    assert driver.dislikes == 0
    assert record["action"] == "advance_unusable_profile_with_priority_like"
    assert record["transport"] == "HingeDriver.like"
    assert record["predicates"]["send_like_tapped"] is True
    (_state, plan), = gate.checkpoints
    assert plan["action"] == "advance_unusable_profile_with_priority_like"
    assert plan["point_source"] == "public HingeDriver.like"
    assert plan["predicates"]["send_like_requested_for_unusable_profile"] is True


def test_preaction_skip_record_carries_the_plaintext_diagnosis_its_own_digest_binds(monkeypatch):
    """A skip record used to keep the human-actionable diagnosis ONLY as `reason_sha256`, which
    is unreadable by the operator who has to fix the cause -- a live run produced two skips in a
    row that were indistinguishable because just the generic code was legible.  The digest covers
    `code\ndetail`, so shipping the detail beside it makes the record self-verifying (recompute
    and compare) rather than weaker; this test pins that exact reproduction."""
    driver, identity = _preaction_skip_fixtures(monkeypatch)

    record = cal._skip_automated_profile_before_heart(
        driver, ordinal=2,
        reason=cal._PreActionProfileRetry(_LIVE_SKIP_CODE, _LIVE_SKIP_DETAIL),
        identity=identity, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=.8)

    assert record["reason_code"] == _LIVE_SKIP_CODE
    assert record["reason_detail"] == _LIVE_SKIP_DETAIL
    assert record["reason_sha256"] == hashlib.sha256(
        f"{record['reason_code']}\n{record['reason_detail']}".encode("utf-8")).hexdigest()


def test_preaction_skip_checkpoint_tells_the_reviewer_why_the_pass_is_being_requested(monkeypatch):
    """The reviewer approving this checkpoint is approving a real Pass on a real person.  A bare
    `skip_reason_code` names a class of failure, never which one occurred, so the approval would
    be made blind.  The detail is added before the checkpoint is hashed, so it is covered by the
    same `evidence_sha256` as every other published predicate."""
    driver, identity = _preaction_skip_fixtures(monkeypatch)
    gate = _RecordingGate()

    cal._skip_automated_profile_before_heart(
        driver, ordinal=2,
        reason=cal._PreActionProfileRetry(_LIVE_SKIP_CODE, _LIVE_SKIP_DETAIL),
        identity=identity, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
        like_template=object(), like_threshold=.8, review_gate=gate)

    (claimed_state, plan), = gate.checkpoints
    assert claimed_state == "pre_action_profile_skip_ready"
    assert plan["action"] == "skip_profile_without_heart"
    assert plan["predicates"]["skip_reason_code"] == _LIVE_SKIP_CODE
    assert plan["predicates"]["skip_reason_detail"] == _LIVE_SKIP_DETAIL


def _wire_inert_skip_capture(monkeypatch, tmp_path):
    """`_cmd_capture`'s outer shell only: config, driver, serial and out-dir, with no real card
    scan.  The profile loop itself is driven by the caller's stubbed capture function."""
    class _Adb:
        def shell(self, cmd):
            if "ro.product.model" in cmd:
                return "Pixel 7a"
            if "wm density" in cmd:
                return "420"
            return "versionName=1.0\n"

        def screen_size(self):
            return 1080, 2400

    class _Driver:
        def __init__(self, _cfg):
            self.identity_band = _IDENTITY_BAND
            self.content_band = _CONTENT_BAND
            self.serial = "PIXEL-TEST"
            self.package = "co.hinge.app"
            self.adb = _Adb()

        def _template(self, _name): return object()
        def open_session(self): pass
        def close(self): pass

    monkeypatch.setattr(cal.cfg_mod, "load", lambda _path: SimpleNamespace(apps={}))
    # The shell now validates before touching the device (the validate call is what installs
    # the process-local still-photo licence); this harness is about the loop's console/manifest
    # wiring, so validation is stubbed inert like every other collaborator here.
    monkeypatch.setattr(cal.cfg_mod, "validate", lambda _cfg: None)
    monkeypatch.setattr(cal, "HingeDriver", _Driver)
    monkeypatch.setattr(cal, "_preflight_serial", lambda _cfg: ("PIXEL-TEST", "adb"))
    monkeypatch.setattr(cal, "_capture_out_dir", lambda *_a, **_kw: tmp_path)


def test_capture_reenumerates_one_large_entry_drift_then_skips_on_recurrence(
        monkeypatch, tmp_path):
    """The outer loop grants one fresh scan for the same ordinal, never an unbounded loop."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("apps:\n  hinge:\n    serial: PIXEL-TEST\n")
    _wire_inert_skip_capture(monkeypatch, tmp_path)
    restart_counts = []

    def capture_once_per_state(_driver, _out_dir, *, ordinal, frame_counter, frames_meta,
                               used_profile_ids, **kwargs):
        restart_counts.append(kwargs.get("entry_drift_restart_attempts", 0))
        if restart_counts == [0]:
            raise cal._RestartProfile("first measured entry drift")
        if restart_counts == [0, 1]:
            raise cal._ProfileSkipped({
                "action": "skip_profile_without_heart", "ordinal": ordinal,
                "reason_code": "pre_heart_navigation_refused",
                "reason_detail": "large entry drift recurred after fresh scan",
                "reason_sha256": hashlib.sha256(
                    b"pre_heart_navigation_refused\nlarge entry drift recurred after fresh scan"
                ).hexdigest(),
            })
        used_profile_ids.add("fresh-profile")
        return ({"ordinal": ordinal, "profile_id": "fresh-profile", "composer_items": [1],
                 "target_strategy_id": kwargs["target_strategy_id"]}, frame_counter)

    monkeypatch.setattr(cal, "_capture_one_profile_unattended", capture_once_per_state)
    args = argparse.Namespace(
        profiles=1, split="calibration", config=str(config_path), out=str(tmp_path),
        unattended=True, hybrid_review=False, confirmation=cal._UNATTENDED_CONFIRMATION,
        record_operational_checks=False, send_like=False, send_like_confirmation="")

    cal._cmd_capture(args)

    assert restart_counts == [0, 1, 0]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["skipped_attempts"]) == 1
    assert manifest["profiles"] == [{"ordinal": 1, "profile_id": "fresh-profile",
                                      "composer_items": [1],
                                      "target_strategy_id": cal._AUTOMATED_TARGET_STRATEGY_ID}]


def test_capture_console_names_the_skip_diagnosis_not_only_its_generic_code(
        monkeypatch, tmp_path, capsys):
    """The operator watching an automated run is the person who decides whether the deck, the
    target depth, or the app build is at fault, and the console is the only place they see a
    skip at all.  Printing the code alone is what left a live run blind through two consecutive
    skips, so the detail must appear in the line, not merely in the manifest."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("apps:\n  hinge:\n    serial: PIXEL-TEST\n")
    _wire_inert_skip_capture(monkeypatch, tmp_path)
    attempts = []

    def _skip_then_stop(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise cal._ProfileSkipped({
                "action": "skip_profile_without_heart", "ordinal": 1,
                "reason_code": _LIVE_SKIP_CODE, "reason_detail": _LIVE_SKIP_DETAIL,
                "reason_sha256": hashlib.sha256(
                    f"{_LIVE_SKIP_CODE}\n{_LIVE_SKIP_DETAIL}".encode("utf-8")).hexdigest()})
        raise cal._CaptureAbort("stop-here")

    monkeypatch.setattr(cal, "_capture_one_profile_unattended", _skip_then_stop)
    args = argparse.Namespace(
        profiles=1, split="calibration", config=str(config_path), out=str(tmp_path),
        unattended=True, hybrid_review=False, confirmation=cal._UNATTENDED_CONFIRMATION,
        record_operational_checks=False, send_like=False, send_like_confirmation="")

    with pytest.raises(SystemExit):
        cal._cmd_capture(args)

    out = capsys.readouterr().out
    assert _LIVE_SKIP_DETAIL in out
    assert f"({_LIVE_SKIP_CODE}: {_LIVE_SKIP_DETAIL})" in out
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["skipped_attempts"][0]["reason_detail"] == _LIVE_SKIP_DETAIL


def test_preaction_skip_reviewer_nonapproval_never_calls_public_dislike(monkeypatch):
    identity = ProfileIdentity((7,), _IDENTITY_BAND, (1, 1), 1, 20.0, "known",
                               agreeing_frames=2)

    class _Driver:
        dwell_s = 0.0
        def _template(self, _name): return object()
        def dislike(self): pytest.fail("non-approved checkpoint must not advance the profile")

    class _Gate:
        def checkpoint(self, *_args, **_kwargs): return {"decision": "retry"}

    monkeypatch.setattr(cal, "_rewind_automated_profile_to_confirmed_top",
                        lambda *_a, **_kw: b"top")
    monkeypatch.setattr(cal, "locate_inline_composer",
                        lambda *_a, **_kw: (_ for _ in ()).throw(ComposerDetectionError("absent")))
    with pytest.raises(cal._CaptureAbort, match="did not approve"):
        cal._skip_automated_profile_before_heart(
            _Driver(), ordinal=1,
            reason=cal._PreActionProfileRetry("target_unavailable_or_incomplete_index", "missing"),
            identity=identity, identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND,
            like_template=object(), like_threshold=.8, review_gate=_Gate())


def test_calibration_only_pass_refuses_a_wrong_profile_header_after_edge_back(monkeypatch):
    surface = ComposerSurface(cal._COMPOSER_LAYOUT_ID, Rect(80, 1150, 1000, 1290),
                              Rect(390, 1300, 985, 1420), (690, 1360))
    identity = ProfileIdentity((7,), _IDENTITY_BAND, (1, 1), 1, 20.0, "known",
                               agreeing_frames=2)

    class _Adb:
        state = "composer"

        def screen_size(self):
            return 1080, 2400

        def screencap(self):
            return self.state.encode()

    class _Driver:
        dwell_s = 0.0

        def __init__(self):
            self.adb = _Adb()

        def _swipe(self, *_args):
            self.adb.state = "wrong-header"

        def _await_button(self, _which):
            pytest.fail("Pass must not be located for a mismatched profile header")

    monkeypatch.setattr(cal, "locate_inline_composer", lambda *_args, **_kw: surface)
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_args, **_kw: SimpleNamespace(matched=True, reason="matched"))
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_args, **_kw: SimpleNamespace(confirmed=False, refuted=True,
                                                               reason="sticky"))
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_args, **_kw: (99,))

    with pytest.raises(cal._CaptureAbort, match="different or unstable profile header"):
        cal._automated_pass_from_verified_composer(
            _Driver(), frame=b"composer", confirm_template=object(), payload=SimpleNamespace(),
            item_number=3, identity=identity, identity_band=_IDENTITY_BAND)


def test_calibration_only_pass_refuses_when_pass_disappears_after_edge_back(monkeypatch):
    surface = ComposerSurface(cal._COMPOSER_LAYOUT_ID, Rect(80, 1150, 1000, 1290),
                              Rect(390, 1300, 985, 1420), (690, 1360))
    identity = ProfileIdentity((7,), _IDENTITY_BAND, (1, 1), 1, 20.0, "known",
                               agreeing_frames=2)

    class _Adb:
        state = "composer"

        def screen_size(self):
            return 1080, 2400

        def screencap(self):
            return self.state.encode()

    class _Driver:
        dwell_s = 0.0

        def __init__(self):
            self.adb = _Adb()
            self.tapped = False

        def _swipe(self, *_args):
            self.adb.state = "composer-offscreen"

        def _await_button(self, which):
            assert which == "pass"
            return 130, 2040

        def _locate_button(self, which):
            assert which == "pass"
            return None

        def _tap(self, *_args):
            self.tapped = True

    driver = _Driver()
    monkeypatch.setattr(cal, "locate_inline_composer",
                        lambda frame, *_args, **_kw: (
                            surface if frame == b"composer"
                            else (_ for _ in ()).throw(ComposerDetectionError("offscreen"))))
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_args, **_kw: SimpleNamespace(matched=True, reason="matched"))
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_delay", lambda _dwell: 0.0)
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_args, **_kw: SimpleNamespace(confirmed=False, refuted=True,
                                                               reason="sticky"))
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_args, **_kw: (7,))

    with pytest.raises(cal._CaptureAbort, match="Pass X disappeared"):
        cal._automated_pass_from_verified_composer(
            driver, frame=b"composer", confirm_template=object(), payload=SimpleNamespace(),
            item_number=3, identity=identity, identity_band=_IDENTITY_BAND)
    assert driver.tapped is False


def _successful_measure_seams(monkeypatch, tmp_path, *, held_identity=None,
                              held_inline_records=None, serial="PIXEL-TEST"):
    calibration = _session(tmp_path, "targeting_20260813T120000Z", "calibration", ["a", "b"],
                           serial=serial)
    heldout = _session(tmp_path, "targeting_20260813T130000Z", "heldout", ["c", "d"],
                       serial=serial)
    sessions = {"cal": calibration, "held": heldout}
    operational = _write_operational_evidence(tmp_path / "operational", device=_device(serial))
    _wire_operational_replay(monkeypatch, operational)
    for session in sessions.values():
        for key in ("item_1_inline_composer_identity", "gesture_transport"):
            session.manifest["operational_checks"][key]["evidence"] = str(operational)
    monkeypatch.setattr(cal.cfg_mod, "load",
                        lambda _path: SimpleNamespace(apps={"hinge": {"serial": serial}}))
    monkeypatch.setattr(cal, "HingeDriver", _InertDriver)
    monkeypatch.setattr(cal, "_load_session", lambda raw: sessions[str(raw)])
    # The measure-focused tests replace every heavyweight evidence pipeline at its boundary.
    # Preliminary observe-artifact validation has its own byte/hash tests in
    # test_hinge_observe_check.py; keep these seams about bound freezing/held-out behavior.
    monkeypatch.setattr(cal, "_preliminary_observe_check_reference_reason", lambda _ref: None)
    monkeypatch.setattr(cal, "_entry_anchor_reference_reason", lambda *_args, **_kw: None)
    monkeypatch.setattr(cal, "_validate_profile_advance_clears", lambda *_args, **_kw: None)
    monkeypatch.setattr(cal, "fingerprint_distance", lambda a, b: abs(a[0] - b[0]))

    calibration_identity = _samples("cal", [(20, 20.4), (22, 22.4)])
    held_identity = held_identity or _samples("held", [(0, 0.4), (3, 3.4)])
    monkeypatch.setattr(
        cal, "_profile_identity_samples",
        lambda profiles, **_kw: calibration_identity if profiles[0].profile_id == "a" else held_identity,
    )

    monkeypatch.setattr(cal, "_build_profile_payloads", lambda profiles, **_kw: profiles)
    calibration_inline = [_record("own_intended", 1), _record("own_intended", 2),
                         _record("own_foreign_item", 8)]
    held_inline_records = held_inline_records or [
        _record("own_intended", 1), _record("foreign_profile", 8)]
    monkeypatch.setattr(
        cal, "_inline_distance_records",
        lambda profiles, **_kw: calibration_inline if profiles[0].profile_id == "a" else held_inline_records,
    )
    return argparse.Namespace(config="does-not-exist.yaml", sessions=["cal", "held"]), calibration, heldout


def test_measure_requires_collective_photo_depth_coverage_for_multi_profile_split(tmp_path):
    profiles = [cal._ProfileData(i, f"p{i}", [], [(1, b"pre", b"composer")], b"clear", b"identity")
                for i in (1, 2)]

    with pytest.raises(cal._MeasureRefused, match="lacks composer evidence"):
        cal._require_collective_target_depths(profiles, split="calibration")


def test_require_collective_target_depths_accepts_photo_1_only_at_depth_1_across_two_profiles():
    """`photo_1_only_v1` never targets item 3 by design (owner decision 2026-08-22): two profiles
    that both only ever touch item 1 collectively cover everything that strategy claims to
    exercise, so the check must accept rather than demand a depth the strategy never collects."""
    profiles = [cal._ProfileData(i, f"p{i}", [], [(1, b"pre", b"composer")], b"clear", b"identity")
                for i in (1, 2)]

    cal._require_collective_target_depths(
        profiles, split="calibration", strategy_id=cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID)


def test_require_collective_target_depths_refuses_same_profiles_under_alternating_strategy():
    """The exact same depth-1-only profiles that pass under `photo_1_only_v1` must still be
    REFUSED under the alternating strategy, naming both the missing depth and the strategy that
    was asked for -- proving the check derives its requirement from the strategy rather than
    having simply gone slack for every strategy."""
    profiles = [cal._ProfileData(i, f"p{i}", [], [(1, b"pre", b"composer")], b"clear", b"identity")
                for i in (1, 2)]

    with pytest.raises(cal._MeasureRefused, match="lacks composer evidence") as exc:
        cal._require_collective_target_depths(
            profiles, split="calibration", strategy_id=cal._AUTOMATED_TARGET_STRATEGY_ID)
    assert "[3]" in str(exc.value)
    assert cal._AUTOMATED_TARGET_STRATEGY_ID in str(exc.value)


def test_require_collective_target_depths_accepts_alternating_strategy_with_both_depths_covered():
    """Unchanged regression: the default alternating strategy still accepts a split that
    collectively covers both photo-1 and photo-3, exactly as before this strategy-aware check."""
    profiles = [cal._ProfileData(1, "p1", [], [(1, b"pre", b"composer")], b"clear", b"identity"),
                cal._ProfileData(2, "p2", [], [(3, b"pre", b"composer")], b"clear", b"identity")]

    cal._require_collective_target_depths(
        profiles, split="calibration", strategy_id=cal._AUTOMATED_TARGET_STRATEGY_ID)
    # Also confirm the keyword-only default (no strategy_id passed at all) is unaffected.
    cal._require_collective_target_depths(profiles, split="calibration")


def test_verified_automated_circular_evidence_returns_the_resolved_target_strategy_id(tmp_path):
    """`_cmd_measure` threads this value straight into `_require_collective_target_depths`
    (never re-deriving it from a manifest independently), so the evidence verifier must expose
    exactly the strategy id it already checked manifest/acceptance/profile consistency for."""
    manifest = _minimal_unattended_manifest(
        target_strategy_id=cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID)
    session = _session_from_manifest(tmp_path, "photo1only-resolved", manifest)

    result = cal._verified_automated_circular_evidence([session])

    assert result["target_strategy_id"] == cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID


def test_verified_automated_circular_evidence_refuses_sessions_with_disagreeing_strategies(
        tmp_path):
    """One measure invocation is one campaign: if two otherwise-valid sessions were captured
    under two different strategies, there is no single answer to "the depths this evidence
    targets" -- refuse rather than silently picking one session's strategy over the other's."""
    alternating = _session_from_manifest(
        tmp_path, "alt",
        _minimal_unattended_manifest(target_strategy_id=cal._AUTOMATED_TARGET_STRATEGY_ID))
    photo_1_only = _session_from_manifest(
        tmp_path, "p1o",
        _minimal_unattended_manifest(target_strategy_id=cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID))

    with pytest.raises(cal._MeasureRefused, match="disagreeing automated target strategies"):
        cal._verified_automated_circular_evidence([alternating, photo_1_only])


def test_measure_depth_gate_uses_resolved_automated_strategy_not_hardcoded_pair(
        monkeypatch, tmp_path, capsys):
    """End-to-end-ish: a consistent `photo_1_only_v1` automated campaign whose every profile only
    ever touches item 1 must pass `_cmd_measure`'s collective-depth gate, using the strategy id
    `_verified_automated_circular_evidence` resolves rather than the hardcoded alternating pair --
    otherwise a legitimate photo_1_only_v1 campaign could never be measured once a split holds two
    or more profiles."""
    args, calibration, heldout = _successful_measure_seams(monkeypatch, tmp_path)
    for session in (calibration, heldout):
        session.profiles = [
            cal._ProfileData(p.ordinal, p.profile_id, p.card_frames,
                             [(1, pre, composer) for _item, pre, composer in p.composer_pairs],
                             p.profile_advance_clear, p.profile_advance_identity)
            for p in session.profiles
        ]
    monkeypatch.setattr(
        cal, "_verified_automated_circular_evidence",
        lambda _sessions: {"kind": "automated_circular_risk_accepted",
                           "not_supervised_operational_evidence": True,
                           "target_strategy_id": cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID})
    monkeypatch.setattr(cal, "_unattended_review_reference_reason", lambda *_a, **_kw: None)
    review_path = tmp_path / "unattended_review.json"
    review_path.write_text("{}")
    args.accept_automated_circular_evidence = True
    args.confirmation = cal._UNATTENDED_CONFIRMATION
    args.unattended_review = str(review_path)

    cal._cmd_measure(args)

    output = capsys.readouterr().out
    assert "REFUSED" not in output
    assert "targeting_calibration" in output


def _unattended_review_world(tmp_path, *, scopes, record_scopes=None):
    """Two capture sessions with DIFFERENT evidence scopes plus a self-hashed review artifact.

    Mixed-scope invocations are supported (a legacy closed-set session may be measured beside a
    target-scoped one), so each review record is bound to the scope of ITS OWN session unless
    `record_scopes` deliberately mis-binds them.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("apps: {}\n")
    config_digest = cal._sha256(config_path.read_bytes())
    sessions = []
    captures = []
    for number, scope in enumerate(scopes, 1):
        session = _session(tmp_path, f"session-{number}", "calibration", [f"p{number}"])
        session.manifest["frame_size_px"] = [1080, 2400]
        session.manifest["capture_evidence_scope"] = scope
        manifest_path = session.dir / "manifest.json"
        manifest_path.write_text(json.dumps(session.manifest, sort_keys=True))
        sessions.append(session)
        captures.append({
            "session": str(session.dir.resolve()),
            "human_ground_truth": False,
            "manifest_sha256": cal._sha256(manifest_path.read_bytes()),
            "device": session.manifest["device"],
            "frame_size_px": [1080, 2400],
            "config_sha256": config_digest,
            "capture_evidence_scope": scope,
        })
    if record_scopes is not None:
        for record, scope in zip(captures, record_scopes, strict=True):
            record["capture_evidence_scope"] = scope
    body = {
        "schema_version": cal._UNATTENDED_REVIEW_SCHEMA_VERSION,
        "kind": cal._UNATTENDED_REVIEW_KIND,
        "human_ground_truth": False,
        "not_independent_ground_truth": True,
        "reviewer": {"deterministic_second_process": True,
                     "imports_capture_or_vision_stack": False,
                     "implementation_sha256": "c" * 64},
        "config": {"sha256": config_digest},
        "captures": captures,
    }
    artifact = dict(body)
    artifact["evidence_sha256"] = cal._canonical_json_digest(body)
    review_path = tmp_path / "unattended_review.json"
    review_path.write_text(json.dumps(artifact))
    return str(review_path), sessions, str(config_path)


def test_unattended_review_binds_each_capture_scope_to_its_own_session(tmp_path):
    """Every other binding on this condition (manifest sha, device, frame size, config sha) is
    taken per record; the evidence scope was read off the leaked `for session in sessions` loop
    variable, so it was compared against whichever session happened to be LAST.  With a legacy
    closed-set session measured beside a target-scoped one that both rejects correct evidence and
    accepts a review artifact carrying the wrong scope for the non-last session."""
    scopes = ["closed_set_profile_v1", cal._TARGET_SCOPED_PREFIX_PROOF_ID]
    reference, sessions, config_path = _unattended_review_world(tmp_path, scopes=scopes)

    assert cal._unattended_review_reference_reason(
        reference, sessions, config_path=config_path) is None

    swapped, sessions, config_path = _unattended_review_world(
        tmp_path / "swapped", scopes=scopes, record_scopes=list(reversed(scopes)))
    reason = cal._unattended_review_reference_reason(
        swapped, sessions, config_path=config_path)
    assert reason is not None
    assert "capture binding differs" in reason


def test_operational_recorder_v2_requires_hashed_frames_and_exact_auto_focused_schema(monkeypatch,
                                                                                       tmp_path):
    device = _device()
    path = _write_operational_evidence(tmp_path / "operational", device=device)
    _wire_operational_replay(monkeypatch, path)

    assert cal._operational_evidence_reference_reason(
        str(path.parent), expected_device=device, identity_band=_IDENTITY_BAND) is None

    manifest = json.loads(path.read_text())
    manifest["frames"][1]["role"] = "composer_focused"
    body = dict(manifest)
    body.pop("evidence_sha256")
    manifest["evidence_sha256"] = cal._canonical_json_digest(body)
    path.write_text(json.dumps(manifest))
    reason = cal._operational_evidence_reference_reason(
        str(path), expected_device=device, identity_band=_IDENTITY_BAND)
    assert reason is not None
    assert "wrongly ordered" in reason


def test_operational_recorder_replays_hashed_frames_and_rejects_rehashed_analysis_tampering(
        monkeypatch, tmp_path):
    device = _device()
    path = _write_operational_evidence(tmp_path / "operational", device=device)
    _wire_operational_replay(monkeypatch, path)
    manifest = json.loads(path.read_text())
    manifest["analyses"]["new_sticky_identity"]["distance_from_item1_profile"] = 99.0
    body = dict(manifest)
    body.pop("evidence_sha256")
    manifest["evidence_sha256"] = cal._canonical_json_digest(body)
    path.write_text(json.dumps(manifest))

    reason = cal._operational_evidence_reference_reason(
        str(path), expected_device=device, identity_band=_IDENTITY_BAND)
    assert reason is not None
    assert "do not exactly match pure replay" in reason


def test_operational_recorder_v1_or_unhashed_assertion_is_never_measure_evidence(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    path = legacy / "manifest.json"
    path.write_text(json.dumps({"tool_version": "1", "completed": True}))

    reason = cal._operational_evidence_reference_reason(
        str(path), expected_device=_device(), identity_band=_IDENTITY_BAND)
    assert reason is not None
    assert "unsupported" in reason
    assert cal._operational_evidence_reference_reason(
        "operator-says-it-worked", expected_device=_device(), identity_band=_IDENTITY_BAND) is not None


def test_entry_anchor_operational_reference_must_name_this_session_ledger(tmp_path):
    session_dir = tmp_path / "capture"
    session_dir.mkdir()
    ledger = session_dir / cal._ENTRY_ANCHOR_LEDGER_FILE
    ledger.write_text("{}")
    session = SimpleNamespace(dir=session_dir)

    assert cal._entry_anchor_reference_reason(session, str(session_dir)) is None
    assert cal._entry_anchor_reference_reason(session, "operator-says-entry-is-good") is not None


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("device", "", "device serial"),
        ("calibrated_at", "", "calibration provenance"),
        ("identity_band", [0.1, 0.2, 0.3], "identity_band"),
        ("content_band", [0.1, float("nan")], "content_band"),
    ],
)
def test_v3_calibration_output_validation_rejects_incomplete_binding_fields(key, value, message):
    """The local emit guard must validate every persisted binding, not only numeric bounds."""
    block = {
        "schema_version": cal._CALIBRATION_SCHEMA_VERSION,
        "device": "PIXEL-TEST",
        "hinge_version_name": "10.1.0",
        "frame_size_px": [1080, 2400],
        "composer_layout_id": cal._COMPOSER_LAYOUT_ID,
        "item_selection_policy_id": cal.PHOTO_ONLY_POLICY_ID,
        "identity_match_max_dist": 1.0,
        "inline_item_max_dist": 5.0,
        "calibrated_at": "2026-08-26T20:17:22+00:00",
        "identity_band": list(_IDENTITY_BAND),
        "content_band": list(_CONTENT_BAND),
    }
    block[key] = value

    with pytest.raises(cal._MeasureRefused, match=message):
        cal._validate_v3_calibration_block(block)


def test_measure_success_prints_exact_calibration_mapping_and_full_ledger(monkeypatch, tmp_path, capsys):
    args, _calibration, heldout = _successful_measure_seams(monkeypatch, tmp_path)

    cal._cmd_measure(args)

    output = capsys.readouterr().out
    yaml_body = output.split("\napps:\n", 1)[1].split("\n==============================================================================", 1)[0]
    block = cal.yaml.safe_load("apps:\n" + yaml_body)["apps"]["hinge"]["targeting_calibration"]
    assert list(block) == ["schema_version", "device", "hinge_version_name", "frame_size_px",
                           "composer_layout_id", "item_selection_policy_id",
                           "identity_match_max_dist", "inline_item_max_dist",
                           "calibrated_at", "identity_band", "content_band"]
    assert block["schema_version"] == 3
    assert block["hinge_version_name"] == "1.0"
    assert block["frame_size_px"] == [1080, 2400]
    assert block["composer_layout_id"] == "hinge_inline_v1"
    assert block["item_selection_policy_id"] == "hinge_photos_only_v2"
    assert block["device"] == "PIXEL-TEST"
    assert re.search(r"20\d\d-\d\d-\d\dT", block["calibrated_at"])
    assert "placeholder" not in block["calibrated_at"].lower()

    ledger = json.loads((heldout.dir / "measurement_ledger.json").read_text())
    assert ledger["identity"]["frozen_bound"] == 1.0
    assert ledger["inline"]["frozen_bound"] == 5.0
    # What is applied to held-out and recorded is exactly what the owner is asked to paste.
    assert block["identity_match_max_dist"] == ledger["identity"]["frozen_bound"]
    assert block["inline_item_max_dist"] == ledger["inline"]["frozen_bound"]
    assert ledger["identity"]["heldout"]["ok"] is True
    assert ledger["inline"]["heldout"]["ok"] is True
    # The ledger retains both things the bounds accept and the alternatives they refuse.
    assert ledger["identity"]["heldout"]["same_profile_pairs"]
    assert ledger["identity"]["heldout"]["different_profile_pairs"]
    assert ledger["inline"]["heldout"]["intended"]
    assert ledger["inline"]["heldout"]["foreign"]


@pytest.mark.parametrize(
    ("samples", "ceiling"),
    [
        (_samples("p", [(0, cal._IDENTITY_FALSE_MATCH_DISTANCE),
                         (10, 11)]), cal._IDENTITY_FALSE_MATCH_DISTANCE),
    ],
)
def test_identity_bound_at_hard_ceiling_is_refused(monkeypatch, samples, ceiling):
    monkeypatch.setattr(cal, "fingerprint_distance", lambda a, b: abs(a[0] - b[0]))
    bound, report = cal._freeze_identity_bound(samples)
    assert bound is None
    assert report["max_same_profile_distance"] >= ceiling


def test_inline_bound_at_hard_ceiling_is_refused():
    records = [_record("own_intended", cal._INLINE_FALSE_MATCH_DISTANCE),
               _record("own_foreign_item", 30)]
    bound, report = cal._freeze_inline_bound(records)
    assert bound is None
    assert report["max_intended_distance"] >= cal._INLINE_FALSE_MATCH_DISTANCE


def test_nonseparable_calibration_refuses_without_yaml(monkeypatch, tmp_path, capsys):
    args, _calibration, _heldout = _successful_measure_seams(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cal, "_profile_identity_samples",
        lambda profiles, **_kw: (_samples("x", [(0, 4), (3, 7)])
                                 if profiles[0].profile_id == "a"
                                 else _samples("held", [(20, 20.2), (25, 25.2)])))
    monkeypatch.setattr(cal, "fingerprint_distance", lambda a, b: abs(a[0] - b[0]))

    with pytest.raises(SystemExit) as exc:
        cal._cmd_measure(args)
    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert "not separable" in captured.err
    assert "targeting_calibration:" not in captured.out


@pytest.mark.parametrize(
    ("held_identity", "held_inline", "signal"),
    [
        (_samples("held", [(0, 0.1), (0.5, 0.6)]), None, "different-profile"),
        (_samples("held", [(0, 5), (8, 9)]), None, "false refusal"),
        (None, [_record("own_intended", 1), _record("foreign_profile", 3)], "foreign-item"),
        (None, [_record("own_intended", 6), _record("foreign_profile", 8)], "false refusal"),
    ],
    ids=["foreign_profile_accept", "identity_false_refusal", "foreign_item_accept", "inline_false_refusal"],
)
def test_heldout_failure_refuses_and_never_prints_yaml(monkeypatch, tmp_path, capsys,
                                                        held_identity, held_inline, signal):
    args, _calibration, _heldout = _successful_measure_seams(
        monkeypatch, tmp_path, held_identity=held_identity, held_inline_records=held_inline)

    with pytest.raises(SystemExit) as exc:
        cal._cmd_measure(args)
    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert signal in captured.err.lower()
    assert "targeting_calibration:" not in captured.out


@pytest.mark.parametrize("splits", [(["calibration"],), (["heldout"],)])
def test_missing_calibration_or_heldout_split_is_refused(monkeypatch, tmp_path, capsys, splits):
    split = splits[0][0]
    session = _session(tmp_path, "only", split, ["a", "b"])
    monkeypatch.setattr(cal.cfg_mod, "load", lambda _path: _Cfg())
    monkeypatch.setattr(cal, "HingeDriver", _InertDriver)
    monkeypatch.setattr(cal, "_load_session", lambda _raw: session)

    with pytest.raises(SystemExit) as exc:
        cal._cmd_measure(argparse.Namespace(config="x", sessions=["only"]))
    assert exc.value.code != 0
    assert "each split" in capsys.readouterr().err.lower()


def test_profile_id_shared_between_splits_is_refused(monkeypatch, tmp_path, capsys):
    calibration = _session(tmp_path, "cal", "calibration", ["same"])
    heldout = _session(tmp_path, "held", "heldout", ["same"])
    monkeypatch.setattr(cal.cfg_mod, "load", lambda _path: _Cfg())
    monkeypatch.setattr(cal, "HingeDriver", _InertDriver)
    monkeypatch.setattr(cal, "_load_session", lambda raw: {"cal": calibration, "held": heldout}[str(raw)])

    with pytest.raises(SystemExit):
        cal._cmd_measure(argparse.Namespace(config="x", sessions=["cal", "held"]))
    assert "both the calibration and held-out" in capsys.readouterr().err.lower()


def test_session_band_mismatch_is_refused_before_any_measurement(monkeypatch, tmp_path, capsys):
    bad_band = (0.10, 0.05, 0.80, 0.094)
    session = _session(tmp_path, "cal", "calibration", ["a"], identity_band=bad_band)
    monkeypatch.setattr(cal.cfg_mod, "load", lambda _path: _Cfg())
    monkeypatch.setattr(cal, "HingeDriver", _InertDriver)
    monkeypatch.setattr(cal, "_load_session", lambda _raw: session)

    with pytest.raises(SystemExit):
        cal._cmd_measure(argparse.Namespace(config="x", sessions=["cal"]))
    assert "does not exactly match" in capsys.readouterr().err.lower()


def test_sessions_with_different_effective_bands_are_refused(monkeypatch, tmp_path, capsys):
    calibration = _session(tmp_path, "cal", "calibration", ["a"], identity_band=_IDENTITY_BAND)
    heldout = _session(tmp_path, "held", "heldout", ["b"], content_band=(0.12, 0.875))
    monkeypatch.setattr(cal.cfg_mod, "load", lambda _path: _Cfg())
    monkeypatch.setattr(cal, "HingeDriver", _InertDriver)
    monkeypatch.setattr(cal, "_load_session", lambda raw: {"cal": calibration, "held": heldout}[str(raw)])

    with pytest.raises(SystemExit):
        cal._cmd_measure(argparse.Namespace(config="x", sessions=["cal", "held"]))
    assert "does not exactly match" in capsys.readouterr().err.lower()


def test_load_session_refuses_changed_frame_bytes(tmp_path):
    data = b"original synthetic frame"
    frame = tmp_path / "00001.png"
    frame.write_bytes(b"changed synthetic frame")
    manifest = {
        "tool_version": cal._TOOL_VERSION, "interrupted": False, "split": "calibration",
        "identity_band": list(_IDENTITY_BAND), "content_band": list(_CONTENT_BAND),
        "device": _device(), "requested_profiles": 1,
        "calibration_schema_version": 3, "composer_layout_id": "hinge_inline_v1",
        "item_selection_policy_id": "hinge_photos_only_v2",
        "frame_size_px": [1080, 2400],
        "profiles": [{"ordinal": 1, "profile_id": "p", "card_scroll_frames": 1,
                      "composer_items": [1], "profile_advance_cleared_composer": True,
                      "profile_advance_identity_mismatched": True}],
        "operational_checks": _checks(), "frame_count": 5,
        "frames": [
            {"file": frame.name, "sha256": hashlib.sha256(data).hexdigest(),
             "profile_ordinal": 1, "profile_id": "p", "role": "card_scroll",
             "item_number": None},
            {"file": "00002.png", "sha256": hashlib.sha256(b"pre").hexdigest(),
             "profile_ordinal": 1, "profile_id": "p", "role": "target_pre", "item_number": 1},
            {"file": "00003.png", "sha256": hashlib.sha256(b"composer").hexdigest(),
             "profile_ordinal": 1, "profile_id": "p", "role": "composer_open", "item_number": 1},
            {"file": "00004.png", "sha256": hashlib.sha256(b"advance").hexdigest(),
             "profile_ordinal": 1, "profile_id": "p", "role": "profile_advance_clear", "item_number": None},
            {"file": "00005.png", "sha256": hashlib.sha256(b"advance identity").hexdigest(),
             "profile_ordinal": 1, "profile_id": "p", "role": "profile_advance_identity", "item_number": None},
        ],
    }
    (tmp_path / "00002.png").write_bytes(b"pre")
    (tmp_path / "00003.png").write_bytes(b"composer")
    (tmp_path / "00004.png").write_bytes(b"advance")
    (tmp_path / "00005.png").write_bytes(b"advance identity")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        cal._load_session(tmp_path)


def test_identity_fingerprints_always_use_production_grid(monkeypatch):
    seen = []
    known = SimpleNamespace(known=True, fingerprint=(1,), reason="ok")
    monkeypatch.setattr(cal, "capture_profile_identity",
                        lambda _frames, **kw: seen.append(kw["grid"]) or known)
    monkeypatch.setattr(cal, "band_fingerprint",
                        lambda _frame, **kw: seen.append(kw["grid"]) or (2,))
    profile = cal._ProfileData(1, "p", [b"card"], [(1, b"pre", b"composer")], b"advance",
                               b"advance-identity")

    cal._profile_identity_samples([profile], identity_band=_IDENTITY_BAND)
    assert seen == [cal._IDENTITY_GRID, cal._IDENTITY_GRID]


def test_measure_source_policy_has_no_device_io_raw_injection_or_fixed_sleep():
    source = Path(cal.__file__).read_text()
    tree = ast.parse(source)
    measure_source = inspect.getsource(cal._cmd_measure)
    assert ".open_session(" not in measure_source
    assert ".screencap(" not in measure_source
    assert ".adb." not in measure_source

    raw_input_calls = []
    bare_sleeps = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "shell":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if re.search(r"\binput\s+(?:tap|swipe)\b", arg.value):
                        raw_input_calls.append(arg.value)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "sleep":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, (int, float)):
                bare_sleeps.append(node.args[0].value)
    assert not raw_input_calls
    assert not bare_sleeps
    # The command writes only its private evidence ledger, never a config mapping.
    assert measure_source.count("atomic_write_private_text(") == 1
    assert ".write_text(" not in measure_source


def _write_complete_manifest(directory, *, interrupted=False, requested_profiles=1,
                             profiles=None, checks=None):
    """Small on-disk completed session accepted by every schema check before a mutation."""
    profiles = profiles or [{"ordinal": 1, "profile_id": "p", "card_scroll_frames": 1,
                             "composer_items": [1], "profile_advance_cleared_composer": True,
                             "profile_advance_identity_mismatched": True}]
    checks = _checks(directory) if checks is None else checks
    card, pre, composer, advance, advance_identity = (
        b"synthetic card", b"synthetic pre", b"synthetic composer", b"synthetic advance",
        b"synthetic advance identity")
    (directory / "00001.png").write_bytes(card)
    (directory / "00002.png").write_bytes(pre)
    (directory / "00003.png").write_bytes(composer)
    (directory / "00004.png").write_bytes(advance)
    (directory / "00005.png").write_bytes(advance_identity)
    manifest = {
        "tool_version": cal._TOOL_VERSION, "interrupted": interrupted,
        "calibration_schema_version": 3, "composer_layout_id": "hinge_inline_v1",
        "item_selection_policy_id": "hinge_photos_only_v2",
        "frame_size_px": [1080, 2400],
        "split": "calibration", "identity_band": list(_IDENTITY_BAND),
        "content_band": list(_CONTENT_BAND), "device": _device(),
        "requested_profiles": requested_profiles, "profiles": profiles,
        "operational_checks": checks, "frame_count": 5,
        "frames": [
            {"file": "00001.png", "sha256": hashlib.sha256(card).hexdigest(),
             "profile_ordinal": 1, "profile_id": "p", "role": "card_scroll",
             "item_number": None},
            {"file": "00002.png", "sha256": hashlib.sha256(pre).hexdigest(),
             "profile_ordinal": 1, "profile_id": "p", "role": "target_pre", "item_number": 1},
            {"file": "00003.png", "sha256": hashlib.sha256(composer).hexdigest(),
             "profile_ordinal": 1, "profile_id": "p", "role": "composer_open", "item_number": 1},
            {"file": "00004.png", "sha256": hashlib.sha256(advance).hexdigest(),
             "profile_ordinal": 1, "profile_id": "p", "role": "profile_advance_clear", "item_number": None},
            {"file": "00005.png", "sha256": hashlib.sha256(advance_identity).hexdigest(),
             "profile_ordinal": 1, "profile_id": "p", "role": "profile_advance_identity", "item_number": None},
        ],
    }
    # The completed v3 session is also bound to the automatic, offline-only entry-anchor
    # replay.  Keep this synthetic fixture structurally identical to a capture: the ledger
    # names the exact card byte sequence, while the manifest authenticates the ledger bytes.
    ledger = {
        "schema_version": cal._ENTRY_ANCHOR_LEDGER_SCHEMA_VERSION,
        "kind": "hinge_entry_anchor_offline_replay",
        "tool_version": cal._TOOL_VERSION,
        "created_utc": "2026-08-13T12:00:00+00:00",
        "offline_only": True,
        "phone_input_issued": False,
        "runtime_targeting_calibration_emitted": False,
        "profiles": [{
            "ordinal": 1,
            "profile_id_sha256": hashlib.sha256(b"p").hexdigest(),
            "source_card_frames": [{
                "file": "00001.png", "sha256": hashlib.sha256(card).hexdigest()}],
            "confirmed_top": True,
            "index_complete": True,
            "index_reached_end": True,
            "index_heart_translation": [1],
            "photo_only_translation": [1],
            "offline_replay_identity_ceiling": 1.0,
            "offline_replay_identity_entry_distance": 0.0,
            "cancellation_before_capture": True,
            "cancellation_replay_capture_calls": 0,
            "cancellation_replay_scroll_up_calls": 0,
            "targets": [{
                "photo_model_item": 1,
                "heart_ordinal": 1,
                "index_model_item": 1,
                "entry_anchor_delta_px": 0,
                "entry_offset_px": 0,
                "landing_frame_sha256": hashlib.sha256(card).hexdigest(),
                "landing_frame_index": 0,
                "landing_page_offset_px": 0,
                "hearts_counted": 1,
                "crosscheck_max_disagreement_px": 0,
                "replay_scroll_steps": 0,
                "replay_shift_deltas_px": [],
            }],
        }],
    }
    ledger_path = directory / cal._ENTRY_ANCHOR_LEDGER_FILE
    ledger_path.write_text(json.dumps(ledger))
    manifest["entry_anchor_ledger"] = {
        "file": ledger_path.name,
        "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize(
    ("interrupted", "requested_profiles", "profiles", "message"),
    [
        (True, 1, None, "interrupted, partial"),
        (False, 2, [{"ordinal": 1, "profile_id": "p", "composer_items": [1],
                     "profile_advance_cleared_composer": True,
                     "profile_advance_identity_mismatched": True}], "partial session"),
    ],
    ids=["interrupted", "requested_but_incomplete"],
)
def test_load_session_refuses_interrupted_or_partial_capture(tmp_path, interrupted,
                                                             requested_profiles, profiles, message):
    _write_complete_manifest(tmp_path, interrupted=interrupted, requested_profiles=requested_profiles,
                             profiles=profiles)
    with pytest.raises(RuntimeError, match=message):
        cal._load_session(tmp_path)


def test_load_session_refuses_duplicate_profile_id_within_one_session(tmp_path):
    _write_complete_manifest(
        tmp_path, requested_profiles=2,
        profiles=[{"ordinal": 1, "profile_id": "same", "card_scroll_frames": 1,
                   "composer_items": [1], "profile_advance_cleared_composer": True,
                   "profile_advance_identity_mismatched": True},
                  {"ordinal": 2, "profile_id": "same", "card_scroll_frames": 1,
                   "composer_items": [1], "profile_advance_cleared_composer": True,
                   "profile_advance_identity_mismatched": True}],
    )
    with pytest.raises(RuntimeError, match="duplicated"):
        cal._load_session(tmp_path)


def test_load_session_accepts_one_exact_completed_attempt(tmp_path):
    _write_complete_manifest(tmp_path)
    session = cal._load_session(tmp_path)
    assert session.split == "calibration"
    assert [(p.profile_id, len(p.card_frames), len(p.composer_pairs))
            for p in session.profiles] == [("p", 1, 1)]


def test_load_session_refuses_entry_anchor_ledger_for_different_card_bytes(tmp_path):
    manifest = _write_complete_manifest(tmp_path)
    ledger_path = tmp_path / cal._ENTRY_ANCHOR_LEDGER_FILE
    ledger = json.loads(ledger_path.read_text())
    ledger["profiles"][0]["source_card_frames"][0]["sha256"] = "0" * 64
    ledger_path.write_text(json.dumps(ledger))
    manifest["entry_anchor_ledger"]["sha256"] = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="does not bind the exact manifest"):
        cal._load_session(tmp_path)


def test_entry_anchor_replay_cancels_before_it_captures_or_scrolls(monkeypatch):
    card = b"synthetic card"
    index = SimpleNamespace(
        usable=True, complete=True, reached_end=True, translation=(1,),
        identity=SimpleNamespace(known=True, fingerprint=(1,)),
    )
    payload = SimpleNamespace(
        usable=True, translation=(1,), items=[SimpleNamespace(number=1, heart_ordinal=1)],
    )
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_args, **_kw: SimpleNamespace(confirmed=True, reason="top"))
    monkeypatch.setattr(cal, "build_item_index", lambda *_args, **_kw: index)
    monkeypatch.setattr(cal, "build_item_payload", lambda *_args, **_kw: payload)
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_args, **_kw: (1,))
    monkeypatch.setattr(cal, "fingerprint_distance", lambda *_args, **_kw: 0.0)

    def fake_navigate(replay, _index, _model_item, *, should_stop=None, **_kw):
        if should_stop is not None:
            assert should_stop() is True
            raise cal.ActionCancelled("operator cancelled")
        return SimpleNamespace(
            anchor=SimpleNamespace(delta_px=0), entry_offset=0, frame=card, frame_index=0,
            page_offset=0, hearts_counted=1, agreement_px=0, steps=(), shifts=(),
        )

    monkeypatch.setattr(cal, "navigate_to_item", fake_navigate)
    report = cal._entry_anchor_profile_report(
        ordinal=1, profile_id="private profile", card_frames=[card],
        frame_records=[{"file": "00001.png", "sha256": hashlib.sha256(card).hexdigest()}],
        identity_band=_IDENTITY_BAND, content_band=_CONTENT_BAND, like_template=object(),
        like_threshold=0.75)

    assert report["confirmed_top"] is True
    assert report["index_complete"] is True
    assert report["photo_only_translation"] == [1]
    assert report["cancellation_before_capture"] is True
    assert report["cancellation_replay_capture_calls"] == 0
    assert report["cancellation_replay_scroll_up_calls"] == 0
    assert report["targets"][0]["landing_frame_sha256"] == hashlib.sha256(card).hexdigest()


def test_verify_entry_anchor_refuses_unmanifested_pngs_without_constructing_driver(
        monkeypatch, tmp_path, capsys):
    root = tmp_path / "ops" / "calibration" / "targeting_unmanifested"
    root.mkdir(parents=True)
    (root / "00001.png").write_bytes(b"private but unclassified")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cal, "HingeDriver",
                        lambda _cfg: pytest.fail("unmanifested input must fail before driver construction"))

    with pytest.raises(SystemExit):
        cal._cmd_verify_entry_anchor(
            argparse.Namespace(session="ops/calibration/targeting_unmanifested", config="x"))
    assert "refusing to reconstruct" in capsys.readouterr().err


def test_verify_entry_anchor_source_never_opens_a_device_session():
    source = inspect.getsource(cal._cmd_verify_entry_anchor)
    assert ".open_session(" not in source
    assert ".adb." not in source


def test_load_session_requires_terminal_next_profile_identity_evidence(tmp_path):
    manifest = _write_complete_manifest(tmp_path)
    manifest["profiles"][0].pop("profile_advance_identity_mismatched")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="identity-mismatch evidence"):
        cal._load_session(tmp_path)


def test_load_session_refuses_reordered_manifest_frames(tmp_path):
    manifest = _write_complete_manifest(tmp_path)
    manifest["frames"].reverse()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="original contiguous capture sequence"):
        cal._load_session(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest["frames"].__setitem__(1, {
            **manifest["frames"][1], "role": "sheet"}), "unknown frame role"),
        (lambda manifest: manifest["frames"].__setitem__(2, {
            **manifest["frames"][2], "role": "target_pre"}), "persistent-composer sequence"),
        (lambda manifest: manifest["frames"].__setitem__(3, {
            **manifest["frames"][3], "role": "composer_open", "item_number": 1}),
         "persistent-composer sequence"),
        (lambda manifest: manifest["frames"].__setitem__(4, {
            **manifest["frames"][4], "role": "composer_open", "item_number": 1}),
         "persistent-composer sequence"),
    ],
    ids=["legacy_sheet_role", "missing_composer_open", "missing_advance_clear",
         "missing_advance_identity"],
)
def test_load_session_refuses_legacy_or_incomplete_persistent_composer_roles(tmp_path, mutate, message):
    manifest = _write_complete_manifest(tmp_path)
    mutate(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match=message):
        cal._load_session(tmp_path)


def test_duplicate_profile_id_within_same_split_is_refused(monkeypatch, tmp_path, capsys):
    calibration = _session(tmp_path, "cal", "calibration", ["same", "same"])
    heldout = _session(tmp_path, "held", "heldout", ["other"])
    monkeypatch.setattr(cal.cfg_mod, "load", lambda _path: _Cfg())
    monkeypatch.setattr(cal, "HingeDriver", _InertDriver)
    monkeypatch.setattr(cal, "_load_session",
                        lambda raw: {"cal": calibration, "held": heldout}[str(raw)])

    with pytest.raises(SystemExit):
        cal._cmd_measure(argparse.Namespace(config="x", sessions=["cal", "held"]))
    assert "duplicated within the calibration split" in capsys.readouterr().err.lower()


def test_measure_refuses_without_all_operational_check_evidence(monkeypatch, tmp_path, capsys):
    args, calibration, heldout = _successful_measure_seams(monkeypatch, tmp_path)
    calibration.manifest["operational_checks"] = _checks(confirmed=False)
    heldout.manifest["operational_checks"] = _checks(confirmed=False)

    with pytest.raises(SystemExit):
        cal._cmd_measure(args)
    assert "operational checks" in capsys.readouterr().err.lower()


def test_measure_refuses_probable_cross_split_profile_reuse_under_new_label(
        monkeypatch, tmp_path, capsys):
    args, _calibration, _heldout = _successful_measure_seams(monkeypatch, tmp_path)
    calibration = _samples("cal-label", [(0, 0.2), (2, 2.2)])
    heldout = _samples("different-label", [(0, 0.2), (8, 8.2)])
    monkeypatch.setattr(
        cal, "_profile_identity_samples",
        lambda profiles, **_kw: calibration if profiles[0].profile_id == "a" else heldout)
    with pytest.raises(SystemExit):
        cal._cmd_measure(args)
    assert "same real profile may have been reused" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("bad_distance", [None, float("nan"), float("inf"), -1],
                         ids=["none", "nan", "infinity", "negative"])
def test_invalid_identity_distance_refuses_instead_of_erasing_guard(monkeypatch, bad_distance):
    samples = _samples("p", [(0, 0), (2, 2)])
    monkeypatch.setattr(cal, "fingerprint_distance", lambda _a, _b: bad_distance)
    with pytest.raises(cal._MeasureRefused, match="invalid distance"):
        cal._freeze_identity_bound(samples)


@pytest.mark.parametrize("bad_distance", [None, float("nan")], ids=["none", "nan"])
def test_unavailable_or_invalid_inline_distance_refuses(monkeypatch, bad_distance):
    payload = SimpleNamespace(items=[SimpleNamespace(number=1)])
    profile = cal._ProfilePayload("p", payload, [(1, b"pre", b"composer")])
    comparison = SimpleNamespace(number=1, distance=bad_distance, reason="synthetic unavailable")
    monkeypatch.setattr(cal, "locate_inline_composer", lambda *_args, **_kw:
                        SimpleNamespace(layout_id=cal._COMPOSER_LAYOUT_ID))
    monkeypatch.setattr(cal, "verify_sheet_item",
                        lambda *_args, **_kw: SimpleNamespace(comparisons=[comparison]))
    with pytest.raises(cal._MeasureRefused):
        cal._inline_distance_records([profile], confirm_template=object())


def test_measure_refuses_a_truncated_item_index(monkeypatch):
    profile = cal._ProfileData(1, "p", [b"card"], [(1, b"pre", b"composer")], b"advance",
                               b"advance-identity")
    index = SimpleNamespace(usable=True, complete=False, at_scroll_top=True,
                            reached_end=False, partial=(), failures=(), selectable=(1,))
    monkeypatch.setattr(cal, "build_item_index", lambda *_args, **_kw: index)
    with pytest.raises(cal._MeasureRefused, match="complete profile"):
        cal._build_profile_payloads(
            [profile], content_band=_CONTENT_BAND, identity_band=_IDENTITY_BAND,
            like_template=object(), like_threshold=0.75)


def test_identity_foreign_distance_equal_to_bound_is_an_invalid_accept(monkeypatch):
    monkeypatch.setattr(cal, "fingerprint_distance", lambda a, b: abs(a[0] - b[0]))
    samples = _samples("p", [(0, 0), (1, 1)])
    ok, _message, detail = cal._apply_identity_bound(samples, 1.0)
    assert ok is False
    assert detail["foreign_accepts"]


def test_safe_yaml_round_trip_preserves_serial_with_yaml_metacharacters(monkeypatch, tmp_path, capsys):
    serial = "Pixel: # burner [A]"
    args, _calibration, _heldout = _successful_measure_seams(monkeypatch, tmp_path, serial=serial)
    cal._cmd_measure(args)
    yaml_body = capsys.readouterr().out.split("\napps:\n", 1)[1].split("\n==============================================================================", 1)[0]
    block = cal.yaml.safe_load("apps:\n" + yaml_body)["apps"]["hinge"]["targeting_calibration"]
    assert block["device"] == serial


def test_capture_output_outside_private_calibration_root_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cal._capture_out_dir(str(tmp_path))
    assert exc.value.code != 0
    assert "must stay under" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_capture_output_directory_is_owner_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    out = cal._capture_out_dir("ops/calibration/private-session")

    assert stat.S_IMODE(out.stat().st_mode) == 0o700


def test_retry_discards_staged_frames_before_committing_the_completed_attempt(monkeypatch, tmp_path):
    class _Adb:
        def __init__(self):
            self.frames = iter([
                b"bad top", b"good top", b"pre", b"composer", b"advanced",
                b"advanced identity",
            ])

        def screencap(self):
            return next(self.frames)

    class _Driver:
        identity_band = _IDENTITY_BAND
        content_band = _CONTENT_BAND
        dwell_s = 0.1
        adb = _Adb()

        def _template(self, _name):
            return object()

    inputs = iter(["first", "1", "", "", "second", "1", "", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda frame, **_kw: SimpleNamespace(
                            confirmed=frame in {b"good top", b"advanced"},
                            refuted=frame == b"advanced identity", reason="not top"))
    monkeypatch.setattr(cal, "capture_profile_identity",
                        lambda *_args, **_kw: SimpleNamespace(known=True, frame_index=0,
                                                               agreeing_frames=1, fingerprint=(0,), reason="ok"))
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_args, **_kw: (10,))
    monkeypatch.setattr(cal, "fingerprint_distance", lambda *_args, **_kw: 10.0)
    monkeypatch.setattr(cal, "build_item_index",
                        lambda *_args, **_kw: SimpleNamespace(usable=True, complete=True,
                                                               selectable=[1], failures=[]))
    monkeypatch.setattr(cal, "build_item_payload",
                        lambda *_args, **_kw: SimpleNamespace(
                            usable=True, items=[SimpleNamespace(number=1)], failures=[]))

    monkeypatch.setattr(cal, "locate_inline_composer",
                        lambda frame, *_args, **_kw: (
                            (_ for _ in ()).throw(cal.ComposerDetectionError("not open"))
                            if frame == b"advanced" else SimpleNamespace(layout_id=cal._COMPOSER_LAYOUT_ID)))

    frames_meta = []
    meta, counter = cal._capture_one_profile(
        _Driver(), tmp_path, ordinal=1, frame_counter=0, frames_meta=frames_meta,
        used_profile_ids=set())
    assert meta["profile_id"] == "second"
    assert counter == 5
    assert [rec["profile_id"] for rec in frames_meta] == ["second"] * 5
    assert [rec["role"] for rec in frames_meta] == [
        "card_scroll", "target_pre", "composer_open", "profile_advance_clear",
        "profile_advance_identity"]
    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "00001.png", "00002.png", "00003.png", "00004.png", "00005.png"]


def test_capture_moves_persistent_composer_between_items_without_a_dismissal(monkeypatch, tmp_path):
    class _Adb:
        def __init__(self):
            self.frames = iter([b"top", b"pre-1", b"composer-1", b"pre-2", b"composer-2",
                                b"advanced", b"advanced identity"])

        def screencap(self):
            return next(self.frames)

    class _Driver:
        identity_band = _IDENTITY_BAND
        content_band = _CONTENT_BAND
        dwell_s = 0.1
        adb = _Adb()

        def _template(self, _name):
            return object()

    prompts = []
    inputs = iter(["p", "1,2", "", "", "", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or next(inputs))
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda frame, **_kw: SimpleNamespace(
                            confirmed=frame != b"advanced identity",
                            refuted=frame == b"advanced identity", reason="top"))
    monkeypatch.setattr(cal, "capture_profile_identity",
                        lambda *_args, **_kw: SimpleNamespace(known=True, frame_index=0,
                                                               agreeing_frames=1, fingerprint=(0,), reason="ok"))
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_args, **_kw: (10,))
    monkeypatch.setattr(cal, "fingerprint_distance", lambda *_args, **_kw: 10.0)
    monkeypatch.setattr(cal, "build_item_index",
                        lambda *_args, **_kw: SimpleNamespace(usable=True, complete=True,
                                                               selectable=[1, 2], failures=[]))
    monkeypatch.setattr(cal, "build_item_payload",
                        lambda *_args, **_kw: SimpleNamespace(
                            usable=True,
                            items=[SimpleNamespace(number=1), SimpleNamespace(number=2)],
                            failures=[]))
    monkeypatch.setattr(cal, "locate_inline_composer",
                        lambda frame, *_args, **_kw: (
                            (_ for _ in ()).throw(cal.ComposerDetectionError("gone"))
                            if frame == b"advanced" else SimpleNamespace(layout_id=cal._COMPOSER_LAYOUT_ID)))

    frames_meta = []
    driver = _Driver()
    meta, _counter = cal._capture_one_profile(
        driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=frames_meta,
        used_profile_ids=set())
    assert meta["composer_items"] == [1, 2]
    assert [rec["role"] for rec in frames_meta] == [
        "card_scroll", "target_pre", "composer_open", "target_pre", "composer_open",
        "profile_advance_clear", "profile_advance_identity"]
    assert not any("dismiss/cancel" in prompt.lower() for prompt in prompts)


def test_capture_card_scan_passes_only_the_prior_exact_index_prefix(monkeypatch, tmp_path):
    class _Adb:
        def __init__(self):
            self.frames = iter([b"top", b"scroll", b"pre", b"composer", b"advanced",
                                b"advanced identity"])

        def screencap(self):
            return next(self.frames)

    class _Driver:
        identity_band = _IDENTITY_BAND
        content_band = _CONTENT_BAND
        dwell_s = 0.0
        adb = _Adb()

        def _template(self, _name):
            return object()

        def _scroll_down_one(self, frac, x_frac):
            self.scroll_args = (frac, x_frac)

    inputs = iter(["p", "1", "", "", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    monkeypatch.setattr(cal, "time", SimpleNamespace(sleep=lambda *_args: None))
    monkeypatch.setattr(cal, "human_delay", lambda *_args: 0.0)
    monkeypatch.setattr(cal, "_settled_read_scroll_frame",
                        lambda driver: driver.adb.screencap())
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda frame, **_kw: SimpleNamespace(
                            confirmed=frame in {b"top", b"advanced"},
                            refuted=frame in {b"scroll", b"advanced identity"}, reason="ok"))
    identity = SimpleNamespace(known=True, frame_index=1, agreeing_frames=2, fingerprint=(0,), reason="ok")
    monkeypatch.setattr(cal, "capture_profile_identity", lambda *_args, **_kw: identity)
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_args, **_kw: (10,))
    monkeypatch.setattr(cal, "fingerprint_distance", lambda *_args, **_kw: 10.0)
    indexes = []
    prefixes = []

    def build(frames, **kwargs):
        prefixes.append(kwargs["_prefix_index"])
        indexes.append(SimpleNamespace(usable=True, complete=len(frames) == 2,
                                       selectable=[1], failures=[]))
        return indexes[-1]

    monkeypatch.setattr(cal, "build_item_index", build)
    monkeypatch.setattr(cal, "build_item_payload",
                        lambda *_args, **_kw: SimpleNamespace(
                            usable=True, items=[SimpleNamespace(number=1)], failures=[]))
    planned = SimpleNamespace(frac=0.1, x_frac=0.5, spacing_px=780)
    plan_calls = []
    monkeypatch.setattr(
        cal, "_plan_card_scroll",
        lambda frame, **kw: (plan_calls.append((frame, kw)) or (planned, 780)))
    monkeypatch.setattr(cal, "locate_inline_composer",
                        lambda frame, *_args, **_kw: (
                            (_ for _ in ()).throw(cal.ComposerDetectionError("gone"))
                            if frame == b"advanced" else SimpleNamespace(layout_id=cal._COMPOSER_LAYOUT_ID)))

    frames_meta = []
    driver = _Driver()
    meta, _counter = cal._capture_one_profile(
        driver, tmp_path, ordinal=1, frame_counter=0, frames_meta=frames_meta,
        used_profile_ids=set())

    assert meta["card_scroll_frames"] == 2
    assert len(indexes) == 2
    # The second scan is allowed to reuse only the exact prior result; the indexer itself
    # hash-validates that its source frames remain the current list's prefix.
    assert prefixes == [None, indexes[0]]
    # Calibration capture spends the alias-safe plan, including its lane, rather than sampling a
    # convenient fixed fraction.  The replay is then checking the same safety envelope that
    # produced the recorded frames.
    assert len(plan_calls) == 1
    frame, kwargs = plan_calls[0]
    assert frame == b"top"
    assert kwargs["content_band"] == _CONTENT_BAND
    assert kwargs["like_threshold"] == cal.hinge_mod._LIKE_MATCH_THRESHOLD
    assert kwargs["profile_min_spacing_px"] is None
    assert driver.scroll_args == (planned.frac, planned.x_frac)


def test_capture_refuses_before_an_unplanned_card_scroll(monkeypatch, tmp_path):
    """A frame that cannot produce an alias-safe plan is a retry, never a probe gesture.

    This is the capture-side counterpart to `item_nav`'s post-gesture overshoot refusal.  It
    prevents creating another complete-looking card sequence which an exact offline replay must
    later reject because its recorded hop was larger than the local spacing allowed.
    """
    class _Adb:
        def screencap(self):
            return b"top"

    class _Driver:
        identity_band = _IDENTITY_BAND
        content_band = _CONTENT_BAND
        dwell_s = 0.0
        adb = _Adb()

        def _template(self, _name):
            return object()

        def _scroll_down_one(self, *_args):
            pytest.fail("unsafe card frame must not produce a scroll gesture")

    inputs = iter(["p", "1", "", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_args, **_kw: SimpleNamespace(confirmed=True, reason="top"))
    monkeypatch.setattr(cal, "capture_profile_identity",
                        lambda *_args, **_kw: SimpleNamespace(known=True, reason="ok"))
    monkeypatch.setattr(cal, "build_item_index",
                        lambda *_args, **_kw: SimpleNamespace(usable=False, complete=False,
                                                               failures=["incomplete"]))
    monkeypatch.setattr(cal, "_plan_card_scroll",
                        lambda *_args, **_kw: (_ for _ in ()).throw(
                            cal.ScrollStepError("local spacing permits no safe step")))

    with pytest.raises(cal._CaptureAbort, match="declined to retry"):
        cal._capture_one_profile(
            _Driver(), tmp_path, ordinal=1, frame_counter=0, frames_meta=[],
            used_profile_ids=set())


def test_terminal_advance_evidence_refuses_an_unconfirmed_top(monkeypatch):
    profile = cal._ProfileData(1, "p", [b"card"], [(1, b"pre", b"composer")], b"advance",
                               b"advance-identity")
    monkeypatch.setattr(cal, "confirm_scroll_top",
                        lambda *_args, **_kw: SimpleNamespace(confirmed=False, reason="not top"))
    with pytest.raises(cal._MeasureRefused, match="not a confirmed profile top"):
        cal._validate_profile_advance_clears(
            [profile], identity_band=_IDENTITY_BAND, confirm_template=object())


def test_terminal_advance_evidence_refuses_same_profile_identity(monkeypatch):
    profile = cal._ProfileData(1, "p", [b"card"], [(1, b"pre", b"composer")], b"advance",
                               b"advance-identity")
    monkeypatch.setattr(
        cal, "confirm_scroll_top",
        lambda frame, **_kw: SimpleNamespace(
            confirmed=frame == b"advance", refuted=frame == b"advance-identity", reason="ok"))
    monkeypatch.setattr(
        cal, "locate_inline_composer",
        lambda *_args, **_kw: (_ for _ in ()).throw(cal.ComposerDetectionError("gone")))
    monkeypatch.setattr(
        cal, "capture_profile_identity",
        lambda *_args, **_kw: SimpleNamespace(known=True, fingerprint=(1,), reason="ok"))
    monkeypatch.setattr(cal, "band_fingerprint", lambda *_args, **_kw: (1,))
    monkeypatch.setattr(cal, "fingerprint_distance",
                        lambda *_args, **_kw: cal._IDENTITY_FALSE_MATCH_DISTANCE)

    with pytest.raises(cal._MeasureRefused, match="same-profile scroll"):
        cal._validate_profile_advance_clears(
            [profile], identity_band=_IDENTITY_BAND, confirm_template=object())


def test_capture_validates_the_config_so_the_licence_its_proof_consults_is_installed(
        tmp_path, monkeypatch, capsys):
    """FOUND LIVE 2026-08-22, attempt 3 of the first hybrid campaign after the licence shipped.

    `config.validate()` is the ONLY thing that installs the process-local still-photo licence,
    and `_verified_still_photo_proof`'s ladder consults it via
    `hinge_targeting_unavailable_reason()`.  The capture command only ever *loaded* the config,
    so the tool read, numbered, and navigated a real profile perfectly and then refused every
    heart at the policy rung -- a full profile's gestures spent to discover the process forgot
    to license itself.  Capture must validate before it touches the device.
    """
    import operation_love.config as cfg_mod_real

    calls = []
    monkeypatch.setattr(cal.cfg_mod, "validate",
                        lambda cfg: calls.append("validate"))
    # Refuse at the driver so the test never proceeds past config handling: validate must
    # already have happened by then.
    monkeypatch.setattr(cal, "HingeDriver",
                        lambda cfg: (_ for _ in ()).throw(SystemExit(3)))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("enabled_apps: [hinge]\napps: {hinge: {serial: synthetic-pixel}}\n")
    args = argparse.Namespace(
        split="calibration", profiles=1, config=str(config_path), out=str(tmp_path / "out"),
        record_operational_checks=False, unattended=False, hybrid_review=False,
        confirmation="", send_like=False, send_like_confirmation="",
        reviewer_model="m", reviewer_process="p", reviewer_id="", reviewer_version="")

    with pytest.raises(SystemExit):
        cal._cmd_capture(args)

    assert calls == ["validate"], (
        "capture must call config.validate() (which installs the still-photo licence) "
        "before constructing the driver")
    assert cfg_mod_real is cal.cfg_mod  # the monkeypatched module is the real config module


def test_capture_marks_the_driver_device_driven_before_opening_the_phone_session(
        tmp_path, monkeypatch):
    """Calibration cannot fall through to the retired passive-Observe preflight.

    Capture has its own guarded heart/Pass protocol, not a Worker ranker policy, so it installs
    the explicit device-driven session marker with ``None`` before the driver's final live-phone
    gate runs.  The test stops at that gate: no ADB or card action is possible here.
    """
    events = []

    class _Driver:
        def __init__(self, _cfg):
            self.identity_band = _IDENTITY_BAND
            self.content_band = _CONTENT_BAND

        def set_auto_session_policy(self, policy):
            events.append(("session_policy", policy))

        def open_session(self):
            events.append(("open_session", None))
            raise RuntimeError("stop before device setup")

    monkeypatch.setattr(cal.cfg_mod, "load", lambda _path: SimpleNamespace(apps={}))
    monkeypatch.setattr(cal.cfg_mod, "validate", lambda _cfg: None)
    monkeypatch.setattr(cal, "HingeDriver", _Driver)
    monkeypatch.setattr(cal, "_preflight_serial", lambda _cfg: ("PIXEL-TEST", "adb"))
    monkeypatch.setattr(cal, "_capture_out_dir", lambda *_a, **_kw: tmp_path / "out")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("apps:\n  hinge:\n    serial: PIXEL-TEST\n")
    args = argparse.Namespace(
        split="calibration", profiles=1, config=str(config_path), out=str(tmp_path / "out"),
        record_operational_checks=False, unattended=False, hybrid_review=False,
        confirmation="", send_like=False, send_like_confirmation="",
        reviewer_model="m", reviewer_process="p", reviewer_id="", reviewer_version="")

    with pytest.raises(SystemExit):
        cal._cmd_capture(args)

    assert events == [("session_policy", None), ("open_session", None)]


# =====================================================================================
# photo-1-only: the second, explicitly-selected automated target strategy
# =====================================================================================
#
# WHY (owner decision 2026-08-22): two live campaigns showed real decks rarely carry three
# numberable photos (prompt cards/videos are common), so the alternating strategy burns its
# bounded per-ordinal skip budget on every even ordinal and no campaign can finish. This second
# strategy always targets photo model item 1, accepting it proves less about deep-item
# navigation. It is opt-in only (`capture --target-items photo-1-only`); the alternating id
# remains the byte-identical default in every respect covered above.

@pytest.mark.parametrize(
    ("ordinal", "expected"), [(1, (1,)), (2, (3,)), (3, (1,)), (4, (3,))])
def test_alternating_strategy_items_unchanged_whether_or_not_strategy_id_is_passed(
        ordinal, expected):
    """The default strategy's behaviour must stay byte-identical: passing no `strategy_id` and
    passing the alternating id explicitly must agree, for every ordinal parity."""
    assert cal._automated_composer_items_for_ordinal(ordinal) == expected
    assert cal._automated_composer_items_for_ordinal(
        ordinal, strategy_id=cal._AUTOMATED_TARGET_STRATEGY_ID) == expected


@pytest.mark.parametrize("ordinal", [1, 2, 3, 4])
def test_photo_1_only_strategy_targets_item_1_for_every_ordinal(ordinal):
    """The owner's second strategy narrows every ordinal -- odd or even -- to item 1 only. This
    is the explicit point of the strategy: it never exercises deep-item navigation."""
    assert cal._automated_composer_items_for_ordinal(
        ordinal, strategy_id=cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID) == (1,)


def test_unknown_target_strategy_id_raises_value_error_naming_it():
    """An unrecognized `strategy_id` must be refused loudly, never silently treated as one of the
    two known strategies (which would misrepresent what was actually targeted)."""
    with pytest.raises(ValueError, match="unknown automated target strategy id 'not_a_real_id'"):
        cal._automated_composer_items_for_ordinal(1, strategy_id="not_a_real_id")


def _fake_unattended_capture_recording_strategy(calls: list):
    """A stand-in for `_capture_one_profile_unattended` that records the exact
    `target_strategy_id` `_cmd_capture` threaded to it and returns a profile record carrying
    that same id and the items the real helper would compute for it -- so the manifest this
    produces is exactly what a real capture would record for that strategy."""
    def _capture(_driver, _out_dir, *, ordinal, frame_counter, frames_meta, used_profile_ids,
                review_gate, skipped_attempts, abort_recoveries, send_like, target_strategy_id):
        calls.append(target_strategy_id)
        used_profile_ids.add(f"profile-{ordinal}")
        items = cal._automated_composer_items_for_ordinal(
            ordinal, strategy_id=target_strategy_id)
        return ({"ordinal": ordinal, "profile_id": f"profile-{ordinal}",
                "target_strategy_id": target_strategy_id, "composer_items": list(items)},
               frame_counter)
    return _capture


def test_capture_with_photo_1_only_records_the_selected_strategy_in_every_recording_site(
        monkeypatch, tmp_path):
    """`--target-items photo-1-only` must be threaded to the manifest's
    `automated_target_strategy_id`, the `automation_acceptance.target_strategy_id`, AND each
    profile record's own `target_strategy_id` -- the manifest must record the strategy that
    actually ran, never the default."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("apps:\n  hinge:\n    serial: PIXEL-TEST\n")
    _wire_inert_skip_capture(monkeypatch, tmp_path)
    calls: list = []
    monkeypatch.setattr(
        cal, "_capture_one_profile_unattended", _fake_unattended_capture_recording_strategy(calls))
    args = argparse.Namespace(
        profiles=1, split="calibration", config=str(config_path), out=str(tmp_path),
        unattended=True, hybrid_review=False, confirmation=cal._UNATTENDED_CONFIRMATION,
        record_operational_checks=False, send_like=False, send_like_confirmation="",
        target_items="photo-1-only")

    cal._cmd_capture(args)

    assert calls == [cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["automated_target_strategy_id"] == cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID
    acceptance = manifest["automation_acceptance"]
    assert acceptance["target_strategy_id"] == cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID
    assert acceptance["targets_photo_model_items"] == [1]
    assert manifest["profiles"][0]["target_strategy_id"] == cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID
    assert manifest["profiles"][0]["composer_items"] == [1]


def test_capture_default_target_items_still_records_and_alternates(monkeypatch, tmp_path):
    """Regression guard for the default path: an `argparse.Namespace` built WITHOUT
    `target_items` at all (matching every pre-existing Namespace in this suite) must still
    record the alternating strategy id and must still alternate item 1/item 3 by ordinal --
    the new flag must not change default behaviour even by omission."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("apps:\n  hinge:\n    serial: PIXEL-TEST\n")
    _wire_inert_skip_capture(monkeypatch, tmp_path)
    calls: list = []
    monkeypatch.setattr(
        cal, "_capture_one_profile_unattended", _fake_unattended_capture_recording_strategy(calls))
    args = argparse.Namespace(
        profiles=2, split="calibration", config=str(config_path), out=str(tmp_path),
        unattended=True, hybrid_review=False, confirmation=cal._UNATTENDED_CONFIRMATION,
        record_operational_checks=False, send_like=False, send_like_confirmation="")
    assert not hasattr(args, "target_items")

    cal._cmd_capture(args)

    assert calls == [cal._AUTOMATED_TARGET_STRATEGY_ID] * 2
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["automated_target_strategy_id"] == cal._AUTOMATED_TARGET_STRATEGY_ID
    acceptance = manifest["automation_acceptance"]
    assert acceptance["target_strategy_id"] == cal._AUTOMATED_TARGET_STRATEGY_ID
    assert acceptance["targets_photo_model_items"] == [1, 3]
    assert [p["composer_items"] for p in manifest["profiles"]] == [[1], [3]]
    assert [p["target_strategy_id"] for p in manifest["profiles"]] == (
        [cal._AUTOMATED_TARGET_STRATEGY_ID] * 2)


def _minimal_unattended_manifest(*, target_strategy_id):
    """The smallest manifest shape `_verified_automated_circular_evidence` accepts, using
    ordinal 1 (photo item 1 under either known strategy) so the same builder serves both."""
    return {
        "capture_mode": "automated_circular_risk_accepted",
        "human_ground_truth": False,
        "automated_target_strategy_id": target_strategy_id,
        "automation_acceptance": {
            "confirmation": cal._UNATTENDED_CONFIRMATION,
            "target_strategy_id": target_strategy_id,
            "not_independent_ground_truth": True,
            "not_supervised_operational_evidence": True,
        },
        "profiles": [{
            "ordinal": 1,
            "target_strategy_id": target_strategy_id,
            "composer_items": [1],
            "automated_actions": [
                {"action": "automated_photo_heart", "photo_model_item": 1,
                 "post_tap_composer_verified": True, "post_tap_item_relative_verified": True},
                {"action": "automated_pass", "send_like_tapped": False,
                 "composer_clear_visible": True},
            ],
        }],
    }


def _session_from_manifest(tmp_path, name, manifest):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return cal._SessionData(directory, "calibration", "serial", (), (), [], manifest)


def test_measure_accepts_a_manifest_consistently_using_photo_1_only(tmp_path):
    """A capture that consistently used the second, explicitly-selected strategy end to end (the
    manifest, its acceptance, and its one profile record all agreeing) must be ACCEPTED exactly
    like an alternating-strategy manifest -- narrowing the evidence is an owner-authorized
    choice, not a defect that measurement should refuse."""
    manifest = _minimal_unattended_manifest(
        target_strategy_id=cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID)
    session = _session_from_manifest(tmp_path, "photo1only", manifest)

    result = cal._verified_automated_circular_evidence([session])

    assert result["kind"] == "automated_circular_risk_accepted"
    assert result["records"][0]["session"] == str(session.dir)


def test_measure_refuses_a_manifest_whose_profile_mixes_target_strategy_ids(tmp_path):
    """A manifest that consistently claims photo-1-only at its manifest/acceptance level but
    whose one profile record still claims the alternating id must be REFUSED -- membership in
    the accepted set alone is not enough; every reference within one session must agree with
    every other."""
    manifest = _minimal_unattended_manifest(
        target_strategy_id=cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID)
    manifest["profiles"][0]["target_strategy_id"] = cal._AUTOMATED_TARGET_STRATEGY_ID
    session = _session_from_manifest(tmp_path, "mixed", manifest)

    with pytest.raises(cal._MeasureRefused, match="lacks the exact automated"):
        cal._verified_automated_circular_evidence([session])


def test_measure_refuses_when_manifest_and_acceptance_strategy_ids_disagree(tmp_path):
    """The manifest-level `automated_target_strategy_id` and the
    `automation_acceptance.target_strategy_id` must agree with EACH OTHER, not merely each
    independently belong to the accepted set."""
    manifest = _minimal_unattended_manifest(
        target_strategy_id=cal._PHOTO_1_ONLY_TARGET_STRATEGY_ID)
    manifest["automation_acceptance"]["target_strategy_id"] = cal._AUTOMATED_TARGET_STRATEGY_ID
    session = _session_from_manifest(tmp_path, "top-level-mismatch", manifest)

    with pytest.raises(
            cal._MeasureRefused, match="not an exact unattended circular-risk capture manifest"):
        cal._verified_automated_circular_evidence([session])
