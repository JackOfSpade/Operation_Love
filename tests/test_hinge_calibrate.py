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
from operation_love.drivers.scroll_top import ScrollTopVerdict


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


def _unattended_single_item_fixtures(monkeypatch):
    """Wire every seam a one-photo, ordinal=1 automated profile touches through to its terminal
    advance, without a real card scan/navigation stack.

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
            self.frames = iter([b"composer-1", b"pass-frame", b"advance-identity"])

        def screencap(self):
            return next(self.frames)

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
            self.scrolls.append((frac, x_frac))

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
            frame_sha256=cal._sha256(target.frame),
            dwell_frame_sha256s=(cal._sha256(target.frame), cal._sha256(b"dwell")),
            dwell_span_s=6.0, still_photo_verified=True,
            reattach_frame_sha256s=(cal._sha256(target.frame), cal._sha256(b"reattach")),
            reattach_dwell_span_s=6.0))
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
                frame_sha256=cal._sha256(b"some-other-frame"), dwell_frame_sha256s=("a", "b"),
                dwell_span_s=6.0, still_photo_verified=True)),
         "no still-photo .C1-C3. verdict is bound"),
        (lambda m: m.setattr(
            cal, "_verified_still_photo_proof",
            lambda *_a, **_kw: cal._StillPhotoProof(
                frame_sha256=cal._sha256(b"target-pre"), dwell_frame_sha256s=("a", "b"),
                dwell_span_s=6.0, still_photo_verified=False)),
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
                        probe_burst=None, probe_span_s=6.0):
    """A driver stub exposing only what the still-photo proof consumes: a dwell burst, the mute
    matcher, the content band the autoplay-centring precondition is measured against, and the
    re-attach probe.

    The FIRST burst issues no input; the probe does, and it hands back the settled frame it
    finished on. `probe=None` is the probe that could not complete, which must refuse the card
    rather than fall through to the first burst's verdict.
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
        _match_video_mute=lambda _frame, _rect: (True, mute_score))


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
    assert proof.frame_sha256 == cal._sha256(frame)
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
    fields = {"frame_sha256": cal._sha256(frame),
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


def test_the_still_photo_proof_refuses_a_probe_that_did_not_restore_the_screen(
        monkeypatch, installed_still_photo_bound):
    """`_fresh_reviewed_target_point` demands byte identity with the reviewed frame immediately
    before the tap, so a probe that left the page displaced would buy a checkpoint that could
    never be spent.  Refusing here means the reviewed frame, the navigator's point and both
    bursts always describe one screen."""
    monkeypatch.setattr(cal, "dwell_exact_over_rect", lambda *_a, **_kw: True)
    moved = SimpleNamespace(anchor=_action_frame(77), frames=(_action_frame(), _action_frame()),
                            span_s=6.0, page_shift_px=12)

    with pytest.raises(cal._CaptureAbort, match="did not restore the screen byte-for-byte"):
        cal._verified_still_photo_proof(
            _still_photo_driver(probe=moved), frame=_action_frame(),
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


@pytest.mark.parametrize(
    ("fresh_frame", "reviewed_point", "detected_point", "identity_matched"),
    [
        (b"changed", [500, 800], (500, 800), True),
        (b"target-pre", [501, 800], (500, 800), True),
        (b"target-pre", [500, 800], (501, 800), True),
        (b"target-pre", [500, 800], (500, 800), False),
    ],
    ids=["frame-changed", "plan-point-changed", "detected-point-changed", "identity-changed"],
)
def test_fresh_reviewed_target_refuses_changed_frame_point_or_identity_without_tapping(
        monkeypatch, fresh_frame, reviewed_point, detected_point, identity_matched):
    prior_identity = SimpleNamespace(identity=SimpleNamespace(), match_max=1.0)
    target = SimpleNamespace(
        frame=b"target-pre", point=(500, 800), block_frame_rows=(700, 900),
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


def test_fresh_reviewed_target_unchanged_revalidates_then_taps_once(monkeypatch):
    target = SimpleNamespace(
        frame=b"target-pre", point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: b"target-pre"), identity_band=_IDENTITY_BAND,
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
        frame=b"target-pre", point=(500, 800), block_frame_rows=(700, 900),
        identity=SimpleNamespace(identity=SimpleNamespace(), match_max=1.0))
    driver = SimpleNamespace(
        adb=SimpleNamespace(screencap=lambda: b"target-pre"), identity_band=_IDENTITY_BAND,
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
    assert trace["post_frame_sha256"] == hashlib.sha256(b"advance-frame").hexdigest()
    assert trace["predicates"] == {
        "inline_composer_and_selected_photo_verified_before_action": True,
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
        def __init__(self): self.adb, self.dislikes, self.taps = _Adb(), 0, 0
        def _template(self, _name): return object()
        def dislike(self):
            self.dislikes += 1
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
    monkeypatch.setattr(cal, "HingeDriver", _Driver)
    monkeypatch.setattr(cal, "_preflight_serial", lambda _cfg: ("PIXEL-TEST", "adb"))
    monkeypatch.setattr(cal, "_capture_out_dir", lambda *_a, **_kw: tmp_path)


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
