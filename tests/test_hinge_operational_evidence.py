"""Hermetic tests for the read-only Hinge operational-evidence recorder."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat

import pytest

from operation_love.drivers.item_identity import ProfileIdentity
from operation_love.drivers.like_composer import ComposerDetectionError, ComposerSurface, Rect
from operation_love.drivers.scroll_top import ScrollTopVerdict
from tools import hinge_operational_evidence as evidence


class _Adb:
    """Only exposes the recorder's allowed transport method."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def screencap(self):
        self.calls += 1
        return self.frames.pop(0)


def _answers(_prompt):
    return ""


def _top(*, confirmed=False, refuted=False):
    state = "confirmed_top" if confirmed else "confirmed_not_top" if refuted else "cannot_tell"
    return ScrollTopVerdict(state, 0.0 if confirmed else 20.0, state, (0.1, 0.1, 0.8, 0.2),
                            (16, 4), 3.0, 9.0)


def _surface():
    return ComposerSurface("hinge_inline_v1", Rect(95, 1597, 985, 1775),
                           Rect(390, 1807, 985, 1916), (695, 1856))


def _device():
    return {"serial": "pixel", "model": "Synthetic Pixel", "display_w": 1080,
            "display_h": 2400, "density": "420", "hinge_package": "co.hinge.app",
            "hinge_version_name": "9.134.0"}


def _wire_success(monkeypatch):
    monkeypatch.setattr(
        evidence, "confirm_scroll_top",
        lambda frame, **_kw: _top(confirmed=frame in (b"top", b"new-top"),
                                  refuted=frame in (b"initial", b"stable", b"moved", b"new-sticky")))
    monkeypatch.setattr(evidence, "locate_inline_composer",
                        lambda frame, *_args, **_kw: (_surface() if frame != b"new-top"
                                                       else (_ for _ in ()).throw(
                                                           ComposerDetectionError("absent"))))
    identity = ProfileIdentity(fingerprint=(10, 10), band=(0.1, 0.1, 0.8, 0.2), grid=(64, 16),
                               frame_index=1, scroll_top_distance=20.0, reason="corroborated")
    monkeypatch.setattr(evidence, "capture_profile_identity", lambda *_args, **_kw: identity)
    monkeypatch.setattr(evidence, "band_fingerprint", lambda *_args, **_kw: (30, 30))


def test_records_all_explicit_roles_with_hashes_and_pure_check_evidence(monkeypatch, tmp_path):
    _wire_success(monkeypatch)
    adb = _Adb([b"top", b"initial", b"stable", b"moved", b"new-top", b"new-sticky"])

    manifest = evidence.record(
        adb=adb, serial="pixel", identity_band=(0.1, 0.1, 0.8, 0.2), confirm_template=object(),
        out_dir=tmp_path, device=_device(), ask=_answers)

    assert manifest["completed"] is True and manifest["interrupted"] is False
    assert [frame["role"] for frame in manifest["frames"]] == list(evidence._ROLES)
    assert [frame["file"] for frame in manifest["frames"]] == [
        "00001.png", "00002.png", "00003.png", "00004.png", "00005.png", "00006.png"]
    assert manifest["analyses"]["new_profile_top_clear"]["composer_absent"] is True
    assert manifest["analyses"]["new_sticky_identity"]["distance_from_item1_profile"] == 20.0
    assert manifest["device"] == _device()
    assert manifest["frame_size_px"] == [1080, 2400]
    assert manifest["config_binding"] == {
        "serial": "pixel", "identity_band": [0.1, 0.1, 0.8, 0.2],
        "composer_layout_id": "hinge_inline_v1", "hinge_package": "co.hinge.app",
    }
    body = dict(manifest)
    digest = body.pop("evidence_sha256")
    assert digest == evidence._canonical_json_digest(body)
    for frame in manifest["frames"]:
        raw = (tmp_path / frame["file"]).read_bytes()
        assert frame["sha256"] == hashlib.sha256(raw).hexdigest()
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    if os.name == "posix":
        assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600
                   for path in tmp_path.iterdir() if path.is_file())


def test_refuses_and_marks_manifest_incomplete_when_the_new_top_still_has_a_composer(
        monkeypatch, tmp_path):
    _wire_success(monkeypatch)
    monkeypatch.setattr(evidence, "locate_inline_composer", lambda *_args, **_kw: _surface())
    adb = _Adb([b"top", b"initial", b"stable", b"moved", b"new-top"])

    with pytest.raises(evidence.EvidenceRefused, match="still structurally contains"):
        evidence.record(adb=adb, serial="pixel", identity_band=(0.1, 0.1, 0.8, 0.2),
                        confirm_template=object(), out_dir=tmp_path, device=_device(), ask=_answers)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["completed"] is False and manifest["interrupted"] is False
    assert manifest["frame_count"] == 5


def test_refuses_if_new_sticky_identity_is_not_distinct_from_item1_profile(monkeypatch, tmp_path):
    _wire_success(monkeypatch)
    monkeypatch.setattr(evidence, "band_fingerprint", lambda *_args, **_kw: (11, 11))
    adb = _Adb([b"top", b"initial", b"stable", b"moved", b"new-top", b"new-sticky"])

    with pytest.raises(evidence.EvidenceRefused, match="same-profile scroll"):
        evidence.record(adb=adb, serial="pixel", identity_band=(0.1, 0.1, 0.8, 0.2),
                        confirm_template=object(), out_dir=tmp_path, device=_device(), ask=_answers)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["completed"] is False
    assert manifest["frame_count"] == 6


def test_recorder_source_has_no_input_injection_surface():
    source = inspect.getsource(evidence)
    for forbidden in (".tap(", ".swipe(", ".scroll_up(", ".text("):
        assert forbidden not in source
    assert "composer_unfocused" not in source
    assert "composer_focused" not in source


def test_private_output_guard_refuses_any_directory_outside_calibration_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(evidence.EvidenceRefused, match="gitignored"):
        evidence._out_dir(str(tmp_path / "not-private"))

    assert not (tmp_path / "not-private").exists()
