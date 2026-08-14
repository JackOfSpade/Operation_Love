"""Focused safety pins for the passive Hinge observe-check evidence tool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operation_love.drivers.like_composer import ComposerDetectionError, ComposerSurface, Rect
from tools import hinge_calibrate as calibration


def _cfg(*, enabled: bool = True):
    return SimpleNamespace(opener=SimpleNamespace(
        enabled=enabled,
        provider="gemini",
        effective_models=["gemini-test"],
        request_timeout_s=12.0,
    ))


def test_synthetic_quota_probe_contains_no_profile_or_item_request(monkeypatch):
    seen = {}

    class Client:
        api_key = "not-recorded"

        @staticmethod
        def transport(url, payload, headers, timeout):
            seen.update(url=url, payload=payload, headers=headers, timeout=timeout)
            return 200, {"candidates": [{"content": {"parts": [{"text": '{"probe":"ok"}'}]}}]}

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(calibration, "GeminiOpener", lambda *a, **kw: Client())

    result = calibration._synthetic_gemini_quota_probe(_cfg())

    assert result["outcome"] == "ok"
    assert result["http_code"] == 200
    assert "gemini-test:generateContent" in seen["url"]
    assert seen["headers"]["X-goog-api-key"] == "not-recorded"
    payload_text = json.dumps(seen["payload"])
    assert "Synthetic Operation Love observe-only quota check" in payload_text
    assert "inlineData" not in payload_text
    assert "image" not in payload_text.lower()
    assert "response" not in result
    assert "test-key" not in json.dumps(result)


def test_observe_check_writes_hashed_evidence_and_only_uses_passive_driver_calls(
    monkeypatch, tmp_path: Path,
):
    calls = []
    frames = iter([
        b"pass-before", b"pass-after", b"heart-before", b"composer", b"send-before", b"send-after",
    ])

    class Adb:
        @staticmethod
        def screencap():
            calls.append("screencap")
            return next(frames)

    class Driver:
        adb = Adb()

        def set_opener_enabled(self, enabled):
            calls.append(("set_opener_enabled", enabled))

        def open_session(self):
            calls.append("open_session")

        def close(self):
            calls.append("close")

        @staticmethod
        def _template(role):
            assert role == "confirm"
            return object()

        @staticmethod
        def _observe_deck_ready(frame):
            return frame == b"send-after"

        # Any accidental action path fails the test before it can reach a fake transport.
        def like(self, *args, **kwargs):  # pragma: no cover - a tripwire
            raise AssertionError("observe-check must not call like")

        def dislike(self, *args, **kwargs):  # pragma: no cover - a tripwire
            raise AssertionError("observe-check must not call dislike")

    surface = ComposerSurface("hinge_inline_v1", Rect(1, 2, 3, 4), Rect(5, 6, 7, 8), (6, 7))

    monkeypatch.setattr(calibration.cfg_mod, "load", lambda _path: _cfg())
    monkeypatch.setattr(calibration, "HingeDriver", lambda _cfg: Driver())
    monkeypatch.setattr(calibration, "_capture_out_dir", lambda *_a, **_kw: tmp_path)
    monkeypatch.setattr(calibration, "_preflight_serial", lambda _cfg: ("pixel", "adb"))
    monkeypatch.setattr(calibration, "_device_evidence", lambda _driver: {
        "serial": "pixel", "model": "Pixel", "display_w": 1080, "display_h": 2400,
        "hinge_package": "co.hinge.app", "hinge_version_name": "9.134.0",
    })
    monkeypatch.setattr(calibration, "_synthetic_gemini_quota_probe", lambda _cfg: {
        "kind": "synthetic_non_personal_gemini_generate_content", "outcome": "ok", "http_code": 200,
    })

    def detect(frame, _template, *, threshold):
        assert threshold == 0.8
        if frame == b"composer":
            return surface
        raise ComposerDetectionError("composer absent")

    monkeypatch.setattr(calibration, "locate_inline_composer", detect)
    answers = iter(["READY", "PASSED", "READY", "COMPOSER", "READY", "SENT"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    calibration._cmd_observe_check(argparse.Namespace(
        config="unused.yaml", out=str(tmp_path), supervised=True,
        confirmation="OBSERVE_ONLY",
    ))

    assert calls == [("set_opener_enabled", False), "open_session", "screencap", "screencap",
                     "screencap", "screencap", "screencap", "screencap", "close"]
    evidence = json.loads((tmp_path / "observe_check.json").read_text())
    assert evidence["completed"] is True
    assert evidence["schema_version"] == 2
    assert evidence["evidence_scope"] == "pre_calibration_passive_manual_cycle_and_synthetic_quota_v1"
    assert evidence["release_status"] == "preliminary_only_not_auto_authorization"
    assert "production_store_label_persistence" in evidence["does_not_prove"]
    assert "persisted_manual_label_in_production_store" in evidence["required_before_auto"]
    assert evidence["passive_guards"] == {
        "driver_gesture_text_or_send_called": False,
        "enumeration_enabled": False,
        "hub_pre_tap_state_checked": False,
        "opener_service_called": False,
        "profile_or_item_request_sent": False,
        "production_store_persistence_checked": False,
        "worker_constructed": False,
    }
    assert [event["event"] for event in evidence["events"]] == [
        "before_manual_pass", "after_manual_pass", "before_manual_heart",
        "after_manual_heart_inline_composer", "before_manual_send", "after_manual_send",
    ]
    assert evidence["events"][3]["comment_rect"] == [1, 2, 3, 4]
    assert evidence["events"][5]["inline_composer_cleared"] is True
    body = dict(evidence)
    digest = body.pop("evidence_sha256")
    assert digest == calibration._canonical_json_digest(body)
    for event in evidence["events"]:
        assert "frame_sha256" in event and "frame_bytes" in event
        assert "frame" not in event


def test_observe_check_requires_explicit_supervised_acknowledgement(capsys, tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        calibration._cmd_observe_check(argparse.Namespace(
            config="unused.yaml", out=str(tmp_path), supervised=False, confirmation="",
        ))
    assert exc.value.code == 2
    assert "--supervised" in capsys.readouterr().err


def test_measure_accepts_valid_preliminary_observe_check_for_numeric_candidate(monkeypatch,
                                                                                tmp_path: Path):
    preliminary = tmp_path / "observe_check.json"
    record = {
        "kind": "hinge_supervised_observe_only_check",
        "evidence_scope": "pre_calibration_passive_manual_cycle_and_synthetic_quota_v1",
        "completed": True,
    }
    record["evidence_sha256"] = calibration._canonical_json_digest(record)
    preliminary.write_text(json.dumps(record))
    checks = {
        key: {"confirmed": True, "evidence": "data/hinge_debug/production-run",
              "recorded_utc": "2026-08-13T00:00:00+00:00"}
        for key in calibration._OPERATIONAL_CHECKS
    }
    checks["observe_timing_and_quota"]["evidence"] = str(preliminary)
    session = SimpleNamespace(manifest={"operational_checks": checks}, dir=tmp_path)
    # This pin exercises the preliminary artifact only.  The calibration-harness suite carries
    # real recorder-v2 and entry-ledger fixtures for the other independent gates.
    monkeypatch.setattr(calibration, "_entry_anchor_reference_reason", lambda *_a, **_kw: None)
    monkeypatch.setattr(calibration, "_operational_evidence_reference_reason", lambda *_a, **_kw: None)
    monkeypatch.setattr(calibration, "_operational_evidence_manifest_path",
                        lambda _ref: (tmp_path / "operational" / "manifest.json", None))
    session.manifest["device"] = {
        "serial": "pixel", "model": "Pixel", "display_w": 1080, "display_h": 2400,
        "density": "420", "hinge_package": "co.hinge.app", "hinge_version_name": "9.134.0",
    }
    session.identity_band = (0.1, 0.1, 0.8, 0.2)

    verified = calibration._verified_operational_checks([session])
    assert verified["observe_timing_and_quota"][0]["evidence"] == str(preliminary)


def test_measure_refuses_free_form_or_missing_preliminary_observe_reference(tmp_path: Path):
    reason = calibration._preliminary_observe_check_reference_reason(
        str(tmp_path / "someone-said-it-passed"))

    assert reason is not None
    assert "free-form path or assertion" in reason
