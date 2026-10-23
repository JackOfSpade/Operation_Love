"""The offline calibration commands replay frames with the like glyph of the captured build.

``measure``, ``verify-entry-anchor`` and ``attach-operational-evidence`` build ``HingeDriver``
without ADB, so the driver never reads the live build and ``_template("like")`` falls back to the
pre-10.8 heart.  These tests use a real ``HingeDriver`` and the shipped glyph assets; only the
session loading and the downstream evidence pipelines are replaced at their boundaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from operation_love.drivers import hinge as hinge_mod
from tools import hinge_calibrate as cal

_SERIAL = "PIXEL-TEST"
_OLD_GLYPH = "hinge_like_button.png"
_NEW_GLYPH = "hinge_like_button_10_8.png"


def _cfg():
    return SimpleNamespace(apps={"hinge": {"serial": _SERIAL}})


class _RecordingDriver(hinge_mod.HingeDriver):
    """A real ADB-less HingeDriver that remembers every like template it handed out."""

    instances: list[_RecordingDriver] = []

    def __init__(self, cfg):
        super().__init__(cfg)
        self.like_templates = []
        _RecordingDriver.instances.append(self)

    def _template(self, role):
        template = super()._template(role)
        if role == "like":
            self.like_templates.append(template)
        return template


@pytest.fixture
def driver_cls(monkeypatch):
    _RecordingDriver.instances = []
    monkeypatch.setattr(cal, "HingeDriver", _RecordingDriver)
    monkeypatch.setattr(cal.cfg_mod, "load", lambda _path: _cfg())
    return _RecordingDriver


def _glyph(name):
    template = hinge_mod._load_template(name)
    assert template is not None
    return template


def _device(version):
    device = {
        "serial": _SERIAL,
        "model": "Synthetic Pixel",
        "display_w": 1080,
        "display_h": 2400,
        "density": 420,
        "hinge_package": "co.hinge.app",
    }
    if version is not None:
        device["hinge_version_name"] = version
    return device


def _session(directory: Path, split: str, version, **manifest_extra):
    directory.mkdir(parents=True)
    probe = hinge_mod.HingeDriver(_cfg())
    return cal._SessionData(
        dir=directory,
        split=split,
        device_serial=_SERIAL,
        identity_band=tuple(probe.identity_band),
        content_band=tuple(probe.content_band),
        profiles=[],
        manifest={"device": _device(version), **manifest_extra},
    )


def _measure(monkeypatch, sessions: dict):
    monkeypatch.setattr(cal, "_load_session", lambda raw: sessions[str(raw)])
    cal._cmd_measure(argparse.Namespace(config="x", sessions=list(sessions)))


def test_adb_less_driver_starts_without_a_build_and_selects_the_old_glyph():
    """The precondition every test below depends on: without a bound build, the old heart."""
    driver = hinge_mod.HingeDriver(_cfg())

    assert driver._targeting_runtime_version_name is None
    assert np.array_equal(driver._template("like"), _glyph(_OLD_GLYPH))
    assert not np.array_equal(_glyph(_OLD_GLYPH), _glyph(_NEW_GLYPH))


@pytest.mark.parametrize(
    ("version", "glyph"),
    [
        ("10.8.0", _NEW_GLYPH),
        ("10.9.0", _NEW_GLYPH),
        ("10.7.0", _OLD_GLYPH),
        ("10.0.1", _OLD_GLYPH),
    ],
)
def test_measure_takes_the_like_glyph_of_the_manifest_build(
    monkeypatch, tmp_path, capsys, driver_cls, version, glyph
):
    session = _session(tmp_path / "cal", "calibration", version)

    # A lone calibration split is refused just after the template is taken.
    with pytest.raises(SystemExit):
        _measure(monkeypatch, {"cal": session})
    assert "each split" in capsys.readouterr().err.lower()

    (driver,) = driver_cls.instances
    assert driver._targeting_runtime_version_name == version
    assert len(driver.like_templates) == 1
    assert np.array_equal(driver.like_templates[0], _glyph(glyph))


def test_measure_refuses_sessions_that_disagree_on_the_build(
    monkeypatch, tmp_path, capsys, driver_cls
):
    sessions = {
        "cal": _session(tmp_path / "cal", "calibration", "10.9.0"),
        "held": _session(tmp_path / "held", "heldout", "10.7.0"),
    }

    with pytest.raises(SystemExit) as exc:
        _measure(monkeypatch, sessions)

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "disagree on device.hinge_version_name" in err
    assert "'10.9.0'" in err and "'10.7.0'" in err
    (driver,) = driver_cls.instances
    assert driver.like_templates == []
    assert driver._targeting_runtime_version_name is None


@pytest.mark.parametrize("version", [None, "", " 10.9.0", "10.9.0\n", 10.9])
def test_measure_refuses_a_session_without_a_usable_build(
    monkeypatch, tmp_path, capsys, driver_cls, version
):
    sessions = {
        "cal": _session(tmp_path / "cal", "calibration", "10.9.0"),
        "held": _session(tmp_path / "held", "heldout", version),
    }

    with pytest.raises(SystemExit) as exc:
        _measure(monkeypatch, sessions)

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "lacks a usable device.hinge_version_name" in err
    assert str(tmp_path / "held") in err
    (driver,) = driver_cls.instances
    assert driver.like_templates == []
    assert driver._targeting_runtime_version_name is None


def _offline_session_dir(monkeypatch, tmp_path) -> Path:
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "ops" / "calibration" / "targeting_10_9"
    return directory


def _stop_at_ledger(monkeypatch, captured: list):
    def _write(_out_dir, **kwargs):
        captured.append(kwargs["like_template"])
        raise cal._MeasureRefused("stop after the replay template is chosen")

    monkeypatch.setattr(cal, "_write_entry_anchor_ledger", _write)


@pytest.mark.parametrize(
    ("version", "glyph"), [("10.9.0", _NEW_GLYPH), ("10.7.0", _OLD_GLYPH)]
)
def test_verify_entry_anchor_replays_with_the_manifest_build_glyph(
    monkeypatch, tmp_path, capsys, driver_cls, version, glyph
):
    directory = _offline_session_dir(monkeypatch, tmp_path)
    session = _session(directory, "calibration", version, frames=[], profiles=[])
    (directory / "manifest.json").write_text("{}")
    monkeypatch.setattr(cal, "_load_session", lambda *_a, **_kw: session)
    captured: list = []
    _stop_at_ledger(monkeypatch, captured)

    with pytest.raises(SystemExit):
        cal._cmd_verify_entry_anchor(
            argparse.Namespace(session="ops/calibration/targeting_10_9", config="x")
        )

    assert "stop after the replay template is chosen" in capsys.readouterr().err
    (template,) = captured
    assert np.array_equal(template, _glyph(glyph))


def test_verify_entry_anchor_refuses_a_manifest_without_a_build(
    monkeypatch, tmp_path, capsys, driver_cls
):
    directory = _offline_session_dir(monkeypatch, tmp_path)
    session = _session(directory, "calibration", None, frames=[], profiles=[])
    (directory / "manifest.json").write_text("{}")
    monkeypatch.setattr(cal, "_load_session", lambda *_a, **_kw: session)
    monkeypatch.setattr(
        cal,
        "_write_entry_anchor_ledger",
        lambda *_a, **_kw: pytest.fail("a ledger must not be replayed without a build"),
    )

    with pytest.raises(SystemExit) as exc:
        cal._cmd_verify_entry_anchor(
            argparse.Namespace(session="ops/calibration/targeting_10_9", config="x")
        )

    assert exc.value.code == 1
    assert "lacks a usable device.hinge_version_name" in capsys.readouterr().err
    (driver,) = driver_cls.instances
    assert driver.like_templates == []


def _attachable_session(directory: Path, version):
    checks = {
        key: {"confirmed": True, "evidence": "synthetic", "recorded_utc": "2026-10-23"}
        for key in (
            "entry_anchor_scroll_ledger",
            "item_1_inline_composer_identity",
            "gesture_transport",
        )
    }
    return _session(
        directory,
        "calibration",
        version,
        frames=[],
        profiles=[],
        operational_checks=checks,
    )


def _wire_attach(monkeypatch, session, tmp_path):
    monkeypatch.setattr(cal, "_load_session", lambda *_a, **_kw: session)
    monkeypatch.setattr(
        cal, "_operational_evidence_reference_reason", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        cal,
        "_operational_evidence_manifest_path",
        lambda _ref: (tmp_path / "operational" / "manifest.json", None),
    )


@pytest.mark.parametrize(
    ("version", "glyph"), [("10.8.0", _NEW_GLYPH), ("10.7.0", _OLD_GLYPH)]
)
def test_attach_operational_evidence_replays_with_the_manifest_build_glyph(
    monkeypatch, tmp_path, capsys, driver_cls, version, glyph
):
    directory = _offline_session_dir(monkeypatch, tmp_path)
    session = _attachable_session(directory, version)
    _wire_attach(monkeypatch, session, tmp_path)
    captured: list = []
    _stop_at_ledger(monkeypatch, captured)

    with pytest.raises(SystemExit):
        cal._cmd_attach_operational_evidence(
            argparse.Namespace(
                session="ops/calibration/targeting_10_9",
                config="x",
                operational_evidence="synthetic",
            )
        )

    assert "stop after the replay template is chosen" in capsys.readouterr().err
    (template,) = captured
    assert np.array_equal(template, _glyph(glyph))


def test_attach_operational_evidence_refuses_a_manifest_without_a_build(
    monkeypatch, tmp_path, capsys, driver_cls
):
    directory = _offline_session_dir(monkeypatch, tmp_path)
    session = _attachable_session(directory, "")
    _wire_attach(monkeypatch, session, tmp_path)
    monkeypatch.setattr(
        cal,
        "_write_entry_anchor_ledger",
        lambda *_a, **_kw: pytest.fail("a ledger must not be replayed without a build"),
    )

    with pytest.raises(SystemExit) as exc:
        cal._cmd_attach_operational_evidence(
            argparse.Namespace(
                session="ops/calibration/targeting_10_9",
                config="x",
                operational_evidence="synthetic",
            )
        )

    assert exc.value.code == 1
    assert "lacks a usable device.hinge_version_name" in capsys.readouterr().err
    (driver,) = driver_cls.instances
    assert driver.like_templates == []
