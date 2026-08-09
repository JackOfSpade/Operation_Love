"""Runtime/device detection tests — run on any OS, no torch/GPU required."""
import os

from operation_love.device import best_device
from operation_love.runtime import Capabilities


def test_best_device_returns_valid():
    assert best_device() in {"cpu", "cuda", "mps"}


def test_env_override(monkeypatch=None):
    os.environ["OPLOVE_DEVICE"] = "cpu"
    try:
        assert best_device() == "cpu"
    finally:
        del os.environ["OPLOVE_DEVICE"]


def test_prefer_arg_wins():
    assert best_device(prefer="cpu") == "cpu"


def test_capabilities_detect():
    caps = Capabilities.detect()
    assert caps.os_name
    assert caps.device in {"cpu", "cuda", "mps"}
    # every optional component is reported as a bool
    for key in ("torch", "clip", "arcface", "quality", "anthropic"):
        assert isinstance(caps.available[key], bool)


def test_missing_helper():
    caps = Capabilities.detect()
    caps.available["fake_component"] = False
    assert "fake_component" in caps.missing("fake_component")


def test_banner_is_str():
    assert isinstance(Capabilities.detect().banner(), str)


def test_capabilities_detect_honours_android_adb_path(tmp_path, monkeypatch):
    # adb not on PATH at all -> default (config-free) check reports it missing.
    empty_dir = tmp_path / "empty_path"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))
    caps = Capabilities.detect()
    assert caps.available["android_driver"] is False
    assert caps.available["hinge_driver"] is False   # pre-rename alias mirrors the new key

    # But a configured apps.<app>.adb_path pointing at a real binary must be honoured,
    # even though it's still not on PATH -- for Hinge OR a future calibrated Bumble.
    fake_adb = tmp_path / "adb"
    fake_adb.write_text("#!/bin/sh\n")
    fake_adb.chmod(0o755)
    caps = Capabilities.detect(android_adb_path=str(fake_adb))
    assert caps.available["android_driver"] is True
    assert caps.available["hinge_driver"] is True


def test_capabilities_detect_still_accepts_legacy_hinge_adb_path_kwarg(tmp_path, monkeypatch):
    # Backwards compatibility: callers that haven't migrated off the pre-rename kwarg name
    # must keep working.
    empty_dir = tmp_path / "empty_path"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))
    fake_adb = tmp_path / "adb"
    fake_adb.write_text("#!/bin/sh\n")
    fake_adb.chmod(0o755)
    caps = Capabilities.detect(hinge_adb_path=str(fake_adb))
    assert caps.available["android_driver"] is True


def test_capabilities_available_exposes_transport_shaped_keys():
    # web_driver/android_driver are the current names (platforms.py's kind vocabulary);
    # bumble_driver/hinge_driver are kept only as aliases for old callers.
    caps = Capabilities.detect()
    for key in ("web_driver", "android_driver"):
        assert isinstance(caps.available[key], bool)
    assert caps.available["bumble_driver"] == caps.available["web_driver"]
    assert caps.available["hinge_driver"] == caps.available["android_driver"]
