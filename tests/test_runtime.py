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


def test_capabilities_detect_honours_hinge_adb_path(tmp_path, monkeypatch):
    # adb not on PATH at all -> default (config-free) check reports it missing.
    empty_dir = tmp_path / "empty_path"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))
    assert Capabilities.detect().available["hinge_driver"] is False

    # But a configured apps.hinge.adb_path pointing at a real binary must be honoured,
    # even though it's still not on PATH.
    fake_adb = tmp_path / "adb"
    fake_adb.write_text("#!/bin/sh\n")
    fake_adb.chmod(0o755)
    caps = Capabilities.detect(hinge_adb_path=str(fake_adb))
    assert caps.available["hinge_driver"] is True
