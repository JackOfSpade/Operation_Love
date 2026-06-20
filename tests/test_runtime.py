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


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
