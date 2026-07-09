"""Shared driver helpers (drivers/base.py) used by both Bumble and Hinge:
open_debug_log() and snapshot_failure_frame(). No prior test file exercised
these directly."""
from operation_love.drivers.base import open_debug_log, snapshot_failure_frame


def test_open_debug_log_returns_none_and_prints_on_construction_failure(monkeypatch, capsys):
    import operation_love.drivers.debuglog as debuglog_mod

    def _boom(base_dir):
        raise OSError("disk full")

    monkeypatch.setattr(debuglog_mod, "DebugLog", _boom)
    assert open_debug_log("./data/debug") is None
    assert "Debug log unavailable" in capsys.readouterr().out


def test_snapshot_failure_frame_noop_when_dbg_is_none():
    calls = []
    snapshot_failure_frame(None, RuntimeError("boom"), lambda: calls.append(1) or b"frame")
    assert calls == []                          # capture_frame never even called


def test_snapshot_failure_frame_logs_with_a_frame():
    logged = []

    class _Dbg:
        def error(self, name, frame, exc):
            logged.append((name, frame, exc))

    snapshot_failure_frame(_Dbg(), RuntimeError("boom"), lambda: b"frame-bytes")
    assert logged == [("unexpected", b"frame-bytes", logged[0][2])]
    assert isinstance(logged[0][2], RuntimeError)


def test_snapshot_failure_frame_logs_without_a_frame_when_capture_raises():
    logged = []

    class _Dbg:
        def error(self, name, frame, exc):
            logged.append((name, frame, exc))

    def _capture():
        raise RuntimeError("device gone")

    snapshot_failure_frame(_Dbg(), ValueError("original"), _capture)
    assert logged == [("unexpected", None, logged[0][2])]
    assert isinstance(logged[0][2], ValueError)


def test_snapshot_failure_frame_never_raises_even_if_dbg_error_raises():
    class _BrokenDbg:
        def error(self, name, frame, exc):
            raise RuntimeError("logging backend is down")

    # Must not raise -- this runs inside the worker's own except handler and must
    # never mask the real error it was called to report.
    snapshot_failure_frame(_BrokenDbg(), RuntimeError("original failure"), lambda: b"frame")
