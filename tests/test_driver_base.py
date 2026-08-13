"""Shared driver helpers (drivers/base.py) used by both Bumble and Hinge:
open_debug_log() and snapshot_failure_frame(). No prior test file exercised
these directly."""
from operation_love.drivers.base import DatingAppDriver, open_debug_log, snapshot_failure_frame


# --- blocked_reason() default (added 2026-08-11, see THE INCIDENT in worker.py/status.py) --
# A minimal CONCRETE driver implementing only the ABC's actual @abstractmethod set
# (open_session/next_profile/like/dislike/out_of_profiles) and deliberately saying nothing
# at all about blocked_reason -- exactly the situation every pre-existing driver (Bumble
# Android, the Playwright web drivers) is in today, since blocked_reason() was added to the
# ABC as a concrete method with a body, not an abstract one (see base.py's own comment on
# why: "every non-Hinge driver ... must be completely unaffected by its existence"). If
# blocked_reason() had been declared abstract instead, this class would fail to instantiate
# at all -- TypeError: Can't instantiate abstract class ... with abstract method
# blocked_reason -- which is exactly the regression this class's mere existence rules out.
class _MinimalConcreteDriver(DatingAppDriver):
    def open_session(self) -> None:
        pass

    def next_profile(self, *, should_stop=None):
        return None

    # Tracks base.Driver.like exactly. The `anchored_opener` keyword this used to carry was
    # removed from the ABC on 2026-08-12 (ops/OPENER-REDESIGN.md 5.6: a driver that cannot land
    # the like on the chosen item raises ItemTargetingError instead of repairing the text against
    # whatever it hit).
    def like(self, opener=None, item_index=None, *, model_item_index=None) -> None:
        pass

    def dislike(self) -> None:
        pass

    def out_of_profiles(self) -> bool:
        return True


def test_blocked_reason_is_not_abstract_and_a_silent_driver_still_instantiates():
    """Pins that blocked_reason() is a concrete method with a safe default body, NOT an
    abstract one -- a driver that implements only the ABC's real abstract methods and never
    mentions blocked_reason must still construct cleanly. Guards against a future edit that
    turns it into @abstractmethod, which would break every driver written before this field
    existed (Bumble Android, the Playwright web drivers) the moment they're instantiated."""
    driver = _MinimalConcreteDriver()          # must not raise
    assert isinstance(driver, DatingAppDriver)


def test_blocked_reason_defaults_to_none_for_an_uncalibrated_driver():
    """The safe default for every driver that has never been calibrated for a blocking
    screen (Bumble Android, the Playwright web drivers, and any future driver that simply
    doesn't override this) is None -- 'honestly does not know whether one is up', per
    blocked_reason's own docstring -- not a guess dressed up as one, and not an exception.
    Only Hinge (drivers/hinge.py) overrides this; every other driver must be completely
    unaffected by the method's mere existence on the ABC."""
    driver = _MinimalConcreteDriver()
    assert driver.blocked_reason() is None
    # Sanity: the rest of the ABC's default-implementation contract is untouched by this
    # addition -- a driver that also says nothing about observe mode still fails loudly
    # there, exactly as before blocked_reason() existed.
    import pytest
    with pytest.raises(NotImplementedError):
        driver.current_profile()


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
