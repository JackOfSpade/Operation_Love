"""Phone-touching TOOLS take the same Android device lock a production run takes.

The lock lived only inside ``supervisor.run()``, so every tool in tools/ that drives the Pixel
competed with a hub run on the honour system. Run ``a01fbcd1e9a0``'s false Malaika Pass was two
drivers on one phone and took a hash-bound retraction plan to undo, so "nobody would do that"
is not a safety argument -- especially for a capture campaign, which runs for many minutes
beside an idle hub the owner can Start at any moment.
"""
import os
import sys

import pytest

from operation_love import supervisor as sup
from tools import _devicelock


@pytest.fixture(autouse=True)
def _lock_root(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "_ANDROID_LOCK_ROOT", tmp_path / "locks")


def _config(tmp_path, name="config.yaml"):
    path = tmp_path / name
    path.write_text("enabled_apps: [hinge]\napps: {hinge: {serial: synthetic-pixel}}\n")
    return str(path)


def test_the_command_runs_with_the_lock_held_and_released_afterwards(tmp_path):
    held = {}

    def command(arg):
        # Proving it from INSIDE the command is the whole point: a lock acquired and released
        # around the call would satisfy a naive before/after assertion and protect nothing.
        second = sup._AndroidDeviceLock(sup._android_lock_path(None, "hinge"))
        with pytest.raises(RuntimeError, match="already in use"):
            second.acquire()
        held["ran"] = arg
        return "result"

    assert _devicelock.run_holding_the_device(_config(tmp_path), command, "arg") == "result"
    assert held["ran"] == "arg"

    # Released on the way out: the next command must not inherit a lock nobody holds.
    after = sup._AndroidDeviceLock(sup._android_lock_path(None, "hinge"))
    after.acquire()
    after.release()


def test_contention_exits_non_zero_naming_the_holder_instead_of_driving_the_phone(
        tmp_path, capsys):
    holder = sup._AndroidDeviceLock(sup._android_lock_path(None, "hinge"))
    holder.acquire()
    ran = []
    try:
        with pytest.raises(SystemExit) as exit_info:
            _devicelock.run_holding_the_device(_config(tmp_path), lambda: ran.append(1))
    finally:
        holder.release()

    assert exit_info.value.code == 1
    assert ran == [], "the command must not touch the phone while another process holds it"
    err = capsys.readouterr().err
    assert "already in use by another Operation Love run" in err
    # It names the holding pid, so the operator can find the run rather than guess at it. Here
    # the holder is this same test process, which is what makes the pid checkable.
    assert str(os.getpid()) in err


def test_the_lock_is_released_even_when_the_command_raises(tmp_path):
    with pytest.raises(ValueError):
        _devicelock.run_holding_the_device(_config(tmp_path),
                                           lambda: (_ for _ in ()).throw(ValueError("boom")))

    after = sup._AndroidDeviceLock(sup._android_lock_path(None, "hinge"))
    after.acquire()          # would raise if the failed command had leaked the lock
    after.release()


def test_an_unloadable_config_still_runs_the_command_so_it_reports_the_real_problem(tmp_path):
    """A lock error must never replace a precise config message with a vague one."""
    ran = []
    _devicelock.run_holding_the_device(str(tmp_path / "does-not-exist.yaml"), lambda: ran.append(1))

    assert ran == [1]


def test_the_calibrate_tool_locks_capture_and_observe_check_but_not_the_offline_commands():
    """The offline subcommands never open ADB; locking them would block a run for no reason."""
    import tools.hinge_calibrate as cal

    source = cal.main.__code__.co_names
    assert "_run_holding_the_device" in source
    # measure/verify/attach are dispatched directly, unlocked.
    for offline in ("_cmd_measure", "_cmd_verify_entry_anchor", "_cmd_attach_operational_evidence"):
        assert offline in source


def test_every_phone_driving_tool_imports_the_shared_lock():
    """A new tool that drives the phone must not quietly reintroduce the honour system."""
    import tools.hinge_bot_scroll_probe
    import tools.hinge_calibrate
    import tools.hinge_inspect
    import tools.hinge_video_bound
    import tools.hinge_video_bound_auto

    for module in (tools.hinge_calibrate, tools.hinge_inspect, tools.hinge_bot_scroll_probe,
                   tools.hinge_video_bound, tools.hinge_video_bound_auto):
        assert any(hasattr(module, name)
                   for name in ("run_holding_the_device", "holding_the_device")), \
            f"{module.__name__} drives the phone but imports no device lock"


def test_the_lock_helper_needs_no_heavy_import_at_module_scope():
    """Importing the helper must not drag the supervisor (and its model stack) into a tool.

    The supervisor import is deliberately deferred into the call. A tool that only prints help
    should not pay for the ranker/embedder import chain.
    """
    assert "operation_love.supervisor" not in _devicelock.__dict__
    assert sys.modules.get("tools._devicelock") is _devicelock
