"""One place every phone-touching tool takes the run's Android device lock.

ADDED 2026-08-22.  The cross-process lock that stops two drivers reaching the one physical
Pixel lived only inside ``supervisor.run()``, so every offline tool here -- the calibration
capture, the video-bound campaigns, the inspect/scroll probes -- competed with a hub run on the
honour system.  That is not a theoretical collision: run ``a01fbcd1e9a0``'s false Malaika Pass
was two drivers on one phone, and undoing it took a hash-bound retraction plan.

A capture campaign is the worst case for an honour system, because it runs for many minutes
beside an idle hub the owner can Start at any moment, and its risky window includes the
reviewer's think time between checkpoints -- when the phone sits mid-profile with a composer
possibly open.  So the lock is held for a whole command, never just around ``open_session()``.

Tools that never open ADB (every ``measure``/``verify``/``attach`` subcommand here) deliberately
do NOT take it: locking them would block a production run for no reason.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager

from operation_love import config as cfg_mod


@contextmanager
def _nothing():
    yield None


def holding_the_device(config_path: str, *, app: str = "hinge"):
    """Context manager holding the same lock a production run holds, for this whole command.

    Contention raises out of the ``with`` as a ``RuntimeError`` naming the holding pid; use
    :func:`run_holding_the_device` if you want that turned into a clean non-zero exit.

    A config that will not load yields an un-held context rather than raising here.  The caller
    is about to load the same file and report the problem in its own words, and a lock error
    would replace a precise config message with a vague one.
    """
    from operation_love.supervisor import exclusive_android_device
    try:
        cfg = cfg_mod.load(config_path)
    except Exception:  # noqa: BLE001 -- see docstring: the caller reports this better
        return _nothing()
    return exclusive_android_device(cfg, app)


def run_holding_the_device(config_path: str, command, *args, app: str = "hinge", **kwargs):
    """Run ``command(*args, **kwargs)`` under the device lock; exit non-zero if it is held.

    The contention message already names the holding pid and says what to do about it, so it is
    printed as-is rather than wrapped in a second explanation.
    """
    try:
        with holding_the_device(config_path, app=app):
            return command(*args, **kwargs)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
