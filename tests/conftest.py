"""Suite-wide isolation for state that lives in a module-global slot.

Only one thing belongs here so far, and it is here because it is genuinely process-global:
the still-photo numbering licence in ``operation_love.targeting_policy``.  There is exactly ONE
slot for it (see the comment above ``_installed_still_photo_licence``), and ``config.validate()``
installs into that slot as a side effect whenever the config it is handed carries
``apps.hinge.still_photo_bound_evidence`` or ``apps.hinge.still_photo_assumption_acceptance``.
The repository's real config.yaml carries the acceptance key, so any test that validates the
REAL config -- tests/test_config_yaml_real.py does, by design -- leaves numbering licensed for
every test that runs after it in the same process.

That leak is invisible when a file is run alone and lethal when files are combined: targeting is
fail-closed by default, so a leaked licence silently flips the default the other way.  Tests that
assert the closed default start failing, and tests whose own fixture installs a licence error out
instead, because the slot refuses a second licence from the other channel rather than
overwriting it.  With pytest-randomly shuffling order, the same bug surfaces at a different
place on every run.

Resetting before AND after each test is deliberate: "after" contains the damage a test does, and
"before" protects the suite from anything that installed a licence outside a test (module import,
a collection-time helper, or a test that died hard enough to skip its own teardown).
"""

import os
import tempfile

import pytest

from operation_love import supervisor as _supervisor
from operation_love import targeting_policy as _targeting_policy

# --- one native thread per xdist worker ------------------------------------------------------
# Measured 2026-08-22, on the ~3820 tests that existed then: 60 tests were 61% of the suite's
# 19m33s serial wall clock, and a cProfile of the single slowest one (28.1s) puts 13.4s inside
# cv2.matchTemplate called from item_index.build_item_index / frameshift.estimate_shift --
# production code, real 1080x2400 frames.  That 60-tests/61% concentration and the 19m33s
# serial figure have NOT been re-measured since and are quoted here only as dated evidence for
# why OpenCV thread-pinning matters at all.  Re-measured 2026-09-02: the suite has grown to
# 4549 tests (4544 passed / 5 skipped) and the parallelised run (see [tool.pytest.ini_options]
# addopts) completes in 259s wall clock under the default -n auto.  OpenCV defaults to one
# thread per CORE (12 here), which is the right default for a serial run and exactly wrong once
# xdist gives every worker its own process: N workers x N cores is an N-fold oversubscription
# whose context switching can make the parallel run slower than the serial one.  Under xdist
# the parallelism comes from the workers, so each worker takes one thread; run serially
# (`-n0`, or a bare file) and OpenCV keeps its own default untouched.
#
# The env vars are set rather than called because OpenMP/MKL read them at import time and torch,
# numpy and onnxruntime are imported by test modules, i.e. AFTER this conftest but BEFORE the
# first test.  Setting them here is early enough; calling a runtime setter would not be.
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_var, "1")
    try:
        import cv2
    except ImportError:      # the cv2-free install still runs the non-vision tests
        pass
    else:
        cv2.setNumThreads(1)


@pytest.fixture(autouse=True)
def _still_photo_readiness_is_never_inherited():
    """Numbering readiness is process-global: never let one test license the next one.

    Uses the module's own ``_reset_installed_still_photo_bound_for_tests`` rather than the public
    ``clear_installed_still_photo_bound``.  Both do the same thing (the former is a one-line
    delegate), but the module publishes the public clear as part of the RUN lifecycle --
    ``config.validate()`` calls it to reset the slot before installing -- and publishes the
    ``_for_tests`` name for exactly this purpose: "drop the process-local licence so one test
    cannot license another".  Keeping test teardown on the test-named entry point means a grep
    for it still finds every place the suite resets readiness, and it matches the identical
    per-file fixtures this one now backstops.
    """
    _targeting_policy._reset_installed_still_photo_bound_for_tests()
    yield
    _targeting_policy._reset_installed_still_photo_bound_for_tests()


@pytest.fixture(scope="session", autouse=True)
def _machine_global_state_is_never_the_operators(tmp_path_factory):
    """Redirect the two paths in this codebase that are deliberately MACHINE-GLOBAL, so the
    test suite can never collide with the operator's real, running bot.

    Both paths exist to enforce "only one Operation Love run may own the phone / a Hinge
    OBSERVE input stream at a time", and they do that by being the SAME path for every
    process on the machine, on purpose:

      * ``supervisor._ANDROID_LOCK_ROOT`` (``~/.operation-love/locks``) backs
        ``_AndroidDeviceLock`` -- the lock that stops two Android runs from ever driving the
        one physical Pixel at once (see supervisor.py's own comments on ``_android_lock_path``
        for the incident that motivated making it a single, serial-independent file).

      * ``tempfile.gettempdir()`` backs ``HingeDriver._observe_input_lease_key()`` -- the OS
        temp directory is the SAME directory for every process on the machine (that is the
        whole point of "the" temp directory), and the lease file's name is a stable hash of
        ``app:serial``, not a PID or anything else per-process, so that a hub run and a
        separately launched controller sharing one real device agree on one file to contend
        over.

    Production must keep BOTH of these machine-global: narrowing either one (e.g. making the
    temp dir or the lock root per-process) would defeat the exclusion they exist to provide,
    letting two real runs fight over one phone again. So this fixture does not change
    production behaviour at all -- it only ever runs inside the test session, and only
    redirects where the SAME two names point to, not what they mean.

    Without this redirect, pytest workers ARE separate processes that all resolve the same
    real paths the operator's live bot would use: tests using fixed fake serials collide with
    each other under xdist (root cause A -- two workers computing the identical
    ``operation-love-observe-<hash>.lock`` path and genuinely contending over it via
    ``fcntl.flock``), and any test that drives ``holding_the_device()`` without its own
    per-test monkeypatch acquires the OPERATOR'S REAL
    ``~/.operation-love/locks/android-device.lock`` (root cause B) -- which is not merely a
    test-isolation bug, it is the suite momentarily holding the exact lock that exists to
    guarantee a live run never shares the phone with anything else.

    Order matters, and is pinned here rather than left to be "obviously fine" later:
    ``tmp_path_factory.getbasetemp()`` is resolved FIRST, before ``tempfile.tempdir`` is
    reassigned, so pytest computes its own basetemp using the REAL system temp directory (its
    normal behaviour) instead of nesting inside the machine-global directory we are about to
    redirect out from under it. Doing this the other way around -- patching
    ``tempfile.tempdir`` and THEN calling ``getbasetemp()`` -- would make pytest resolve its
    basetemp inside our own redirected root, which happens to still work but only by accident
    and stops being true the moment either implementation detail shifts.

    This also composes with xdist for free: ``getbasetemp()`` is already per-WORKER under
    xdist (``.../pytest-of-<user>/pytest-N/popen-gwK/``), so redirecting both machine-global
    paths underneath it gives every worker its own isolated "machine" -- no two workers, and
    no worker and the operator's real bot, can ever compute the same lock path. Run serially
    (no xdist), it still does its job: it isolates the whole suite from the operator's home
    directory for the run's entire session.

    Session-scoped and autouse: function/class-scoped fixtures that need to assert on the REAL
    production default (see test_supervisor.py / test_hinge_observe.py) monkeypatch the
    attribute back for the one assertion that needs it and let mp.undo() here restore the
    redirected value afterwards, same as any other monkeypatch layering.
    """
    base = tmp_path_factory.getbasetemp()      # resolve pytest's OWN basetemp FIRST -- see above
    root = base / "machine-global"
    (root / "locks").mkdir(parents=True, exist_ok=True)
    mp = pytest.MonkeyPatch()
    mp.setattr(_supervisor, "_ANDROID_LOCK_ROOT", root / "locks")
    mp.setattr(tempfile, "tempdir", str(root))
    yield
    mp.undo()
