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

import pytest

from operation_love import targeting_policy as _targeting_policy


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
