"""The platform registry is the guard that keeps an unrunnable platform from ever
constructing a driver, so its refusals are load-bearing safety behaviour, not cosmetics.
"""
import pytest

from operation_love import platforms


@pytest.fixture(autouse=True)
def _restore_registry():
    """_apply_calibration() mutates module state; put it back so tests stay independent."""
    saved = platforms._PLATFORMS
    yield
    platforms._PLATFORMS = saved
    platforms._BY_APP = {p.app: p for p in saved}


def test_kinds_are_ordered_app_based_before_web_based():
    # The hub renders kinds in this order. App-based first because it is the only one
    # with a live target -- leading with the dead path would be a strange front door.
    assert platforms.kinds() == ("android", "web")
    assert platforms.KIND_LABELS["android"] == "App-based"
    assert platforms.KIND_LABELS["web"] == "Web-based"


def test_both_dating_apps_are_android_targets_now():
    # Bumble discontinued its web app in Aug 2026, so it is an Android target beside
    # Hinge. If this ever flips back, the two-tier picker's whole premise changes.
    assert platforms.get("hinge").kind == platforms.KIND_ANDROID
    assert platforms.get("bumble").kind == platforms.KIND_ANDROID
    assert platforms.get("bumble_web").kind == platforms.KIND_WEB


def test_web_platform_is_unavailable_with_the_additional_work_reason():
    reason = platforms.unavailable_reason("bumble_web")
    assert reason and "Additional work needed to get this to run" in reason
    # And it explains itself rather than just refusing.
    assert "no live target" in reason


def test_no_storage_bucket_field_pretends_history_is_pooled():
    # There WAS a store_key/bucket here, claiming Bumble-web and Bumble-app history pooled
    # under one id. Nothing outside this module ever read it: what actually reaches the
    # store is the raw registry id, because supervisor passes `app` straight into Worker,
    # which passes self.app into add_label/record_decision/record_profile/count_today.
    #
    # This test exists so the field cannot come back as an unread property. A property that
    # merely LOOKS like it groups history is worse than none -- it invites the belief that
    # daily rate-limit counting already pools across a dating app's transports when it does
    # not. Re-adding pooling means wiring it at the store boundary, and this test failing is
    # the prompt to do that rather than to reintroduce a decorative field.
    assert not hasattr(platforms.get("bumble_web"), "bucket")
    assert not hasattr(platforms.get("bumble_web"), "store_key")


def test_only_android_platforms_hold_the_device():
    assert platforms.get("hinge").exclusive_resource == platforms.RESOURCE_ANDROID_DEVICE
    assert platforms.get("bumble").exclusive_resource == platforms.RESOURCE_ANDROID_DEVICE
    assert platforms.get("bumble_web").exclusive_resource is None


def test_two_android_platforms_cannot_run_together():
    # The constraint that actually bites now both apps live on one handset: Android
    # foregrounds a single app, screencap captures whatever is on top, and the virtual
    # touchscreen delivers to whatever holds focus. Parallel is impossible, not merely
    # unwise -- so this must be a refusal, not a warning.
    platforms._apply_calibration({"bumble": True})   # even fully calibrated
    msg = platforms.check_runnable(["hinge", "bumble"])
    assert msg and "one app in the foreground" in msg
    assert platforms.check_runnable(["hinge"]) is None
    assert platforms.check_runnable(["bumble"]) is None


def test_unavailable_platform_is_refused_before_the_pairing_check():
    # An uncalibrated or dead platform must be rejected on its own merits; reporting the
    # device-contention message instead would send you chasing the wrong problem.
    msg = platforms.check_runnable(["bumble_web"])
    assert msg and "Additional work needed" in msg


def test_empty_and_unknown_selections_are_refused():
    assert platforms.check_runnable([]) == "Select a platform to run."
    msg = platforms.check_runnable(["tinder"])
    assert msg and "Unknown app 'tinder'" in msg
    with pytest.raises(ValueError, match="Unknown app 'tinder'"):
        platforms.get("tinder")


def test_uncalibrated_android_platform_fails_closed():
    # This is the guard that stops placeholder coordinates firing real touches at guessed
    # points on a real account. Default must be refusal.
    platforms._apply_calibration({"bumble": False})
    msg = platforms.check_runnable(["bumble"])
    assert msg and "not calibrated" in msg
    assert platforms.unavailable_reason("hinge") is None


def test_calibration_flips_availability_and_clears_the_reason():
    original = platforms.unavailable_reason("bumble")
    platforms._apply_calibration({"bumble": True})
    assert platforms.get("bumble").available is True
    assert platforms.unavailable_reason("bumble") is None
    platforms._apply_calibration({"bumble": False})
    # Asserting the reason is merely non-None was too weak: rebuilding the entry from the
    # already-mutated one copied the cleared None forward, so a round trip permanently lost
    # the specific "placeholder coordinates" explanation and the hub degraded to a generic
    # "Bumble is not available." -- which a non-None assertion happily accepts. Pin the
    # actual text.
    assert platforms.unavailable_reason("bumble") == original
    assert "not calibrated" in platforms.unavailable_reason("bumble")


def test_calibration_never_changes_which_platforms_exist():
    # KNOWN_APPS is assigned once so `from platforms import KNOWN_APPS` cannot go stale.
    before = set(platforms.KNOWN_APPS)
    platforms._apply_calibration({"bumble": True})
    assert set(platforms.KNOWN_APPS) == before
    assert set(platforms._BY_APP) == before


def test_calibration_for_an_unregistered_app_is_a_loud_error():
    # A spec whose id was never registered would otherwise be silently ignored, leaving
    # a driver that thinks it is calibrated and a registry that never lets it run.
    with pytest.raises(ValueError, match="unregistered app"):
        platforms._apply_calibration({"tinder": True})


def test_every_unavailable_platform_explains_itself():
    # An unavailable platform with no reason would surface in the hub as a bare refusal.
    for p in platforms.all_platforms():
        if not p.available:
            assert platforms.unavailable_reason(p.app), f"{p.app} refuses without saying why"


def test_for_kind_matches_the_registry_grouping():
    android = {p.app for p in platforms.for_kind(platforms.KIND_ANDROID)}
    web = {p.app for p in platforms.for_kind(platforms.KIND_WEB)}
    assert android == {"hinge", "bumble"}
    assert web == {"bumble_web"}
    assert android | web == set(platforms.KNOWN_APPS)


def test_config_validation_allows_configuring_an_uncalibrated_platform():
    # You must be able to write Bumble's coordinates into config.yaml BEFORE Bumble is
    # calibrated -- that is how it gets calibrated. Availability changes without the
    # config file changing, so it cannot be a config-load-time error. check_selection()
    # is the structural check config uses; check_runnable() is the start-time gate.
    platforms._apply_calibration({"bumble": False})
    assert platforms.check_selection(["bumble"]) is None
    assert platforms.check_selection(["bumble_web"]) is None
    assert platforms.check_runnable(["bumble"]) is not None
    assert platforms.check_runnable(["bumble_web"]) is not None


def test_check_selection_still_rejects_structural_problems():
    # It relaxes availability, not coherence: unknown ids and two-apps-on-one-phone are
    # wrong no matter what the world looks like today.
    assert platforms.check_selection([]) == "Select a platform to run."
    assert "Unknown app 'tinder'" in platforms.check_selection(["tinder"])
    assert "one app in the foreground" in platforms.check_selection(["hinge", "bumble"])


def test_calibration_reaches_the_registry_in_a_FRESH_process():
    """The spec's `calibrated` flag must actually drive availability in a real run.

    This has to be a SUBPROCESS. Calibration used to be applied only as an import side
    effect of `operation_love.drivers.android`, and the one production import of that
    package sits inside make_driver()'s bumble branch -- which is reached only AFTER
    check_runnable() has already decided. So in a real process the registry answered from
    its hand-written literals and the specs were never consulted; flipping a spec's
    `calibrated` flag, the documented way to enable a platform, changed nothing.

    Running in-process cannot catch it: tests/test_android_spec.py imports the driver
    package at collection time, which reconciles the registry before any test body runs and
    hides the bug for the entire session. Hence a clean interpreter.
    """
    import subprocess
    import sys
    probe = (
        "import sys\n"
        "from operation_love import platforms\n"
        # Nothing has imported the driver package yet -- exactly a fresh supervisor/hub start.
        "assert 'operation_love.drivers.android' not in sys.modules\n"
        "platforms.check_runnable(['hinge'])\n"
        # Querying the registry must itself pull the specs in.
        "assert 'operation_love.drivers.android' in sys.modules, 'specs never consulted'\n"
        "from operation_love.drivers.android.bumble import BUMBLE_SPEC\n"
        "from operation_love.drivers.hinge import HINGE_SPEC\n"
        # And availability must MATCH the specs, not this module's literals.
        "assert platforms.get('bumble').available == BUMBLE_SPEC.calibrated\n"
        "assert platforms.get('hinge').available == HINGE_SPEC.calibrated\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"fresh-process calibration probe failed:\n{r.stdout}\n{r.stderr}"
